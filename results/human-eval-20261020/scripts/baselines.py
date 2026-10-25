"""The two RQ3 baselines, compared with AskGLP1 (weightloss.agent, agent-0.8.4) in the blind human evaluation.

B1  LLM only: the same model with no tools and no data; it answers from its own knowledge.
B2  Text retrieval: the same model with one search tool over the 258,954 source records the knowledge graph was built
    from (build_index.py: SQLite FTS5, BM25 ranking). The tool finds records by keywords, reports how many records match
    on each platform and shows the best-ranked records. It has no FAERS data and nothing the pipeline derived.

Both use the agent's model, reasoning effort and request settings, its medical-advice policy and its answer-type rule
(copied from agent.INSTRUCTIONS, so they cannot drift), and a strict JSON final answer. The agent's verifier is part of
AskGLP1 and is not given to the baselines. Every run appends a full trace (request bodies, tool inputs and outputs, raw
responses, usage, cost; never the API key) to a JSONL file.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from weightloss.agent import agent
from weightloss.extraction.structured import (ExtractionAnomaly, compact_json, final_output_text, mask, parse_json_object,
                                              sha256_text, strict_json_schema, token_split, validate_model)

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / '.cache/human-eval-20261020/records.sqlite'
VERSION = 'baselines-1.0.1'
MODEL, EFFORT = agent.MODEL, agent.EFFORT
MAX_TOOL_ROUNDS = agent.MAX_TOOL_ROUNDS
MAX_LIMIT = 20
FULL_TEXT_CHARS = 700     # records up to this length are shown whole; longer ones as a snippet around the matches
SNIPPET_TOKENS = 64

DRUGS = ('GLP-1 receptor agonists (ingredients semaglutide, tirzepatide, liraglutide, dulaglutide; brands Ozempic, Wegovy, '
         'Rybelsus, Mounjaro, Zepbound, Victoza, Saxenda, Trulicity)')
POLICY = agent.INSTRUCTIONS.split('Medical questions:', 1)[1].split('How to work:', 1)[0].strip()
ANSWER_TYPE = re.search(r'^- answer_type: .*$', agent.INSTRUCTIONS, re.M).group(0)
SHARED_FACTS = """- Reddit and WebMD are separate sources with different populations; report them separately and never add them together.
- Counts are records (one Reddit post or comment, or one WebMD review), not patients; writers are not verified patients. Percentages are shares of records that report something; they are not incidence, prevalence or risk, and no causal claim can be made.
- Groups of fewer than 10 records support no conclusion: say the data are insufficient. Groups of fewer than 30 must be called small in words."""
WORDING = ('- Wording: name the unit (Reddit posts or comments, WebMD reviews); write "Reddit writers report" or "on Reddit", '
           'never "Reddit reports"; never call writers patients. Never write record ids, user names or links in the answer.')
LENGTH = ('- Length and caveats: keep the answer under about 220 words. Put the general caveats (records are not patients; '
          'shares are not incidence, risk or causation; sources are separate) in limitations, not in the answer; keep in the '
          'answer only the caveats specific to this question.')


# ---------------------------------------------------------------------------------------------------- B1: LLM only
class B1Answer(BaseModel):
    answer: str = Field(description='Markdown answer to the question')
    answer_type: Literal['answered', 'limited', 'declined'] = Field(
        description='answered: the question is answered; limited: only part of it can be answered, or what is known is '
                    'insufficient; declined: medical advice or another question that cannot be answered')
    limitations: list[str] = Field(description='Short statements of the limits relevant to this answer')


B1_SCHEMA = strict_json_schema(B1Answer)

B1_INSTRUCTIONS = f"""You answer questions about patient-reported purposes and side effects of {DRUGS}. The questions are about what people write in Reddit posts and comments and in WebMD reviews, and about FAERS adverse event reports. You have no access to these data and no tools. Answer from your own knowledge of these drugs and of what people commonly report about them online and to FAERS, as a knowledgeable assistant would: give your best answer to every part of the question, including which drugs, groups, platforms or channels you expect to show more of something, and say how sure you are. Never invent exact counts, quotes or details of particular posts; give a figure only when you know it from a published source and name the source. In the answer-type rule below, "the data" means what you know.

Facts about such data (state the ones that matter for the question):
{SHARED_FACTS}
- On Reddit a writer's age and sex are known only when the writer states them; WebMD reviews carry a reviewer profile with age group, gender and the condition chosen from the site's list.
- FAERS holds spontaneous reports from patients, clinicians and manufacturers; FAERS, Reddit and WebMD are different reporting channels and their shares are not rates.

Medical questions: {POLICY}

Final answer (JSON):
- answer: concise markdown that answers the question directly, separated by platform where the question involves platforms.
{WORDING}
{ANSWER_TYPE}
{LENGTH}"""


# ---------------------------------------------------------------------------------------------------- B2: text retrieval
class B2Citation(BaseModel):
    id: str = Field(description='Tag id used in the answer, c1, c2, ...')
    call: str = Field(description='Label of the search output, e.g. t3 (its "call" field)')
    row: int = Field(description='Result row number in that output; 0 for the match counts of the search itself')


class B2Quote(BaseModel):
    call: str = Field(description='Label of the search output the quoted record comes from')
    row: int = Field(description='Result row number of the quoted record')
    text: str = Field(description='The quoted words, copied exactly from that record\'s text in the search output')


class B2Answer(BaseModel):
    answer: str = Field(description='Markdown answer; every number taken from a search output and every statement about what '
                                    'a record says is followed by its tag, e.g. 33 [c1]')
    answer_type: Literal['answered', 'limited', 'declined'] = Field(
        description='answered: the data answer the question; limited: only part of it can be answered from the data, or the '
                    'data are insufficient; declined: medical advice or another question the data cannot answer')
    citations: list[B2Citation]
    quotes: list[B2Quote] = Field(description='Every passage quoted in the answer, with the record it comes from; [] if none')
    limitations: list[str] = Field(description='Short statements of data limits relevant to this answer')


B2_SCHEMA = strict_json_schema(B2Answer)

SEARCH_TOOL = {
    'type': 'function', 'name': 'search_records', 'strict': True,
    'description': 'Keyword search over all 258,954 records (Reddit posts and comments, WebMD reviews). groups is a list of '
                   'lists of words or phrases: a record matches when it contains at least one phrase of every list, e.g. '
                   '[["tirzepatide", "mounjaro", "zepbound"], ["muscle cramp*", "leg cramp*", "charley horse*"]]. Matching '
                   'ignores case and accents and uses whole words; end a phrase with * to match word starts (nause* matches '
                   'nausea, nauseous, nauseated). The record metadata is searched too: the community (e.g. r/zepbound) for '
                   'Reddit, the reviewer profile (brand, ingredient, age group such as "age 45-54", gender, condition, '
                   'duration) for WebMD. Returns the number of matching records on each platform and the best-ranked records '
                   '(BM25), each with its metadata and its text (whole, or a snippet around the matches when long).',
    'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['groups', 'platform', 'limit'],
                   'properties': {
                       'groups': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}},
                                  'description': 'Lists of alternatives; every list must be matched by at least one of its '
                                                 'phrases'},
                       'platform': {'type': 'string', 'enum': ['reddit', 'webmd', 'any']},
                       'limit': {'type': 'integer', 'description': f'Records to show, 1 to {MAX_LIMIT}'}}}}

B2_INSTRUCTIONS = f"""You answer questions about patient-reported purposes and side effects of {DRUGS}. The data are 258,954 records: Reddit posts and comments from GLP-1 communities and WebMD reviews of the eight brands. You reach them only through the search_records tool. Use only search outputs for any statement about the data; never fill gaps with outside knowledge. You have no FAERS data.

Data facts (state the ones that matter for the question):
{SHARED_FACTS}
- Reddit records carry their community and date; a Reddit writer's age and sex are known only when the writer states them in the text. WebMD reviews carry a reviewer profile: brand, ingredient, age group, gender, the condition chosen from the site's list, and medication duration.
- A keyword match is not a report: a record that contains "nausea" may say the writer had none, describe someone else, or be about another drug. Read the shown records before you count, say how you counted, and call counts of matching records approximate.

Medical questions: {POLICY}

How to work:
1. Search with the words writers use: put a drug's ingredient and brand names in one list (tirzepatide, Mounjaro, Zepbound), put lay terms, synonyms and common misspellings of a symptom or reason in another, and search Reddit and WebMD separately when the question involves both.
2. For a count, use the number of matching records the search reports; for a share, also search the group it is a share of (for example all records that mention the drug), and give both counts. Do not compute other new numbers (no sums across platforms, no differences or ratios).
3. For what people say or for examples, quote from the shown records only.
4. Stop searching once you have what you need, or when the data clearly cannot answer the question.

Every search output has a "call" label and numbered result rows. Final answer (JSON):
- answer: concise markdown that answers the question directly, separated by platform. Every number taken from a search output is written as it appears there and followed by a tag such as [c1]; a statement about what a record says is followed by the tag of that record.
- citations: one entry per tag: id, call label, row number (0 for the match counts of the search).
- quotes: every passage you put in quotation marks, with the call label and row of the record and the exact words; [] if none. Quote a short clause (about 5 to 15 words) that shows the experience asked about.
{WORDING}
{ANSWER_TYPE}
{LENGTH}"""

INSTRUCTIONS = {'b1': B1_INSTRUCTIONS, 'b2': B2_INSTRUCTIONS}
SCHEMAS = {'b1': (B1Answer, B1_SCHEMA), 'b2': (B2Answer, B2_SCHEMA)}
TOOLS = {'b1': [], 'b2': [SEARCH_TOOL]}


# ---------------------------------------------------------------------------------------------------- search
_LOCAL = threading.local()


def _db():
    if getattr(_LOCAL, 'db', None) is None:
        _LOCAL.db = sqlite3.connect(f'file:{INDEX}?mode=ro', uri=True, check_same_thread=False)
    return _LOCAL.db


def _phrase(text):
    prefix = text.strip().endswith('*')
    words = re.findall(r"[^\W_]+", text.lower())
    if not words:
        return None
    return '"' + ' '.join(words) + '"' + (' *' if prefix else '')


def match_expression(groups):
    parts = []
    for group in groups:
        alts = [p for p in map(_phrase, group) if p]
        if alts:
            parts.append(alts[0] if len(alts) == 1 else '(' + ' OR '.join(alts) + ')')
    if not parts:
        raise ValueError('give at least one word or phrase')
    return ' AND '.join(parts)


def search(groups, platform, limit):
    expr = match_expression(groups)
    limit = max(1, min(int(limit), MAX_LIMIT))
    where = 'records_fts MATCH ?' + ('' if platform == 'any' else ' AND r.platform = ?')
    args = (expr,) if platform == 'any' else (expr, platform)
    db = _db()
    totals = dict(db.execute(f'SELECT r.platform, count(*) FROM records_fts f JOIN records r ON r.rid = f.rowid WHERE {where} '
                             'GROUP BY r.platform', args).fetchall())
    rows = db.execute(f"SELECT r.record_key, r.platform, r.kind, r.date, r.meta, r.text, "
                      f"snippet(records_fts, 0, '', '', ' … ', {SNIPPET_TOKENS}) FROM records_fts f JOIN records r "
                      f"ON r.rid = f.rowid WHERE {where} ORDER BY rank LIMIT ?", args + (limit,)).fetchall()
    results = []
    for i, (key, plat, kind, date, meta, text, snip) in enumerate(rows, 1):
        results.append({'row': i, 'record': f'{plat} {kind}', 'date': date,
                        ('community' if plat == 'reddit' else 'profile'): meta,
                        'text': text if len(text) <= FULL_TEXT_CHARS else snip, '_record_key': key})
    return {'matching_records': {'reddit': totals.get('reddit', 0), 'webmd': totals.get('webmd', 0)},
            'shown': len(results), 'results': results}


def public(output):
    """The search output as the model sees it (record keys stay in the trace only)."""
    return {**output, 'results': [{k: v for k, v in r.items() if not k.startswith('_')} for r in output.get('results', [])]}


# ---------------------------------------------------------------------------------------------------- run
def request_body(system, items, allow_tools, effort=EFFORT, model=MODEL):
    body = {'model': model, 'instructions': INSTRUCTIONS[system], 'input': items,
            'text': {'format': {'type': 'json_schema', 'name': f'{system}_answer', 'strict': True, 'schema': SCHEMAS[system][1]}},
            'reasoning': {'effort': effort}, 'include': ['reasoning.encrypted_content'], 'store': False,
            'max_output_tokens': agent.MAX_OUTPUT_TOKENS}
    if TOOLS[system]:
        body.update(tools=TOOLS[system], tool_choice='auto' if allow_tools else 'none', parallel_tool_calls=True)
    return body


def run(question, system, *, client=None, trace_path=None, label=None, budget_usd=agent.BUDGET_USD, effort=EFFORT, model=MODEL):
    """Answer one question with baseline 'b1' or 'b2'; returns a result dict shaped like agent.run's."""
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    items = [{'role': 'user', 'content': f'Question: {question}'}]
    name = f'{VERSION}:{system}'
    trace = {'run_id': run_id, 'label': label, 'system': name, 'started_utc': started.isoformat(timespec='seconds'),
             'question': question, 'model': model, 'reasoning_effort': effort,
             'instructions_sha256': sha256_text(INSTRUCTIONS[system]), 'tools_sha256': sha256_text(compact_json(TOOLS[system])),
             'schema_sha256': sha256_text(compact_json(SCHEMAS[system][1])), 'instructions': INSTRUCTIONS[system],
             'responses': [], 'tool_calls': []}
    usage_total, cost, rounds, stop_reason, final = collections.Counter(), 0.0, 0, None, None
    try:
        if client is None:
            agent.load_key()
            from openai import OpenAI
            client = OpenAI(max_retries=5, timeout=300)
        while True:
            if cost >= budget_usd:
                stop_reason = f'budget reached (${cost:.4f} of ${budget_usd})'
                break
            if len(trace['responses']) >= MAX_TOOL_ROUNDS + 3:
                stop_reason = 'response cap reached'
                break
            t0 = time.time()
            try:
                raw = client.responses.create(**request_body(system, items, rounds < MAX_TOOL_ROUNDS, effort, model)).model_dump()
            except Exception as error:
                stop_reason = f'transport_error: {type(error).__name__}: {mask(error)[:300]}'
                break
            usd = agent.cost_usd(model, raw.get('usage') or {}) or 0.0
            cost += usd
            for k, v in token_split(raw.get('usage') or {}).items():
                usage_total[k] += v
            trace['responses'].append({'seconds': round(time.time() - t0, 2), 'usd': usd, 'raw': raw})
            calls = [i for i in raw.get('output') or [] if i.get('type') == 'function_call']
            if calls and raw.get('status') == 'completed':
                rounds += 1
                items.extend(agent._strip_none(i) for i in raw['output'])
                for c in calls:
                    label_ = f't{len(trace["tool_calls"]) + 1}'
                    t1 = time.time()
                    try:
                        args = json.loads(c['arguments'])
                        out = {'call': label_, **search(args['groups'], args['platform'], args['limit'])}
                        error = None
                    except (json.JSONDecodeError, KeyError, ValueError, sqlite3.Error) as exc:
                        args, out, error = c.get('arguments'), {'call': label_, 'error': f'{type(exc).__name__}: {exc}'[:300]}, str(exc)
                    trace['tool_calls'].append({'call': label_, 'tool': 'search_records', 'arguments': args, 'output': out,
                                                'seconds': round(time.time() - t1, 2), 'error': error})
                    items.append({'type': 'function_call_output', 'call_id': c['call_id'],
                                  'output': compact_json(public(out))})
                continue
            try:
                model_cls = SCHEMAS[system][0]
                parsed = validate_model(model_cls, parse_json_object(final_output_text(raw)))
                final = parsed.model_dump()
                final.setdefault('citations', [])
                final.setdefault('quotes', [])
            except ExtractionAnomaly as error:
                stop_reason = f'anomaly: {error.kind}'
            break
    finally:
        result = {'run_id': run_id, 'label': label, 'question': question, 'system': name, 'verified': None,
                  'answer': final, 'problems': [] if final else [stop_reason or 'no answer'], 'warnings': [],
                  'repair_used': False, 'stop_reason': stop_reason, 'tool_rounds': rounds,
                  'tool_calls': [{'call': c['call'], 'tool': c['tool'], 'arguments': c['arguments'], 'seconds': c['seconds'],
                                  'error': c['error']} for c in trace['tool_calls']],
                  'usage': dict(usage_total), 'cost_usd': round(cost, 6)}
        trace.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), result=result)
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            with agent.TRACE_LOCK, open(trace_path, 'a') as f:
                f.write(mask(json.dumps(trace, ensure_ascii=False, default=str)) + '\n')
    return result
