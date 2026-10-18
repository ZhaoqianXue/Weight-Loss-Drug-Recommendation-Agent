"""Evidence-grounded question answering over the knowledge graph (project document Section 3.7).

One Responses API conversation per question: strict function tools over the graph tools (tools.py: link_terms,
query_graph, get_evidence, compare_faers on the local Neo4j graph), reasoning items passed back encrypted (store=False),
and a strict JSON final answer whose numbers cite tool-output rows and whose quotes cite returned source sentences.
A deterministic verifier checks every citation and quote against the tool outputs; on failure the model gets one repair
round with the verifier messages, and the result is marked verified true or false. Every run appends a full trace
(request bodies, tool inputs and outputs, raw responses, usage, cost; never the API key) to a JSONL file.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from weightloss.extraction.structured import (ExtractionAnomaly, compact_json, cost_usd, final_output_text, mask,
                                              parse_json_object, sha256_text, strict_json_schema, token_split, validate_model)
from weightloss.graph import neo4j_db

from . import VERSION
from .tools import TOOL_ARGS, KnowledgeGraph, QueryError

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / '.cache/agent-runs'
TRACE = RUN_DIR / 'traces.jsonl'
MODEL, EFFORT = 'gpt-6-luna', 'medium'
MAX_TOOL_ROUNDS = 10          # responses that call tools before the answer is forced
REPAIR_TOOL_ROUNDS = 2        # extra tool rounds allowed after a failed verification
MAX_OUTPUT_TOKENS = 16000
BUDGET_USD = 0.15             # per question; the loop stops before a request once this is spent
TRACE_LOCK = threading.Lock()  # concurrent runs append whole trace lines

TOOL_DESCRIPTIONS = {
    'link_terms': 'Entity linking. Find the graph nodes for a condition, goal or symptom: MedDRA PT nodes whose name or whose '
                  'statement wording matches the words, with record counts per platform, example wordings and FAERS report '
                  'counts, plus the MedDRA groups (HLT, HLGT, SOC) that match or sit above the best PTs. Call it before using '
                  'any code.',
    'query_graph': 'Graph pattern query over the patient statements. Match statements of the target role by drug, platform, '
                   'writer age group and sex, statements of the same drug in the same record (require_purposes / '
                   'require_side_effects) and target concepts (a PT or a whole MedDRA group), then count records: a concept '
                   'counts once per record; the denominator of a row is the number of records with at least one target '
                   'statement in the filtered universe and the same population group. Platforms are always split. Summary rows '
                   'give the universe per platform and how many records an age or sex filter excluded because the value was missing.',
    'get_evidence': 'Example statements for the same pattern: statement id, record, drug, PT and its SOC, wording, writer age and '
                    'sex, side-effect details and the cited source sentences. Fixed pseudo-random order, one statement per record.',
    'compare_faers': 'The same side-effect concepts and drug scopes in the validated FAERS reports, side by side with the '
                     'Reddit and WebMD records under the RQ2 comparison rules (window from FDA approval to 2026-06-30, '
                     'clinical-event PTs only). Hospitalization and the action taken with the drug (withdrawn = the drug was stopped; '
                     'dose reduced) are outcomes: set include_outcomes=true for them and never look them up as PTs (Drug withdrawal '
                     'syndrome is a different, clinical event). Without targets it ranks PTs.',
}
TOOLS = [{'type': 'function', 'name': name, 'description': TOOL_DESCRIPTIONS[name], 'strict': True,
          'parameters': strict_json_schema(model)} for name, model in TOOL_ARGS.items()]


# ---------------------------------------------------------------------------------------------------- final answer
class Citation(BaseModel):
    id: str = Field(description='Tag id used in the answer, c1, c2, ...')
    call: str = Field(description='Label of the tool output, e.g. t3 (the "call" field of the output)')
    row: int = Field(description='The "row" number inside that output')
    field: str = Field(description='Field name in that row, e.g. records, denominator, percent, faers_reports')
    value: str = Field(description='The field value exactly as in the row, written as text')


class Quote(BaseModel):
    edge_id: str = Field(description='edge_id of the statement, from a get_evidence output')
    text: str = Field(description='The quoted words, copied exactly from that statement\'s evidence sentences')


class FinalAnswer(BaseModel):
    answer: str = Field(description='Markdown answer; every number from a tool output is followed by its tag, e.g. 8 [c1]')
    answer_type: Literal['answered', 'limited', 'declined'] = Field(
        description='answered: the data answer the question; limited: only part of it can be answered from the data, or the '
                    'data are insufficient; declined: medical advice or another question the data cannot answer')
    citations: list[Citation]
    quotes: list[Quote] = Field(description='Every passage quoted in the answer, with its statement; [] if none')
    limitations: list[str] = Field(description='Short statements of data limits relevant to this answer')


FINAL_SCHEMA = strict_json_schema(FinalAnswer)

INSTRUCTIONS = """You answer questions about patient-reported purposes and side effects of GLP-1 receptor agonists (ingredients semaglutide, tirzepatide, liraglutide, dulaglutide; brands Ozempic, Wegovy, Rybelsus, Mounjaro, Zepbound, Victoza, Saxenda, Trulicity). The data are a knowledge graph built from Reddit posts and comments and WebMD reviews, linked to the MedDRA 29.1 hierarchy and to validated FAERS reports, and you reach it only through the tools. Use only tool outputs for any statement about the data; never fill gaps with outside knowledge.

The graph: Drug nodes (ingredient and brand), Purpose and SideEffect nodes at the MedDRA Preferred Term (PT) level, each PT on its MedDRA primary path PT -> HLT -> HLGT -> SOC, and one statement edge per thing a writer said: Drug -USED_FOR-> Purpose and Drug -SIDE_EFFECT-> SideEffect, carrying the record, the writer's age group and sex, the wording, the cited sentences and, for side effects, dose, duration, intensity, impact and action taken. FAERS reports link to the same Drug and SideEffect nodes.

Data facts (state the ones that matter for the question):
- Reddit and WebMD are separate sources with different populations; report them separately and never add them together.
- Counts are records (one Reddit post or comment, or one WebMD review), not patients; writers are not verified patients. A concept counts once per record. Percentages are shares of records that report something; they are not incidence, prevalence or risk, and no causal claim can be made.
- Side effects are symptoms the writer reports having and links to the drug; purposes are the writer's stated reasons for using it. Most WebMD purposes are the condition the reviewer chose from the site's list in the profile, while Reddit purposes all come from free text, so purpose shares measure different things on the two platforms. A condition such as "people who take it for X" means a purpose statement for the same drug in the same record.
- Writer age group and sex/gender: on Reddit they are known only when writers state them about themselves, so those records are a self-disclosed subset; on WebMD they come from the reviewer profile. Missing values are "not stated". An age or sex filter drops records with missing values.
- Ingredient-level results include references that name no drug but come from a single-ingredient Reddit community (ingredient_basis "community"); named_drugs_only=true restricts to named drugs. Keep named_drugs_only false unless the question asks to count only posts that name the drug, or you run it as a separately labelled robustness check; whenever you set it to true, say in the answer that community-assigned records were left out. A brand filter already counts named brands only. Brand-level results include only brands named in the text or the WebMD profile; brand "unspecified" is never a brand.
- The "[uncoded ...]" node collects mentions without an adequate MedDRA term; it is not a PT and must not be ranked as one. When it holds relevant wordings, say that PT-based counts miss them.
- FAERS numbers are spontaneous reports (validated, primary configuration) from patients, clinicians and manufacturers. Describe FAERS, Reddit and WebMD shares side by side as different reporting channels; never compute ratios or differences between them and never call any of them a rate.
- Rows flagged sparse (denominator below 10) and empty groups support no conclusion: say the data are insufficient. Rows flagged small (denominator below 30) must be called small in words.
- Comparisons: call one share higher, lower or different from another (between groups, drugs, platforms or FAERS) only when their 95% intervals (ci95_low to ci95_high) do not overlap; when they overlap, say the shares are similar or that the data show no clear difference. query_graph lists the population groups that clearly differ in clear_differences, and compare_faers rows say faers_vs_reddit, faers_vs_webmd and reddit_vs_webmd; base every comparison word on these, never on your own reading of the intervals. Answer yes/no comparison questions by this rule, and give the intervals for the shares a conclusion rests on.

Medical questions: you inform about what writers report; you do not give medical advice. If the question asks what someone should do (take, stop, switch, dose or combine a drug, how to treat a symptom, whether something is safe for them) or asks about risk, incidence or causation, do not advise or estimate. Set answer_type "declined" when nothing in the question can be answered from the data, and say plainly that you cannot give medical advice and that a doctor or pharmacist can help; for severe symptoms (for example severe abdominal pain, fainting, signs of an allergic reaction) say to seek medical care promptly. Set answer_type "limited" when part of the question can be answered: answer that part from the data and state what cannot be answered. Never recommend a drug, dose or action.

How to work:
1. Concepts. Map every specific condition, goal or symptom in the question with link_terms (kind purpose or side_effect). Do not call link_terms for generic words such as "side effects", "adverse experiences", "complaints", "problems" or "issues", or for drug names (drugs are tool arguments): a question about side effects in general needs no target. Writers' wordings for one meaning are often coded to several PTs; for example a purpose "weight loss" sits on Weight decreased and Weight control, and "diabetes" on Diabetes mellitus, Type 2 diabetes mellitus and Diabetes mellitus management. Read every row link_terms returns (name, example wordings, matching-wording counts) and its related PTs (same MedDRA HLT, worded differently, such as an abbreviation or a more specific term), and give each node that expresses the asked meaning the same concept name in targets, require_purposes, require_side_effects or compare_faers targets, so a record counts once for the concept. For a broad category ("stomach problems", "mental health", "skin", "heart", "sleep problems"), make the MedDRA group that matches the category as a whole (level hlt, hlgt or soc) a member, and add any PT outside that group that clearly belongs; never count only some PTs of a category. Leave out rows whose matches are incidental or whose meaning is narrower or different, and when such a row is large say in the answer that it was not counted and why (code checks every large matching row). If the question names a subtype (for example type 2 diabetes), also call link_terms for the general term ("diabetes"); when the data hold an unspecified term for the same condition (Diabetes mellitus, Diabetes mellitus management), report it as its own concept next to the subtype; do not add it to the subtype or drop it. Never use the uncoded node as a target unless the question is about mentions without a term. Use only codes that link_terms returned, and name in the answer the PTs or group that each concept counts.
2. Use query_graph for counts. Pass both platforms unless the question names one. Use group_by for comparisons (for example ["sex"] or ["age_group"] with targets, ["sex", "pt"], ["ingredient"]). For "how often is X reported" pass targets with group_by []; add "pt" to group_by to show a concept's members. For "which side effects" use group_by ["pt"] (or ["soc"] for an overview). For questions about consequences ("led people to stop", "sent people to hospital", "severe"), use the actions, impacts or intensities filters, which make the denominator the records with such statements. When purposes are compared across platforms, you may add a purposes_from_text_only=true query as the like-for-like check.
3. Use get_evidence for a few example statements when they help (limit 3-5); use wording to find statements phrased a particular way. When the question asks what people say, write or describe, or asks for examples or their own words, you must call get_evidence and quote the returned sentences; counts alone do not answer it. Quote only words from the returned evidence sentences.
4. Use compare_faers when the question asks about FDA, FAERS or official reports. Pass the symptoms or groups the question names as targets; use rank_by (targets null) only when the question asks for the top or most common reactions. For hospitalization, stopping, withdrawal or dose reduction, set include_outcomes=true (FAERS "withdrawn" is the action taken with the drug, compared with writers who stopped it).
5. When the question involves age or sex, report for each platform how many records state the value out of all records (records_with_age_stated or records_with_sex_stated of records_before_age_sex_filter in the summary rows).
6. Stop calling tools once you have what you need, or when the data are clearly insufficient.

Every tool output has a "call" label (t0 is the graph overview given with the question) and numbered rows. Final answer (JSON):
- answer: concise markdown that answers the question directly, separated by platform, with missing or sparse population attributes called out. Give numbers, not just names: for every group and concept you report, state its records and denominator (and percent when useful). For long rankings, give the top few per group and say that more rows exist. Write PT and group names exactly as in the tool output; when a PT name may not tell a reader what writers said, add its common wording in parentheses, for example Malaise ("sick").
- Wording: name the unit (Reddit posts or comments, WebMD reviews) and phrase shares with their group, for example "Of 3,728 tirzepatide side-effect posts from writers who said they are women, 1,207 (32.4%) mention Nausea"; write "Reddit writers report" or "on Reddit", never "Reddit reports"; never call writers patients. Never write record or statement ids, user names or links in the answer.
- Every number taken from a tool output is written in digits, exactly as it appears in the row (counts of 1,000 or more with thousands separators), and is followed immediately by a tag such as [c1]; for "8 of 30 records (26.7%)" write 8 [c1] of 30 [c2] records (26.7% [c3]), never 8 of 30 records (26.7%) [c1][c2][c3] or 8 records (26.7% [c3]) [c1]. The same holds in tables: each cell number carries its own tag. Tags are for numbers only. Do not compute new numbers (no sums across platforms or groups, no differences or ratios).
- citations: one entry per tag: id, call label, row number, field name, and the value exactly as in that row.
- quotes: every passage you put in quotation marks, with the edge_id it comes from and the exact words; [] if none. Never put a [c] tag after a quote or a name; quotes are checked through this list.
- answer_type: "answered" whenever the data answer the question, even with caveats such as missing values, a self-disclosed subset or a few sparse groups; "limited" only when part of the question cannot be answered from the data at all (a risk, a cause, a cross-channel rate comparison, a field the graph lacks) or every relevant group is too sparse to say anything; "declined" for medical advice or when nothing can be answered.
- Length and caveats: keep the answer under about 220 words. Put the general caveats (records are not patients; shares are not incidence, risk or causation; sources are separate) in limitations, not in the answer; keep in the answer only the caveats specific to this question (sparse or small groups, missing values, a restriction you applied, concepts or rows you left out, uncoded wordings that matter, the WebMD profile list when comparing purposes)."""

REPAIR = ('Verification of your final answer failed:\n{problems}\n\nReturn a corrected final answer in the same JSON format. Cite only '
          'values present in tool outputs (call label, row number, field, value exactly as in the row), write each cited number '
          'exactly as cited immediately before its own tag (one tag per number, never several tags after one number; write '
          '"5851 [c1] records (36.7% [c3])", not "5851 records (36.7% [c3]) [c1]"), give '
          'every tag a citation, and quote only exact words from evidence '
          'sentences returned by get_evidence, listing each quote with its edge_id. You may call tools again if a needed number '
          'or sentence is missing.')


# ---------------------------------------------------------------------------------------------------- session and tools
class Session:
    """Executes tool calls against the graph and keeps every output by its call label. Keys starting with '_' (full source
    sentences for quote checking) stay in the session and are never sent to the model."""

    def __init__(self, graph):
        self.graph = graph
        self.outputs = {}
        self.calls = []
        self.next_call = 1

    def _store(self, label, name, arguments, output, seconds):
        out = collections.OrderedDict(call=label)
        out.update(output)
        self.outputs[label] = out
        self.calls.append({'call': label, 'tool': name, 'arguments': arguments, 'seconds': round(seconds, 3),
                           'output': public(out)})
        return out

    def overview(self):
        t0 = time.time()
        return self._store('t0', 'overview', {}, self.graph.overview(), time.time() - t0)

    def run_tool(self, name, arguments):
        label = f't{self.next_call}'
        self.next_call += 1
        t0 = time.time()
        try:
            if name not in TOOL_ARGS:
                raise QueryError(f'unknown tool {name}')
            args = validate_model(TOOL_ARGS[name], json.loads(arguments) if isinstance(arguments, str) else arguments).model_dump()
            output = getattr(self.graph, name)(**args)
        except (QueryError, ExtractionAnomaly, json.JSONDecodeError, TypeError) as error:
            output = {'tool': name, 'error': str(error)}
        return self._store(label, name, arguments, output, time.time() - t0)


def public(output):
    """The part of a tool output the model sees."""
    return collections.OrderedDict((k, v) for k, v in output.items() if not k.startswith('_'))


# ---------------------------------------------------------------------------------------------------- verifier
TAG = re.compile(r'\[(c\d+)\]')
NUMBER = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
NOT_DATA_NUMBERS = re.compile(r'\b95 ?% ?(?:CIs?|confidence|intervals?)\b|\b\d{2}[-–]\d{2}\b|\b75 or over\b|\btype [12]\b|\bMedDRA \d+(?:\.\d+)?|\b(?:19|20)\d{2}\b|\b\d{8}\b|'
                              r'\bt\d+\b|\bc\d+\b|\bGLP-1\b|\bPT\b|\[c\d+\]|\b\d+(?:\.\d+)? ?mg\b', re.I)
QUOTED = re.compile(r'["“]([^"”]{8,})["”]')
# Numbers that are not data counts: ages and age bands written out, ordinals, list sizes, numbered items.
EXEMPT_NUMBERS = re.compile(r'\b(?:aged?|ages)\s+\d{1,3}\b|\b\d{1,3}\s*(?:\+|-and-over|and over|or over|or older|years?\b|year-olds?|yo\b)|'
                            r'#\d+|\btop\s+\d+\b|\b\d+(?:st|nd|rd|th)\b', re.I)
NUMBER_WORDS = {w: str(i) for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                                                  'fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
NUMBER_WORD = re.compile(r'\b(' + '|'.join(NUMBER_WORDS) + r')\b', re.I)


def _number(text):
    text = str(text).strip().replace(',', '').rstrip('%').strip()
    try:
        return float(text)
    except ValueError:
        return None


def same_value(cited, actual):
    """A cited value (text) agrees with the row value; a number may be cited rounded to fewer decimals."""
    cited = str(cited).strip()
    if actual is None:
        return cited.lower() in ('null', 'none', '')
    if isinstance(actual, bool):
        return cited.lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        c = _number(cited)
        if c is None:
            return False
        if c == float(actual):
            return True
        bare = cited.replace(',', '').rstrip('%').strip()
        decimals = len(bare.split('.', 1)[1]) if '.' in bare else 0
        return round(float(actual), decimals) == c
    if isinstance(actual, list):
        return cited.casefold() in {json.dumps(actual).casefold(), ', '.join(map(str, actual)).casefold()} | {str(a).casefold() for a in actual}
    return cited.casefold() == str(actual).strip().casefold()


def _segment(text, pos):
    """The clause before position pos: back to the last line break, table cell, or sentence or clause end."""
    start = max(text.rfind('\n', 0, pos), text.rfind('|', 0, pos))
    for sep in ('. ', '; ', '? ', '! '):
        start = max(start, text.rfind(sep, 0, pos))
    return text[start + 1:pos]


def _rows(output):
    rows = {}
    for value in output.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get('row'), int):
                    rows[item['row']] = item
    return rows


def _norm(text):
    return re.sub(r'\s+', ' ', str(text).replace('’', "'").replace('‘', "'")).strip().casefold()


def uncited_numbers(text, question=''):
    """Problems for numbers in the answer that no citation tag follows before the next number on the same line. Numbers
    inside quotation marks, numbers repeated from the question, doses, age bands, years, codes and ordinals are exempt."""
    blank = lambda m: ' ' * len(m.group(0))
    masked = TAG.sub(blank, text)
    for pattern in (QUOTED, NOT_DATA_NUMBERS, EXEMPT_NUMBERS):
        masked = pattern.sub(blank, masked)
    asked = {_number(x) for x in NUMBER.findall(question or '')}
    tags = [m.start() for m in TAG.finditer(text)]
    numbers = list(NUMBER.finditer(masked))
    problems = []
    for i, m in enumerate(numbers):
        if _number(m.group()) in asked:
            continue
        limit = numbers[i + 1].start() if i + 1 < len(numbers) else len(text)
        cited = any(m.end() <= t < limit and '\n' not in text[m.end():t] and '|' not in text[m.end():t] for t in tags)
        if not cited:
            context = text[max(0, m.start() - 50):m.end() + 30].replace('\n', ' ')
            problems.append(f'the number {m.group()} has no citation tag after it ("...{context}..."); cite it or leave it out')
    return problems


COVER_MIN_RECORDS = 20   # a link_terms row this large (records whose wording matched) ...
COVER_SHARE = 0.10       # ... and at least this share of the best row's matching records must be counted or mentioned
GENERIC_TAIL = re.compile(r'\s+(?:mellitus|syndrome|disorders?|disease|nec)$')
RECORD_ID = re.compile(r'\bt[13]_[a-z0-9]{5,}\b|\breddit:t[13]_|\bwebmd:\d{5,}')


def concept_gaps(text, outputs):
    """Problems for large link_terms rows that no counted concept includes and the answer does not name. A link_terms
    output is checked once one of its nodes is counted somewhere (as a target, a required co-statement or a FAERS concept)."""
    counted = {'purpose': set(), 'side_effect': set()}
    groups = {'purpose': set(), 'side_effect': set()}
    for out in outputs.values():
        for kind, codes in (out.get('_counted') or {}).items():
            counted[kind] |= set(codes)
        rc = out.get('resolved_concepts')
        entries = []
        if isinstance(rc, dict):
            target = (out.get('filters') or {}).get('target')
            entries = [(target, x) for x in rc.get('targets', [])] + [('purpose', x) for x in rc.get('require_purposes', [])] + \
                      [('side_effect', x) for x in rc.get('require_side_effects', [])]
        elif isinstance(rc, list):
            entries = [('side_effect', x) for x in rc]
        for kind, x in entries:
            for m in x.get('members', []):
                if kind in groups and m.get('level') != 'pt':
                    groups[kind].add(m['code'])
    low = _norm(text)
    named = lambda name: any(v and v in low for v in {_norm(name), GENERIC_TAIL.sub('', _norm(name)).strip()})
    problems = []
    for label, out in outputs.items():
        if out.get('tool') != 'link_terms' or out.get('_kind') not in counted:
            continue
        kind = out['_kind']
        rows = [r for r in out.get('rows', []) if r.get('code') not in (None, 'uncoded')]
        used = any(r['code'] in counted[kind] for r in rows) or any(g.get('code') in groups[kind] for g in out.get('groups', []))
        if not used or not rows:
            continue
        size = lambda r: r.get('matching_wording_records_reddit', 0) + r.get('matching_wording_records_webmd', 0)
        top = max(size(r) for r in rows)
        for r in rows:
            m = size(r)
            if (m >= max(COVER_MIN_RECORDS, COVER_SHARE * top) and r['code'] not in counted[kind]
                    and not named(r['name'])):
                problems.append(f"link_terms \"{out.get('text')}\" ({label}) returned {r['name']} with {m} records whose wording "
                                f"matched, which no counted concept includes: give it the same concept name if it expresses the "
                                f"asked meaning, or say in the answer that {r['name']} was not counted and why")
    return problems


def leaked_ids(text, outputs):
    ids = {i for out in outputs.values() for i in out.get('_record_ids') or []}
    found = sorted({i for i in ids if i and i in text} | set(RECORD_ID.findall(text)))
    return [f'the answer shows record or statement ids ({", ".join(found[:3])}); remove them (ids belong only in the quotes list)'] if found else []


def verify(answer, outputs, question=''):
    """(problems, warnings) for a FinalAnswer against the tool outputs keyed by call label. Problems fail verification.
    A passage in quotation marks must be a listed quote, unless it repeats the question's words or a name or wording the tools
    returned (a PT, a MedDRA group, a statement's wording)."""
    problems, warnings = [], []
    cites = {}
    for c in answer.citations:
        if c.id in cites:
            problems.append(f'citation id {c.id} is used twice')
            continue
        cites[c.id] = c
        out = outputs.get(c.call)
        if out is None:
            problems.append(f'{c.id}: no tool output labelled {c.call!r}')
            continue
        row = _rows(out).get(c.row)
        if row is None:
            problems.append(f'{c.id}: {c.call} has no row {c.row}')
            continue
        if c.field not in row:
            problems.append(f'{c.id}: {c.call} row {c.row} has no field {c.field!r} (fields: {", ".join(k for k in row if k != "row")})')
            continue
        if not same_value(c.value, row[c.field]):
            problems.append(f'{c.id}: cited value {c.value!r} but {c.call} row {c.row} {c.field} = {row[c.field]!r}')
    text = answer.answer
    tags = TAG.findall(text)
    for c in answer.citations:
        if not re.fullmatch(r'c\d+', c.id):
            problems.append(f'citation id {c.id!r} is not of the form c1, c2, ...')
    order = lambda x: (0, int(x[1:])) if re.fullmatch(r'c\d+', x) else (1, x)
    for t in sorted(set(tags) - set(cites), key=order):
        problems.append(f'tag [{t}] has no citation')
    for t in sorted(set(cites) - set(tags), key=order):
        warnings.append(f'citation {t} is not used in the answer')
    for m in TAG.finditer(text):
        c = cites.get(m.group(1))
        if c is None:
            continue
        num = _number(c.value)
        if num is None:  # tags are for numbers; a tag on a name or passage is only flagged (its value was checked above)
            warnings.append(f'[{c.id}] tags a non-numeric value {c.value[:60]!r}')
            continue
        seg = NUMBER_WORD.sub(lambda w: NUMBER_WORDS[w.group(1).lower()], TAG.sub(' ', _segment(text, m.start())))
        nums = NUMBER.findall(seg)  # the number written last before the tag must be the cited one: each tag follows its own number
        if not (nums and _number(nums[-1]) == num):
            problems.append(f'[{c.id}]: the cited value {c.value!r} is not the number written just before the tag')
    # quotes: each must be exact words from the evidence sentences of a statement returned by get_evidence
    sentences = {}
    for out in outputs.values():
        for edge_id, spans in (out.get('_full_evidence') or {}).items():
            sentences.setdefault(edge_id, []).extend(_norm(s) for s in spans)
    quoted = []
    for q in answer.quotes:
        if q.edge_id not in sentences:
            problems.append(f'quote from {q.edge_id!r}: that statement was not returned by get_evidence')
            continue
        text_q = _norm(q.text)
        own = sentences[q.edge_id]
        if not (any(text_q in x for x in own) or text_q in ' '.join(own)):  # one sentence, or consecutive ones of the statement
            elsewhere = [e for e, xs in sentences.items() if e != q.edge_id and (any(text_q in x for x in xs) or text_q in ' '.join(xs))]
            hint = f'; it is a passage of {elsewhere[0]}, cite that edge_id' if elsewhere else ''
            problems.append(f'quote {q.text[:80]!r} is not an exact passage of the evidence sentences of {q.edge_id}{hint}')
            continue
        quoted.append(text_q)
    names = {_norm(v) for out in outputs.values() for row in _rows(out).values() for v in row.values() if isinstance(v, str)}
    names |= {_norm(x) for out in outputs.values() for row in _rows(out).values() for v in row.values() if isinstance(v, list)
              for x in v if isinstance(x, str)}
    asked = _norm(question)
    for m in QUOTED.finditer(text):
        passage = _norm(m.group(1)).strip(' .,;:')
        if passage in names or (asked and passage in asked):
            continue
        if len(passage.split()) >= 3 and not any(passage in q or q in passage for q in quoted):
            problems.append(f'the answer quotes "{m.group(1)[:80]}" but quotes lists no exact source passage for it')
    problems += uncited_numbers(text, question)
    problems += concept_gaps(text + ' ' + ' '.join(answer.limitations), outputs)
    problems += leaked_ids(text, outputs)
    for item in answer.limitations:
        if NUMBER.search(NOT_DATA_NUMBERS.sub(' ', item)):
            warnings.append(f'number in limitations (not verified): {item.strip()[:120]!r}')
    return problems, warnings[:15]


# ---------------------------------------------------------------------------------------------------- loop
def load_key():
    """OPENAI_API_KEY from the environment or the repository .env; never printed."""
    if not os.environ.get('OPENAI_API_KEY'):
        env = ROOT / '.env'
        if env.exists():
            for line in open(env):
                if line.startswith('OPENAI_API_KEY='):
                    os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('"').strip("'")
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set (environment or .env)')


def _strip_none(value):
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def request_body(items, model, effort, allow_tools):
    return {'model': model, 'instructions': INSTRUCTIONS, 'input': items, 'tools': TOOLS,
            'tool_choice': 'auto' if allow_tools else 'none', 'parallel_tool_calls': True,
            'text': {'format': {'type': 'json_schema', 'name': 'agent_answer', 'strict': True, 'schema': FINAL_SCHEMA}},
            'reasoning': {'effort': effort}, 'include': ['reasoning.encrypted_content'], 'store': False,
            'max_output_tokens': MAX_OUTPUT_TOKENS}


def run(question, graph=None, *, model=MODEL, effort=EFFORT, max_tool_rounds=MAX_TOOL_ROUNDS, budget_usd=BUDGET_USD,
        client=None, trace_path=TRACE, label=None):
    """Answer one question; returns the result dict (also appended, with the full trace, to trace_path)."""
    own_graph = graph is None
    graph = graph or KnowledgeGraph()
    session = Session(graph)
    overview = session.overview()
    items = [{'role': 'user', 'content': f'Question: {question}\n\nGraph overview (call t0, citable like any tool output):\n'
                                         f'{compact_json(public(overview))}'}]
    started = dt.datetime.now(dt.timezone.utc)
    manifest_sha = hashlib.sha256(neo4j_db.MANIFEST.read_bytes()).hexdigest() if neo4j_db.MANIFEST.exists() else None
    trace = {'run_id': f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}", 'label': label,
             'started_utc': started.isoformat(timespec='seconds'), 'agent_version': VERSION, 'question': question,
             'graph': {'uri': neo4j_db.URI, 'manifest_sha256': manifest_sha, 'kg_version': graph.manifest.get('kg_version'),
                       'loaded_utc': graph.manifest.get('loaded_utc')},
             'model': model, 'reasoning_effort': effort, 'max_tool_rounds': max_tool_rounds, 'budget_usd': budget_usd,
             'instructions_sha256': sha256_text(INSTRUCTIONS), 'tools_sha256': sha256_text(compact_json(TOOLS)),
             'final_schema_sha256': sha256_text(compact_json(FINAL_SCHEMA)), 'instructions': INSTRUCTIONS, 'tools': TOOLS,
             'final_schema': FINAL_SCHEMA, 'requests': [], 'responses': [], 'attempts': []}
    usage_total, cost, tool_rounds, repair_used, sent = collections.Counter(), 0.0, 0, False, 0
    stop_reason, final = None, None
    try:
        if client is None:
            load_key()
            from openai import OpenAI
            client = OpenAI(max_retries=2, timeout=300)
        while True:
            if cost >= budget_usd:
                stop_reason = f'budget reached (${cost:.4f} of ${budget_usd})'
                break
            if len(trace['responses']) >= max_tool_rounds + REPAIR_TOOL_ROUNDS + 3:
                stop_reason = f"response cap reached ({len(trace['responses'])} responses)"
                break
            allow_tools = tool_rounds < max_tool_rounds + (REPAIR_TOOL_ROUNDS if repair_used else 0)
            body = request_body(items, model, effort, allow_tools)
            trace['requests'].append({**{k: v for k, v in body.items() if k not in ('input', 'instructions', 'tools')},
                                      'input_items': len(items), 'input_new': items[sent:]})
            sent = len(items)
            t0 = time.time()
            try:
                raw = client.responses.create(**body).model_dump()
            except Exception as error:  # transport or HTTP error; the SDK already retried
                stop_reason = f'transport_error: {type(error).__name__}: {mask(error)[:300]}'
                break
            usage = raw.get('usage') or {}
            usd = cost_usd(model, usage) or 0.0
            cost += usd
            for k, v in token_split(usage).items():
                usage_total[k] += v
            trace['responses'].append({'seconds': round(time.time() - t0, 2), 'usd': usd, 'raw': raw})
            calls = [i for i in raw.get('output') or [] if i.get('type') == 'function_call']
            if calls and raw.get('status') == 'completed':
                tool_rounds += 1
                items.extend(_strip_none(i) for i in raw['output'])
                for c in calls:
                    out = session.run_tool(c['name'], c['arguments'])
                    items.append({'type': 'function_call_output', 'call_id': c['call_id'], 'output': compact_json(public(out))})
                continue
            attempt = {'raw_text': None, 'parsed': None, 'problems': [], 'warnings': [], 'anomaly': None}
            try:
                attempt['raw_text'] = final_output_text(raw)
                parsed = validate_model(FinalAnswer, parse_json_object(attempt['raw_text']))
                attempt['parsed'] = parsed.model_dump()
                attempt['problems'], attempt['warnings'] = verify(parsed, session.outputs, question)
            except ExtractionAnomaly as error:
                attempt['anomaly'] = error.as_dict()
                attempt['problems'] = [f'answer not accepted ({error.kind}): {error.detail}']
            trace['attempts'].append(attempt)
            final = attempt
            if not attempt['problems'] or repair_used or (attempt['anomaly'] or {}).get('kind') in ('refusal', 'content_filter'):
                break
            repair_used = True
            if raw.get('status') == 'completed':
                items.extend(_strip_none(i) for i in raw['output'])
            items.append({'role': 'user', 'content': REPAIR.format(problems='\n'.join(f'- {p}' for p in attempt['problems']))})
    finally:
        verified = bool(final and final['parsed'] is not None and not final['problems'])
        result = {'run_id': trace['run_id'], 'label': label, 'question': question, 'verified': verified,
                  'answer': (final or {}).get('parsed'), 'problems': (final or {}).get('problems', []),
                  'warnings': (final or {}).get('warnings', []), 'repair_used': repair_used, 'stop_reason': stop_reason,
                  'tool_rounds': tool_rounds, 'responses': len(trace['responses']),
                  'tool_calls': [{'call': c['call'], 'tool': c['tool'], 'arguments': c['arguments'],
                                  'seconds': c['seconds'], 'error': c['output'].get('error')} for c in session.calls],
                  'usage': dict(usage_total), 'cost_usd': round(cost, 6)}
        trace.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), tool_calls=session.calls,
                     result=result)
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            line = mask(json.dumps(trace, ensure_ascii=False, default=str)) + '\n'
            with TRACE_LOCK, open(trace_path, 'a') as f:
                f.write(line)
        if own_graph:
            graph.close()
    return result
