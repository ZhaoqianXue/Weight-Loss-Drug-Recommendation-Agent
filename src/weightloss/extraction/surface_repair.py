"""Anomaly-only representation repair of saved extraction replies (2026-09-30 technical repair). A record's normal
extraction stays one call per attempt. This path runs only after those attempts are exhausted, on the last saved reply.
Nothing is fuzzily matched, dropped or emptied; the raw reply stays unchanged in the run ledger.

Policies (the one frozen in a repair ledger's manifest decides its behaviour; older ledgers keep theirs):

``surface-select-1.0.0`` (val200)
- Scope: a record whose last reply failed with surface_not_found and is otherwise valid. It must parse, satisfy the
  canonical contract and the drug references, and every quote must resolve to exactly one span under the run's
  localization policy.
- Request: one per effect or indication that is not verbatim. The same model selects the literal wording. The strict
  output is an enum of exact candidates or null.
- Candidates: word n-grams (<= 10 whitespace tokens, edge punctuation trimmed), taken only from that item's own verified
  quote spans. When the rejected value has content words, only candidates sharing one are kept.

``literal-select-1.1.0`` (dev400) = surface-select-1.0.0 plus two changes:
- Surface inflection fallback: when no candidate shares a content word (e.g. "purging" vs the text's "purged"), the
  exact candidates of the item's own quotes that have a word sharing a 4-letter prefix with a content word of the
  rejected value are offered. The limits are unchanged. A value with no such relative in its own quotes is not eligible;
  it is never matched to an unrelated phrase.
- Quote selection: a quote that is not verbatim, with no marker left for localize-1.1.0, is replaced by the model's
  choice among exact source passages.
  - Passages are single sentences, or two consecutive sentences with no blank line, of one field. Each must occur
    exactly once and share content words with the rejected quote (2, or all of them if it has fewer).
  - Sentences are never cut, so negations and numbers stay whole.
  - A rejected quote repeated across items gets one request, and the choice fills every copy.
  - Records with both a bad quote and a bad surface, ambiguous quotes, or drug surfaces are not eligible.

``error-retry-1.0.0`` (with literal-select-1.1.0): for a record whose surface (effect, indication or, from the 2026-09-30
retest, drug surface) or quote still has no exact equivalent after the selection pass (not eligible or null), one
ordinary extraction request is sent. "Once" counts sent retry requests; a skipped retry uses nothing.
- the unchanged normal request of the run's spec, plus one technical user message naming the rejected path and value
  and the verbatim-copy rule;
- its reply is evaluated exactly like a normal attempt;
- every request and reply is archived here;
- it is sent once and never retried;
- the change against the failed reply is reported item by item.

``literal-select-1.2.0`` quote re-selection (with literal-select-1.1.0 ledgers): the 1.1.0 quote instruction asked for
a passage that "contains all of" the rejected quote, which no candidate can when the quote changed a word. A record whose
1.1.0 quote selection returned null and whose error-note retry also failed gets one more request. It uses the same exact
candidates, rebuilt from the same failed reply, and only the instruction changes: a near copy, choose the passage it was
copied from, or null. The request is archived with its instruction sha and sent once. The retry must have ended as an
evidence_not_found anomaly: a held, transport, HTTP or other retry outcome never dispatches here.

``canonical-revalidate-1.0.0`` (offline, no request), used together with literal-select-1.1.0:
- Scope: a record whose last reply failed with schema_invalid.
- The saved reply is re-evaluated in full under the successor spec compact-p7t2 (the same request; canonical
  compact-canonical-negdose-sev-1.0.0).
- It is accepted only when every severity the successor newly allows (on a not_occurred or not_asserted effect) is
  supported by a severity word of that grade in the item's own quotes. The words are those of the prompt's severity rule:
  mild/slight, moderate, severe/extreme/intense.
- Occurrence, statement kind and every other field stay as the model wrote them.

Each change is re-validated end to end (contract, references, surfaces, evidence, compact conversion), and the final
export re-checks every record with the canonical contract that produced it.

Bounds, all failing closed:
- Each record gets one repair pass and each request is sent at most once, never retried.
- An exact request body with its candidate rule is written before sending (request_bodies.jsonl).
- An HTTP 200 body that is not a JSON object is held: its cap is booked as an uncertain charge and it is never re-sent.
- A null choice, a refusal, a truncated or incomplete reply, a value outside the enum, more than 250 candidates or 7,500
  candidate characters, or a failed re-validation leaves the record an anomaly, which blocks the final export.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import compact, localize
from .structured import (ExtractionAnomaly, build_body, check_strict, compact_json, cost_usd, final_output_text, mask,
                         parse_json_object, request_cap_usd, sha256_text, token_split, validate_model)

SURFACE_SELECT_1, LITERAL_SELECT_1_1 = 'surface-select-1.0.0', 'literal-select-1.1.0'
POLICY = SURFACE_SELECT_1  # the val200 policy (kept for its frozen ledger); new ledgers use LATEST
LATEST = LITERAL_SELECT_1_1
REVALIDATE = 'canonical-revalidate-1.0.0'
SUCCESSOR_SPEC = 'compact-p7t2'
LITERAL_SELECT_1_3 = 'literal-select-1.3.0'  # path_repair.py (contract 2.0.0 runs; explicit, never the default)
LITERAL_SELECT_1_3_1 = 'literal-select-1.3.1'  # path_repair.py successor ledger over a 1.3.0 ledger's unresolved records (archived)
LITERAL_SELECT_1_3_2 = 'literal-select-1.3.2'  # path_repair.py successor: 1.3.1 + typed candidate boundaries
LITERAL_SELECT_1_3_3 = 'literal-select-1.3.3'  # path_repair.py successor: 1.3.2 + plain, complete temporal phrases
LITERAL_SELECT_1_4 = 'literal-select-1.4.0'  # path_repair.py, contract 2.1.0 runs (iteration-20260930-01a0f4c2)
LITERAL_SELECT_1_5 = 'literal-select-1.5.0'  # path_repair.py: 1.4.0 + numeric fields keep numbers and units (full batch 20261001)
PATH_POLICIES = (LITERAL_SELECT_1_3, LITERAL_SELECT_1_3_1, LITERAL_SELECT_1_3_2, LITERAL_SELECT_1_3_3, LITERAL_SELECT_1_4,
                 LITERAL_SELECT_1_5)
POLICIES = (SURFACE_SELECT_1, LITERAL_SELECT_1_1, LITERAL_SELECT_1_3, LITERAL_SELECT_1_3_1, LITERAL_SELECT_1_3_2, LITERAL_SELECT_1_3_3,
            LITERAL_SELECT_1_4, LITERAL_SELECT_1_5)
MAX_TOKENS_PER_CANDIDATE, MAX_CANDIDATES, MAX_CANDIDATE_CHARS = 10, 250, 7500
MAX_OUTPUT_TOKENS = 1024
EDGE = '"\'“”‘’()[]{}<>,.;:!?*_-–—'
STOP = frozenset('the and for with was were are this that from have has had but not you your they them then than its his her '
                 'she him our out all any can did does its too very just'.split())
FIELD_MEANING = {'effect': 'the symptom or bodily change, as written in the text',
                 'indication': 'the condition or goal the drug is used for, as written in the text'}
INSTRUCTIONS = ('A drug-statement extraction was rejected because one field value is not copied verbatim from the source text. '
                'You get the field, the rejected value and the exact source quotes that the item cites. Choose the candidate '
                'that states the same thing as the rejected value in the words of the text, preferring the shortest candidate '
                'that covers all of it. Answer null if no candidate states the same thing. Do not reinterpret the item.')
ERROR_RETRY = 'error-retry-1.0.0'
ERROR_NOTE = ('Technical validation note: an earlier answer to this input was rejected because {path} {value} is not copied '
              'verbatim from the input text. Every surface, indication, effect and quote must be copied character for character '
              'from the input. Answer the same task again under the same instructions.')
LITERAL_SELECT_1_2 = 'literal-select-1.2.0'
QUOTE_INSTRUCTIONS_1_2 = ('A drug-statement extraction was rejected because an evidence quote is not an exact copy of the source '
                          'text. The rejected quote may contain wording changes. You get the rejected quote and the items that cite '
                          'it. Choose the exact original source passage that expresses that evidence in context; the passage may '
                          'contain words the rejected quote lacks and lack words it added. Answer null if no passage does. Keep '
                          'negation, numbers and timeline as the source has them; do not change any facts or reinterpret the items.')
QUOTE_INSTRUCTIONS = ('A drug-statement extraction was rejected because an evidence quote is not copied verbatim from the source '
                      'text. You get the rejected quote and the items that cite it. Choose the exact source passage that the '
                      'rejected quote reproduces, preferring the shortest passage that contains all of it. Answer null if no '
                      'passage reproduces it. Do not reinterpret the items.')
# frozen with every request body (request_bodies.jsonl) so the exact enum can be reconstructed after later code changes
CANDIDATE_RULE = {'policy': POLICY, 'source': "the item's own verified quote spans", 'tokens': 'whitespace tokens', 'max_tokens': MAX_TOKENS_PER_CANDIDATE,
                  'edge_trim': EDGE, 'verbatim_check': 'phrase must occur in its quote span text',
                  'content_filter': 'when the rejected value has words of >= 3 letters/digits not in STOP, keep candidates sharing one',
                  'stop': sorted(STOP), 'max_candidates': MAX_CANDIDATES, 'max_candidate_chars': MAX_CANDIDATE_CHARS}
CANDIDATE_RULE_1_1 = {**CANDIDATE_RULE, 'policy': LITERAL_SELECT_1_1,
                      'surface_fallback': 'when no candidate shares a content word: candidates with a word sharing a 4-letter prefix with a rejected content word',
                      'quote_source': 'single sentences or two consecutive sentences (no blank line) of one field, occurring exactly once',
                      'quote_filter': 'share >= min(2, n) content words with the rejected quote (n = its content words)',
                      'sentence_rule': 'lines split after [.!?]+ (plus closing quotes/brackets) followed by whitespace or line end; decimals never split; trimmed; >= 3 words'}
SEVERITY_WORDS = {'mild': r'\b(?:mild|mildly|slight|slightly)\b', 'moderate': r'\bmoderate(?:ly)?\b',
                  'severe': r'\b(?:severe|severely|extreme|extremely|intense|intensely)\b'}
SENTENCE_END = re.compile(r'[.!?]+["”’\')\]]*(?=\s|$)')  # a terminator ends a sentence only before whitespace: "0.25", "1.5 mg" stay whole
BLANK_LINE = re.compile(r'\n[^\S\n]*\n')


def _words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:['’][a-z]+)?", s.lower()) if len(w) >= 3 and w not in STOP}


def candidates(spans, fields, rejected, fallback=False):
    """Exact candidate phrases from the item's own quote spans ('field:start-end'), in first-occurrence order."""
    seen, out = set(), []
    for span in spans:
        text = compact.render_span(span, fields)
        tokens = text.split()
        for n in range(1, MAX_TOKENS_PER_CANDIDATE + 1):
            for i in range(len(tokens) - n + 1):
                phrase = ' '.join(tokens[i:i + n]).strip(EDGE)
                if phrase and phrase not in seen and phrase in text:  # a phrase joining a line break is not verbatim
                    seen.add(phrase)
                    out.append(phrase)
    words = _words(rejected)
    if words:
        kept = [c for c in out if _words(c) & words]
        if not kept and fallback:  # inflection only: a shared 4-letter prefix, never an unrelated phrase
            stems = {w[:4] for w in words if len(w) >= 4}
            kept = [c for c in out if any(w[:4] in stems for w in _words(c) if len(w) >= 4)]
        out = kept
    return out


def sentences(text):
    """[(start, end)] of the trimmed sentence spans of one field: each line is split after sentence punctuation that is
    followed by whitespace or the line end, so decimals ("0.25", "1.5 mg") and the words around them are never cut."""
    out = []
    for line in re.finditer(r'[^\n]+', text):
        start = line.start()
        for m in SENTENCE_END.finditer(line.group()):
            out.append((start, line.start() + m.end()))
            start = line.start() + m.end()
        if start < line.end():
            out.append((start, line.end()))
    trimmed = []
    for start, end in out:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if len(text[start:end].split()) >= 3:
            trimmed.append((start, end))
    return trimmed


def quote_candidates(fields, rejected):
    """Exact source passages for a rejected quote (literal-select-1.1.0), in text order."""
    words = _words(rejected)
    need = min(2, len(words))
    seen, out = set(), []
    for name, text in fields.items():
        spans = sentences(text)
        windows = spans + [(a[0], b[1]) for a, b in zip(spans, spans[1:]) if not BLANK_LINE.search(text[a[1]:b[0]])]
        for start, end in sorted(windows):
            passage = text[start:end]
            if passage in seen or len(passage) > 1000 or len(_words(passage) & words) < need:
                continue
            seen.add(passage)
            if len(compact.locate(passage, fields)) == 1:
                out.append(passage)
    return out


def _source_checked(wire, spans, inp, spec):
    wire = validate_model(spec.canonical_wire, wire)
    compact.check_references(wire)
    return wire


def inspect(value, inp, spec, policy, repair_policy=SURFACE_SELECT_1):
    """(localized wire dict, repair tasks) for a reply the repair policy can handle; ExtractionAnomaly otherwise."""
    tag = repair_policy
    if spec.to_canonical is not None:
        value = spec.to_canonical(validate_model(spec.wire_model, value))
    wire = validate_model(spec.canonical_wire, value)
    compact.check_references(wire)
    if policy is not None:
        wire, _ = localize.apply(wire, inp.fields, policy)
        wire = validate_model(spec.canonical_wire, wire.model_dump())
    texts = list(inp.fields.values())
    if any(not any(d.surface in t for t in texts) for d in wire.drugs):
        raise ExtractionAnomaly('surface_not_found', f'{tag}: drug surfaces are not repaired', retriable=False)
    if repair_policy == LITERAL_SELECT_1_1:
        missing = {}
        for group, i, item in compact.items(wire):
            for q, quote in enumerate(item.quotes):
                hits = compact.locate(quote, inp.fields)
                if len(hits) > 1:
                    raise ExtractionAnomaly('evidence_ambiguous', f'{tag}: {group}[{i}].quotes[{q}] is ambiguous', retriable=False)
                if not hits:
                    missing.setdefault(quote, []).append((group, i, q))
        if missing:
            bad_surfaces = [1 for group, name in (('purposes', 'indication'), ('side_effects', 'effect'))
                            for item in getattr(wire, group) if not any(getattr(item, name) in t for t in texts)]
            if bad_surfaces:
                raise ExtractionAnomaly('evidence_not_found', f'{tag}: a record with both a bad quote and a bad surface is not repaired', retriable=False)
            tasks = []
            for quote, targets in missing.items():
                cands = quote_candidates(inp.fields, quote)
                if not cands or len(cands) > MAX_CANDIDATES or sum(map(len, cands)) > MAX_CANDIDATE_CHARS:
                    raise ExtractionAnomaly('evidence_not_found', f'{tag}: rejected quote has {len(cands)} passages', retriable=False)
                group, i, q = targets[0]
                tasks.append({'kind': 'quote', 'path': f'{group}[{i}].quotes[{q}]', 'targets': targets, 'rejected': quote,
                              'items': [{'path': f'{g}[{j}]', 'drug': wire.model_dump()[g][j]['drug'],
                                         'value': wire.model_dump()[g][j].get('effect') or wire.model_dump()[g][j].get('indication')}
                                        for g, j, _ in targets],
                              'candidates': cands})
            return wire.model_dump(), tasks
    spans = compact.locate_evidence(wire, inp.fields)
    tasks = []
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect')):
        for i, item in enumerate(getattr(wire, group)):
            rejected = getattr(item, name)
            if not any(rejected in t for t in texts):
                cands = candidates(spans[group, i], inp.fields, rejected, fallback=repair_policy == LITERAL_SELECT_1_1)
                if not cands or len(cands) > MAX_CANDIDATES or sum(map(len, cands)) > MAX_CANDIDATE_CHARS:
                    raise ExtractionAnomaly('surface_not_found', f'{tag}: {group}[{i}].{name} has {len(cands)} candidates', retriable=False)
                tasks.append({'kind': 'surface', 'path': f'{group}[{i}].{name}', 'group': group, 'index': i, 'field': name, 'rejected': rejected,
                              'quotes': [compact.render_span(s, inp.fields) for s in spans[group, i]], 'candidates': cands})
    if not tasks:
        raise ExtractionAnomaly('surface_not_found', f'{tag}: no effect or indication to repair', retriable=False)
    return wire.model_dump(), tasks


def schema(cands):
    out = {'type': 'object', 'properties': {'choice': {'anyOf': [{'type': 'string', 'enum': list(cands)}, {'type': 'null'}]}},
           'required': ['choice'], 'additionalProperties': False}
    check_strict(out)
    return out


def request_body(task, model, reasoning_effort, quote_instructions=QUOTE_INSTRUCTIONS):
    if task.get('kind') == 'quote':
        items = '\n'.join(f"- {it['path']}: drug {it['drug']}, {json.dumps(it['value'], ensure_ascii=False)}" for it in task['items'])
        content = f"Rejected quote: {json.dumps(task['rejected'], ensure_ascii=False)}\nItems that cite it:\n{items}"
        return build_body(model=model, instructions=quote_instructions, user_content=content, schema_name='quote_select',
                          schema=schema(task['candidates']), reasoning_effort=reasoning_effort, max_output_tokens=MAX_OUTPUT_TOKENS)
    quotes = '\n'.join(f'<<<\n{q}\n>>>' for q in task['quotes'])
    content = (f"Field: {task['path']} ({FIELD_MEANING[task['field']]})\nRejected value: {json.dumps(task['rejected'], ensure_ascii=False)}\n"
               f'Source quotes cited by the item:\n{quotes}')
    return build_body(model=model, instructions=INSTRUCTIONS, user_content=content, schema_name='surface_select',
                      schema=schema(task['candidates']), reasoning_effort=reasoning_effort, max_output_tokens=MAX_OUTPUT_TOKENS)


def read_choice(response, task, tag=POLICY):
    """The selected candidate, or ExtractionAnomaly (refusal, truncation, incomplete, null, outside the enum)."""
    kind = 'evidence_not_found' if task.get('kind') == 'quote' else 'surface_not_found'
    value = parse_json_object(final_output_text(response))
    if set(value) != {'choice'}:
        raise ExtractionAnomaly('schema_invalid', f'{tag}: reply keys {sorted(value)}', retriable=False)
    choice = value['choice']
    if choice is None:
        raise ExtractionAnomaly(kind, f'{tag}: no candidate selected (null)', retriable=False)
    if choice not in task['candidates']:
        raise ExtractionAnomaly('schema_invalid', f'{tag}: choice outside the candidate list', retriable=False)
    return choice


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _manifest(ledger, policy):
    out = {'policy': policy, 'base_run': ledger.run_id, 'base_manifest_sha256': sha256_text((ledger.dir/'manifest.json').read_bytes().decode()),
           'model': ledger.manifest['model'], 'reasoning_effort': ledger.manifest['reasoning_effort'],
           'max_output_tokens': MAX_OUTPUT_TOKENS, 'instructions_sha256': sha256_text(INSTRUCTIONS), 'localize': ledger.manifest.get('localize')}
    if policy in PATH_POLICIES:
        from . import path_repair
        out.update(instructions_sha256=path_repair.INSTRUCTIONS_SHA, contract=ledger.spec.contract, retry_policy=ledger.manifest.get('retry_policy'))
    if policy == LITERAL_SELECT_1_1:
        out.update(quote_instructions_sha256=sha256_text(QUOTE_INSTRUCTIONS), revalidate=REVALIDATE, successor_spec=SUCCESSOR_SPEC,
                   successor_fingerprint=compact.get_spec(SUCCESSOR_SPEC).fingerprint(), error_retry=ERROR_RETRY,
                   error_note_sha256=sha256_text(ERROR_NOTE))
    return out


class RepairLedger:
    """Append-only ledger of the repairs for one base run (private: holds source quotes and replies). A new ledger freezes
    `policy` (default LATEST) in its manifest; an existing one keeps the policy it was created with. `extra` (creation
    only) adds frozen keys that cannot be recomputed from the base run, e.g. a successor's predecessor and scope."""

    def __init__(self, directory, ledger, create=True, policy=LATEST, extra=None):
        self.dir, self.base = Path(directory), ledger
        path = self.dir/'manifest.json'
        if path.exists():
            known = json.loads(path.read_text())
            if known.get('policy') not in POLICIES:
                raise ValueError(f'{path}: unknown repair policy {known.get("policy")!r}')
            manifest = _manifest(ledger, known['policy'])
            if {k: known.get(k) for k in manifest} != manifest:
                raise ValueError(f'{path} was created for another base run or repair version')
            stored = known
        elif create:
            if policy not in POLICIES:
                raise ValueError(f'unknown repair policy {policy!r}')
            manifest = _manifest(ledger, policy)
            if set(extra or {}) & (set(manifest) | {'created_utc'}):
                raise ValueError('extra manifest keys must not override computed ones')
            self.dir.mkdir(parents=True, exist_ok=True)
            stored = {**manifest, **(extra or {}), 'created_utc': now()}
            path.write_text(json.dumps(stored, indent=1) + '\n')
        else:
            raise ValueError(f'{self.dir} holds no surface-repair ledger')
        self.manifest, self.policy = {**stored, **manifest}, manifest['policy']

    @classmethod
    def open(cls, directory, ledger):
        """Read-only: an existing repair ledger of this base run and repair version; nothing is created or written."""
        return cls(directory, ledger, create=False)

    def append(self, name, row):
        with open(self.dir/f'{name}.jsonl', 'ab+') as stream:
            if stream.seek(0, 2):
                stream.seek(-1, 2)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')  # close a torn tail so the new row stays parseable
            stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())

    def rows(self, name):
        """Committed rows; a line that does not parse is a torn tail from an interrupted append and carries no fact (as in
        RunLedger.rows). An intent whose outcome was torn is accounted for by path_repair.recover."""
        path = self.dir/f'{name}.jsonl'
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def spent_and_uncertain(self):
        return (sum(r['cost_usd'] or 0 for r in self.rows('usage')),
                sum(r.get('uncertain_charge_usd', 0) for r in self.rows('attempts')))

    def results(self):
        """{record_id: last result row} (ok rows carry the repaired compact record)."""
        out = {}
        for row in self.rows('results'):
            out[row['record_id']] = row
        return out


def eligible(ledger, policy=SURFACE_SELECT_1):
    """[(record_id, custom_id of the last failed attempt)] for records whose final state the policy can repair."""
    kinds = {'surface_not_found', 'evidence_not_found'} if policy == LITERAL_SELECT_1_1 else {'surface_not_found'}
    out = []
    for rid, state in ledger.states().items():
        last = state['last']
        if not state['ok'] and not state['held'] and last and last['anomaly']['kind'] in kinds:
            out.append((rid, last['custom_id']))
    return out


def _severity_support(record, inp):
    """[(effect, grade, supported)] for severities on not_occurred/not_asserted effects (the ones the successor allows)."""
    out = []
    for se in record['side_effects']:
        if se.get('occurrence') in ('not_occurred', 'not_asserted') and se.get('severity'):
            text = ' '.join(compact.render_span(s, inp.fields) for s in se['ev'])
            out.append((se.get('effect'), se['severity'], bool(re.search(SEVERITY_WORDS[se['severity']], text, re.I))))
    return out


def revalidate(ledger, repair):
    """canonical-revalidate-1.0.0 (offline, literal-select-1.1.0 ledgers only): re-evaluate the last saved reply of each
    schema_invalid record under compact-p7t2; accept only with quote-supported severities. Returns [(record_id, outcome)]."""
    if repair.policy != LITERAL_SELECT_1_1:
        return []
    successor = compact.get_spec(SUCCESSOR_SPEC)
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    done, outcomes = set(repair.results()), []
    for rid, state in ledger.states().items():
        last = state['last']
        if state['ok'] or state['held'] or not last or last['anomaly']['kind'] != 'schema_invalid' or rid in done:
            continue
        cid, inp = last['custom_id'], ledger.inputs()[rid]
        base = {'record_id': rid, 'from_custom_id': cid, 'policy': REVALIDATE, 'spec': SUCCESSOR_SPEC,
                'canonical_version': successor.canonical[0], 'at': now()}
        try:
            if cid not in responses:
                raise ExtractionAnomaly('schema_invalid', f'{REVALIDATE}: no saved reply for {cid}', retriable=False)
            record = compact.evaluate_value(parse_json_object(final_output_text(responses[cid])), inp, ledger.run_id, successor,
                                            localize_policy=ledger.manifest.get('localize'), repairs=[])
            support = _severity_support(record, inp)
            if not all(ok for _, _, ok in support):
                raise ExtractionAnomaly('schema_invalid', f'{REVALIDATE}: severity not supported by the quotes: {support}', retriable=False)
        except ExtractionAnomaly as anomaly:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': anomaly.as_dict()})
            outcomes.append((rid, anomaly.kind))
            continue
        repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': [],
                                  'severity_support': [{'effect': e, 'severity': g, 'supported': ok} for e, g, ok in support],
                                  'reply_sha256': hashlib.sha256(compact_json(responses[cid]).encode()).hexdigest()})
        outcomes.append((rid, 'ok'))
    return outcomes


def plan(ledger, repair):
    """Offline preview: [(record_id, custom_id, tasks or the not-eligible reason)] under the ledger's policy."""
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    out = []
    for rid, cid in eligible(ledger, repair.policy):
        if rid in repair.results():
            continue
        try:
            value = parse_json_object(final_output_text(responses[cid]))
            out.append((rid, cid, inspect(value, ledger.inputs()[rid], ledger.spec, ledger.manifest.get('localize'), repair.policy)[1]))
        except (ExtractionAnomaly, KeyError) as error:
            out.append((rid, cid, str(error)[:160]))
    return out


def run(ledger, repair, send, *, budget_usd):
    """Repair every eligible record once. Requests are sent sequentially within budget_usd (this ledger's spend and
    uncertain charges included); a record or field with any earlier repair request is never sent again."""
    if repair.policy in PATH_POLICIES:
        from . import path_repair
        return path_repair.run(ledger, repair, send, budget_usd=budget_usd)
    spec, policy, model = ledger.spec, ledger.manifest.get('localize'), ledger.manifest['model']
    tag = repair.policy
    rule = CANDIDATE_RULE_1_1 if tag == LITERAL_SELECT_1_1 else CANDIDATE_RULE
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    sent = {r['custom_id'] for r in repair.rows('requests')}
    done = set(repair.results())  # one repair pass per record, whatever its outcome
    spent, uncertain = repair.spent_and_uncertain()
    outcomes = []
    for rid, cid in eligible(ledger, tag):
        if rid in done:
            outcomes.append((rid, 'already_repaired' if repair.results()[rid]['outcome'] == 'ok' else 'already_tried'))
            continue
        inp = ledger.inputs()[rid]
        base = {'record_id': rid, 'from_custom_id': cid, 'policy': tag, 'at': now()}
        try:
            if cid not in responses:
                raise ExtractionAnomaly('surface_not_found', f'{tag}: no saved reply for {cid}', retriable=False)
            value, tasks = inspect(parse_json_object(final_output_text(responses[cid])), inp, spec, policy, tag)
        except ExtractionAnomaly as anomaly:
            repair.append('results', {**base, 'outcome': 'not_eligible', 'anomaly': anomaly.as_dict()})
            outcomes.append((rid, 'not_eligible'))
            continue
        changes, failure = [], None
        for task in tasks:
            rcid = f"{rid}#{task['path']}"
            if rcid in sent:
                failure = ExtractionAnomaly('surface_not_found', f'{tag}: {task["path"]} was already requested once', retriable=False)
                break
            body = request_body(task, model, ledger.manifest['reasoning_effort'])
            cap = request_cap_usd(model, body)
            if spent + uncertain + cap > budget_usd:
                failure = ExtractionAnomaly('surface_not_found', f'{tag}: budget', retriable=False)
                break
            body_sha = sha256_text(compact_json(body))
            repair.append('request_bodies', {'custom_id': rcid, 'body_sha256': body_sha, 'candidate_rule': rule, 'body': body})
            repair.append('requests', {'custom_id': rcid, 'record_id': rid, 'path': task['path'], 'kind': task.get('kind', 'surface'),
                                       'body_sha256': body_sha, 'candidates': len(task['candidates']), 'at': now()})
            sent.add(rcid)
            try:
                status, headers, text = send(body)
            except Exception as error:  # transport: the request may have been billed; never re-sent
                uncertain += cap
                repair.append('attempts', {'custom_id': rcid, 'error': type(error).__name__, 'uncertain_charge_usd': round(cap, 9), 'at': now()})
                failure = ExtractionAnomaly('transport_error', f'{tag}: {type(error).__name__}', retriable=False)
                break
            attempt = {'custom_id': rcid, 'http_status': status, 'request_id': (headers or {}).get('x-request-id'), 'at': now()}
            if status != 200:
                uncertain += cap if status >= 500 else 0.0
                repair.append('attempts', {**attempt, 'error': mask(text)[:300], 'uncertain_charge_usd': round(cap if status >= 500 else 0.0, 9)})
                failure = ExtractionAnomaly('http_error', f'{tag}: HTTP {status}', retriable=False)
                break
            try:
                response = json.loads(text)
            except json.JSONDecodeError:
                response, unreadable = None, 'HTTP 200 body is not JSON'
            else:
                unreadable = None if isinstance(response, dict) else 'HTTP 200 body is JSON but not an object'
            if unreadable:  # answered and possibly billed, but unreadable: book the cap, hold, never re-send
                uncertain += cap
                repair.append('attempts', {**attempt, 'error': f'{unreadable}: ' + mask(text)[:200], 'uncertain_charge_usd': round(cap, 9)})
                repair.append('held', {'custom_id': rcid, 'reason': f'{unreadable}; held, not re-sent', 'at': now()})
                failure = 'held'
                break
            repair.append('attempts', attempt)
            repair.append('responses', {'custom_id': rcid, 'response': response})
            usage = response.get('usage') or {}
            cost = cost_usd(model, usage) or 0.0
            spent += cost
            repair.append('usage', {'custom_id': rcid, 'response_id': response.get('id'), 'status': response.get('status'), **token_split(usage),
                                    'cost_usd': cost, 'at': now()})
            try:
                choice = read_choice(response, task, tag)
            except ExtractionAnomaly as anomaly:
                failure = anomaly
                break
            if task.get('kind') == 'quote':
                for group, i, q in task['targets']:
                    value[group][i]['quotes'][q] = choice
                changes.append({'path': task['path'], 'kind': 'quote', 'targets': [f'{g}[{i}].quotes[{q}]' for g, i, q in task['targets']],
                                'from': task['rejected'], 'to': choice, 'candidates': len(task['candidates']), 'custom_id': rcid})
            else:
                value[task['group']][task['index']][task['field']] = choice
                changes.append({'path': task['path'], 'from': task['rejected'], 'to': choice, 'candidates': len(task['candidates']), 'custom_id': rcid})
        if failure is None:
            try:
                record = compact.evaluate_value(value, inp, ledger.run_id, spec, localize_policy=policy, repairs=[])
            except ExtractionAnomaly as anomaly:
                failure = anomaly
        if failure == 'held':
            repair.append('results', {**base, 'outcome': 'held', 'reason': 'HTTP 200 body is not a JSON object', 'changes': changes})
            outcomes.append((rid, 'held'))
        elif failure is not None:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': failure.as_dict(), 'changes': changes})
            outcomes.append((rid, failure.kind))
        else:
            repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': changes})
            outcomes.append((rid, 'ok'))
    return outcomes


def _send_once(repair, send, rcid, body, cap, tag):
    """Send one archived request; (response or None, failure or None, cost). Shared bookkeeping of the repair steps."""
    try:
        status, headers, text = send(body)
    except Exception as error:  # transport: the request may have been billed; never re-sent
        repair.append('attempts', {'custom_id': rcid, 'error': type(error).__name__, 'uncertain_charge_usd': round(cap, 9), 'at': now()})
        return None, ExtractionAnomaly('transport_error', f'{tag}: {type(error).__name__}', retriable=False), 0.0
    attempt = {'custom_id': rcid, 'http_status': status, 'request_id': (headers or {}).get('x-request-id'), 'at': now()}
    if status != 200:
        repair.append('attempts', {**attempt, 'error': mask(text)[:300], 'uncertain_charge_usd': round(cap if status >= 500 else 0.0, 9)})
        return None, ExtractionAnomaly('http_error', f'{tag}: HTTP {status}', retriable=False), 0.0
    try:
        response = json.loads(text)
    except json.JSONDecodeError:
        response, unreadable = None, 'HTTP 200 body is not JSON'
    else:
        unreadable = None if isinstance(response, dict) else 'HTTP 200 body is JSON but not an object'
    if unreadable:
        repair.append('attempts', {**attempt, 'error': f'{unreadable}: ' + mask(text)[:200], 'uncertain_charge_usd': round(cap, 9)})
        repair.append('held', {'custom_id': rcid, 'reason': f'{unreadable}; held, not re-sent', 'at': now()})
        return None, 'held', 0.0
    repair.append('attempts', attempt)
    repair.append('responses', {'custom_id': rcid, 'response': response})
    usage = response.get('usage') or {}
    cost = cost_usd(repair.manifest['model'], usage) or 0.0
    repair.append('usage', {'custom_id': rcid, 'response_id': response.get('id'), 'status': response.get('status'), **token_split(usage),
                            'cost_usd': cost, 'at': now()})
    return response, None, cost


def _item_keys(record):
    return {g: sorted(json.dumps({k: v for k, v in it.items() if k != 'ev'}, sort_keys=True, ensure_ascii=False) for it in record[g])
            for g in compact.GROUPS}


def error_retry(ledger, repair, send, *, budget_usd):
    """error-retry-1.0.0 (literal-select-1.1.0 ledgers): one ordinary extraction request with a verbatim-copy note for each
    surface/evidence record that the selection pass left without a validated result. Returns [(record_id, outcome)]."""
    if repair.policy != LITERAL_SELECT_1_1:
        return []
    spec, policy, model = ledger.spec, ledger.manifest.get('localize'), ledger.manifest['model']
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    sent = {r['custom_id'] for r in repair.rows('requests')}
    retried = {r['record_id'] for r in repair.rows('requests') if r.get('kind') == 'error_retry'}  # a sent retry, not a skipped one
    rows = repair.rows('results')
    spent, uncertain = repair.spent_and_uncertain()
    outcomes = []
    for rid, cid in eligible(ledger, LITERAL_SELECT_1_1):
        mine = [r for r in rows if r['record_id'] == rid]
        if rid in retried or not mine or any(r['outcome'] == 'ok' for r in mine) or not any(r.get('policy') == LITERAL_SELECT_1_1 for r in mine):
            continue
        inp, rcid = ledger.inputs()[rid], f'{rid}#{ERROR_RETRY}'
        failed = parse_json_object(final_output_text(responses[cid]))
        path, value = _rejected_literal(failed, inp)
        base = {'record_id': rid, 'from_custom_id': cid, 'policy': ERROR_RETRY, 'at': now(), 'rejected': {'path': path, 'value': value}}
        if rcid in sent or path is None:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': ExtractionAnomaly(
                'surface_not_found', f'{ERROR_RETRY}: {"already requested" if rcid in sent else "no rejected literal found"}', retriable=False).as_dict()})
            outcomes.append((rid, 'not_eligible'))
            continue
        body = compact.build_request(inp, spec=spec, model=model, reasoning_effort=ledger.manifest['reasoning_effort'],
                                     max_output_tokens=ledger.manifest['max_output_tokens'])
        normal_sha = sha256_text(compact_json(body))
        note = ERROR_NOTE.format(path=path, value=json.dumps(value, ensure_ascii=False))
        body = {**body, 'input': body['input'] + [{'role': 'user', 'content': note}]}
        cap = request_cap_usd(model, body)
        if spent + uncertain + cap > budget_usd:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': ExtractionAnomaly('surface_not_found', f'{ERROR_RETRY}: budget', retriable=False).as_dict()})
            outcomes.append((rid, 'budget'))
            continue
        body_sha = sha256_text(compact_json(body))
        repair.append('request_bodies', {'custom_id': rcid, 'body_sha256': body_sha, 'normal_request_sha256': normal_sha, 'note': note, 'body': body})
        repair.append('requests', {'custom_id': rcid, 'record_id': rid, 'path': path, 'kind': 'error_retry', 'body_sha256': body_sha, 'at': now()})
        sent.add(rcid)
        response, failure, cost = _send_once(repair, send, rcid, body, cap, ERROR_RETRY)
        spent, uncertain = repair.spent_and_uncertain()  # the ledger is authoritative for the next budget check
        if failure is None:
            try:
                record = compact.evaluate_response(response, inp, ledger.run_id, spec, localize_policy=policy, repairs=[])
            except ExtractionAnomaly as anomaly:
                failure = anomaly
        if failure == 'held':
            repair.append('results', {**base, 'outcome': 'held', 'reason': 'HTTP 200 body is not a JSON object'})
            outcomes.append((rid, 'held'))
        elif failure is not None:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': failure.as_dict()})
            outcomes.append((rid, failure.kind))
        else:
            before, after = _item_keys(_loose_record(failed)), _item_keys(record)
            change = {g: {'kept': len(set(before[g]) & set(after[g])), 'only_failed_reply': len(set(before[g]) - set(after[g])),
                          'only_retry': len(set(after[g]) - set(before[g]))} for g in compact.GROUPS}
            repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': [{'kind': 'error_retry', 'custom_id': rcid, 'items': change}]})
            outcomes.append((rid, 'ok'))
    return outcomes


def quote_reselect(ledger, repair, send, *, budget_usd):
    """literal-select-1.2.0 quote re-selection (see the module docstring). Returns [(record_id, outcome)]."""
    if repair.policy != LITERAL_SELECT_1_1:
        return []
    spec, policy, model = ledger.spec, ledger.manifest.get('localize'), ledger.manifest['model']
    responses = {r['custom_id']: r['response'] for r in ledger.rows('responses')}
    rows = repair.rows('results')
    sent = {r['custom_id'] for r in repair.rows('requests')}
    spent, uncertain = repair.spent_and_uncertain()
    outcomes = []
    for rid, cid in eligible(ledger, LITERAL_SELECT_1_1):
        mine = [r for r in rows if r['record_id'] == rid]
        null_quote = any(r.get('policy') == LITERAL_SELECT_1_1 and r['outcome'] == 'anomaly' and 'no candidate selected' in r['anomaly']['detail']
                         and r['anomaly']['kind'] == 'evidence_not_found' for r in mine)
        last = mine[-1] if mine else None
        # only the old-request defect: the error-note retry itself failed on evidence. A held, transport, HTTP or any other
        # retry outcome never dispatches here and stays unresolved (the final export refuses it).
        retry_failed_on_evidence = (last is not None and last.get('policy') == ERROR_RETRY and last['outcome'] == 'anomaly'
                                    and last['anomaly']['kind'] == 'evidence_not_found')
        if not retry_failed_on_evidence or not null_quote or any(r.get('policy') == LITERAL_SELECT_1_2 for r in mine):
            continue
        inp = ledger.inputs()[rid]
        base = {'record_id': rid, 'from_custom_id': cid, 'policy': LITERAL_SELECT_1_2, 'at': now()}
        value, tasks = inspect(parse_json_object(final_output_text(responses[cid])), inp, spec, policy, LITERAL_SELECT_1_1)
        changes, failure = [], None
        for task in tasks:
            rcid = f"{rid}#{task['path']}@1.2"
            if task.get('kind') != 'quote' or rcid in sent:
                failure = ExtractionAnomaly('evidence_not_found', f'{LITERAL_SELECT_1_2}: not a quote task or already requested', retriable=False)
                break
            body = request_body(task, model, ledger.manifest['reasoning_effort'], QUOTE_INSTRUCTIONS_1_2)
            cap = request_cap_usd(model, body)
            if spent + uncertain + cap > budget_usd:
                failure = ExtractionAnomaly('evidence_not_found', f'{LITERAL_SELECT_1_2}: budget', retriable=False)
                break
            body_sha = sha256_text(compact_json(body))
            repair.append('request_bodies', {'custom_id': rcid, 'body_sha256': body_sha, 'candidate_rule': CANDIDATE_RULE_1_1,
                                             'instructions_sha256': sha256_text(QUOTE_INSTRUCTIONS_1_2), 'policy': LITERAL_SELECT_1_2, 'body': body})
            repair.append('requests', {'custom_id': rcid, 'record_id': rid, 'path': task['path'], 'kind': 'quote', 'policy': LITERAL_SELECT_1_2,
                                       'body_sha256': body_sha, 'candidates': len(task['candidates']), 'at': now()})
            sent.add(rcid)
            response, failure, _ = _send_once(repair, send, rcid, body, cap, LITERAL_SELECT_1_2)
            spent, uncertain = repair.spent_and_uncertain()
            if failure is not None:
                break
            try:
                choice = read_choice(response, task, LITERAL_SELECT_1_2)
            except ExtractionAnomaly as anomaly:
                failure = anomaly
                break
            for group, i, q in task['targets']:
                value[group][i]['quotes'][q] = choice
            changes.append({'path': task['path'], 'kind': 'quote', 'targets': [f'{g}[{i}].quotes[{q}]' for g, i, q in task['targets']],
                            'from': task['rejected'], 'to': choice, 'candidates': len(task['candidates']), 'custom_id': rcid})
        if failure is None:
            try:
                record = compact.evaluate_value(value, inp, ledger.run_id, spec, localize_policy=policy, repairs=[])
            except ExtractionAnomaly as anomaly:
                failure = anomaly
        if failure == 'held':
            repair.append('results', {**base, 'outcome': 'held', 'reason': 'HTTP 200 body is not a JSON object', 'changes': changes})
            outcomes.append((rid, 'held'))
        elif failure is not None:
            repair.append('results', {**base, 'outcome': 'anomaly', 'anomaly': failure.as_dict(), 'changes': changes})
            outcomes.append((rid, failure.kind))
        else:
            repair.append('results', {**base, 'outcome': 'ok', 'record': record, 'changes': changes})
            outcomes.append((rid, 'ok'))
    return outcomes


def _rejected_literal(value, inp):
    """(path, value) of the first effect/indication, drug surface or quote of a failed reply that is not verbatim in the input."""
    texts = list(inp.fields.values())
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect')):
        for i, item in enumerate(value.get(group) or []):
            if not any(item.get(name, '') in t for t in texts):
                return f'{group}[{i}].{name}', item.get(name)
    for i, drug in enumerate(value.get('drugs') or []):
        if not any(drug.get('surface', '') in t for t in texts):
            return f'drugs[{i}].surface', drug.get('surface')
    for group in compact.GROUPS:
        for i, item in enumerate(value.get(group) or []):
            for q, quote in enumerate(item.get('quotes') or []):
                if not any(quote in t for t in texts):
                    return f'{group}[{i}].quotes[{q}]', quote
    return None, None


def _loose_record(value):
    """Items of a raw (failed) reply in the compact item shape, for the item-level change report only."""
    return {g: [{k: v for k, v in it.items() if k != 'quotes' and v is not None and v != []} for it in value.get(g) or []] for g in compact.GROUPS}


def _chain(repair):
    return [] if repair is None else list(repair) if isinstance(repair, (list, tuple)) else [repair]


def successor_scope(ledger, chain):
    """Record IDs with no validated result from the run or any ledger of the chain (a successor's frozen scope)."""
    have = delivered(ledger, chain)
    return sorted(rid for rid in ledger.inputs() if rid not in have)


def successor_extra(ledger, predecessor):
    """Frozen manifest keys of a successor ledger over `predecessor` (a finished ledger of the same base run)."""
    return {'predecessor': {'dir': predecessor.dir.name, 'parent': predecessor.dir.parent.name, 'policy': predecessor.policy,
                            'results_sha256': sha256_text((predecessor.dir/'results.jsonl').read_text()) if (predecessor.dir/'results.jsonl').exists() else None},
            'scope': successor_scope(ledger, [predecessor])}


def check_chain(ledger, chain):
    """A repair chain for export: the first ledger is the run's own; each later ledger names the previous one as its
    unchanged predecessor, its frozen scope is exactly what the earlier ledgers left undelivered, and every result it holds
    is inside that scope. ValueError otherwise."""
    for k, successor in enumerate(chain):
        pred = successor.manifest.get('predecessor')
        if k == 0:
            if pred is not None:
                raise ValueError(f'{successor.dir}: a successor ledger needs its predecessor before it in the chain')
            continue
        prev = chain[k - 1]
        want = successor_extra(ledger, prev)
        if pred != want['predecessor']:
            raise ValueError(f'{successor.dir}: predecessor {pred} does not match {prev.dir} as it is now ({want["predecessor"]})')
        if successor.manifest.get('scope') != successor_scope(ledger, chain[:k]):
            raise ValueError(f'{successor.dir}: frozen scope differs from the records the earlier ledgers left undelivered')
        outside = sorted(set(successor.results()) - set(successor.manifest['scope']))
        if outside:
            raise ValueError(f'{successor.dir}: results outside its scope: {outside[:5]}')


def delivered(ledger, repair, with_canonical=False):
    """{record_id: compact record} over the frozen inputs: the extraction result, else the first repaired result along
    the repair chain (one ledger, or a list: a ledger and its successors). With with_canonical, values are (record,
    canonical version that produced it)."""
    states, fixed = ledger.states(), [r.results() for r in _chain(repair)]
    own = ledger.manifest.get('canonical_version')
    out = {}
    for rid in ledger.inputs():
        if states[rid]['ok']:
            out[rid] = (states[rid]['ok']['record'], own)
            continue
        for results in fixed:
            if results.get(rid, {}).get('outcome') == 'ok':
                out[rid] = (results[rid]['record'], results[rid].get('canonical_version', own))
                break
    return out if with_canonical else {rid: rec for rid, (rec, _) in out.items()}


def export_complete(ledger, repair, path):
    """Final-delivery export over extraction + repair results; refuses (writes nothing) unless every frozen input has a
    validated record. Each record is re-checked with compact.validate_compact under the canonical contract that produced
    it (the run's, or a revalidation's successor)."""
    check_chain(ledger, _chain(repair))
    records = delivered(ledger, repair, with_canonical=True)
    missing = [rid for rid in ledger.inputs() if rid not in records]
    if missing:
        raise ValueError(f'final export refused: {len(missing)} of {len(ledger.inputs())} records have no validated result: {", ".join(missing[:5])}')
    for rid, (record, version) in records.items():
        compact.validate_compact(record, ledger.inputs()[rid], compact.CANONICAL_MODELS[version], contract=ledger.spec.contract)
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(compact_json(records[rid][0]) + '\n' for rid in ledger.inputs()))
    os.replace(tmp, path)
    return len(records)
