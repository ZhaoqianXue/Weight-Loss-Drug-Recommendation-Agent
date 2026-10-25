"""Single-call role extraction: candidate role-single-1.0.0 (spec compact-role1, GPT-6 Luna, reasoning medium).

One request per record on the full source text extracts every core group at once, with no stage-1 output, gold or hints.
Purposes carry their own evidence: a reason quote that holds the indication and, when that passage does not name the drug, an
anchor quote that does. A separate medication_decisions group keeps the decisions whose stated reason is not a treatment goal
of the drug: tolerability, dose concern, access, or another reason. Such a reason is never a purpose of the old or the new drug.
Side effects and no-side-effect items keep the compact-p7 fields and rules unchanged.

This module sends nothing. The runner sends request_body(inp), or RunLedger with ledger_kwargs(). It stores the raw Responses
JSON, and evaluate_response() judges it offline:

  status 'held': the reply has no usable answer. That is a refusal, truncation, incomplete or failed status, no single final
                 message, invalid JSON, or a top-level shape that is not the role schema. The raw reply is kept by the runner.
                 There is no record and no fallback, and the record stays in every denominator.
  status 'ok':   every item is checked on its own. An item that cannot be validated safely is rejected and logged, never
                 repaired by a model or re-asked. The other items continue. Accepted P/S/N items form a legacy canonical
                 compact record: compact-contract-2.1.0, readable by core_export 1.0.0 unchanged. In a purpose, ev[0] is the
                 reason span and ev[1], when present, the drug-anchor span. Accepted decisions form the sidecar.

The only deterministic repairs are the accepted ones, called unchanged:
- localize-1.3.0 for surfaces, S/N quotes and as-written fields;
- the contract-2.1.0 alias identity rule.
Reason and anchor passages may differ from the source only in whitespace runs or letter case. They get no gap restoration
and no join split. Code never fills in text, polarity, subject or drug attribution.

  python -m weightloss.extraction.role_single fingerprint
  python -m weightloss.extraction.role_single bodies --inputs INPUTS.jsonl --out BODIES.jsonl
  python -m weightloss.extraction.role_single export --inputs INPUTS.jsonl --responses RESPONSES.jsonl --run RUN_ID --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import model_validator

from . import compact, contract2, core_export, localize
from .compact import (DoseText, Handle, Label, Quote, ShortText, StatementKind, Subject, Surface, UseStatus, Wire, WireDrug,
                      WireExtraction, WireNoSideEffect, WirePurpose, WireSideEffect)
from .localize import splits_token_v2
from .structured import (ExtractionAnomaly, compact_json, final_output_text, parse_json_object, request_cap_usd, sha256_text,
                         validate_model)
from .variants import PROMPT_V7

VERSION = 'role-single-1.0.0'
SPEC_NAME = 'compact-role1'
SCHEMA_VERSION = 'compact-wire-role-1.0.0'
PROMPT_VERSION = 'compact-prompt-role-1.0.0'
SCHEMA_NAME = 'weightloss_role_extraction'
MODEL = 'gpt-6-luna'
REASONING = 'medium'
TEXT_VERBOSITY = 'low'
MAX_OUTPUT_TOKENS = 8192
LOCALIZE = localize.LOCALIZE_1_3
CONTRACT = contract2.CONTRACT_2_1
GROUPS = ('drugs', 'purposes', 'side_effects', 'no_side_effects', 'medication_decisions')
ACTIONS = ('start', 'stop', 'switch', 'resume', 'dose_change', 'dose_hold', 'schedule_change', 'other')
REASON_ROLES = ('tolerability', 'dose_concern', 'access', 'other')
PRONOUNS = frozenset('it that this these those them they which one'.split())

# PROMPT_V7 with these replacements; each target must occur exactly once. Every side-effect, no-side-effect, drug, subject,
# use and quote rule of compact-p7 stays verbatim; the copy rule only adds reason, and the quote rule only appends one sentence.
REPLACEMENTS = [
    ('- purposes: what a drug is used, prescribed or wanted for, as the text states it: a condition or goal given as the reason for use, '
     'or a condition the drug is said to treat or help (e.g. "for my type 2 diabetes", "prescribed due to PCOS", "I started it to lose '
     'weight", "I take it to quiet the food noise", "it helps my inflammation"). indication: that condition or goal as written (e.g. '
     '"type 2 diabetes", "lose weight"), never a result, amount, duration or dose. Not purposes: results by themselves (e.g. "lost 9 lbs", '
     '"A1C is 5.5", "the food noise is gone"), conditions that are only listed (e.g. "I also have PCOS"), and dosing or diet plans.',
     '- purposes: why a drug is taken, prescribed, recommended, wanted, continued, resumed or switched to, as the text states it: a '
     'condition or goal given as the reason, or a condition the drug is said to treat, relieve or help (e.g. "for my type 2 diabetes", '
     '"prescribed due to PCOS", "I started it to lose weight", "I take it to quiet the food noise", "switched to Zepbound because the '
     'cravings came back", "it helps my inflammation"). This includes a medicine or remedy taken for a symptom or side effect, also when '
     'the symptom is named in a nearby sentence and the text then says the remedy is used for it or helps (e.g. "My reflux was bad on '
     'the higher dose. Pepcid fixed it" -> Pepcid: "reflux"), and advice to take a drug for a condition (subject: whom the advice is '
     'for). indication: that condition or goal as written (e.g. "type 2 diabetes", "lose weight"), never a result, amount, duration or '
     'dose. reason_quote: one exact passage that contains the indication. anchor_quote: null when reason_quote names the drug (or the '
     'words used for it), else one exact passage that names that same drug. Not purposes: results or benefits the writer only noticed '
     '(e.g. "lost 9 lbs", "A1C is 5.5", "the food noise is gone", "my knee hurts less since starting it") unless the text gives them as '
     'a reason to take, keep or resume the drug; conditions that are only listed (e.g. "I also have PCOS"); dosing or diet plans without '
     'a stated goal; and reasons for stopping, switching or changing the dose or timing of a drug to avoid that drug\'s own side '
     'effects, from worry about its dose, or because of insurance, cost or supply: these belong to medication_decisions and are never a '
     'purpose of the old or the new drug, while another medicine taken to treat a symptom stays a purpose.'),
    ('- surface, indication and effect are copied character for character from the input (same spelling and capitalization), never '
     'normalized or translated; indication and effect name the condition, goal or symptom itself, never a pronoun such as "that" or "it" '
     'standing for it.',
     '- medication_decisions: a decision to start, stop, switch, resume, change or hold the dose of, or re-time a drug, for a stated '
     'reason that is not a condition or goal the drug is taken for: its own side effects or tolerability (e.g. "staying on 0.5 because the nausea '
     'was too much", "not going up yet so I can eat more"), worry about a dose (e.g. "scared to start at 10, so 7.5"), insurance, cost or '
     'supply, or another stated reason. One item per decision and reason. drug: the drug the decision is about (for a switch, the drug '
     'switched to); from_drug: for a switch, the drug switched from if the text refers to it, else null. action: "start", "stop", '
     '"switch", "resume", "dose_change" (a different, split or lower starting dose), "dose_hold" (not increasing), "schedule_change" '
     '(timing or interval) or "other" (e.g. injection site). reason: the reason as written, from inside reason_quote. reason_role: '
     '"tolerability", "dose_concern", "access" (insurance, coverage, cost, shortage) or "other". reason_quote and anchor_quote as for '
     'purposes (the anchor names drug, and from_drug, when reason_quote does not). polarity: whether the text affirms, negates or leaves '
     'uncertain that the action is or was taken. A side effect given as a reason is also a side_effects item. A decision whose reason is '
     'a condition or goal the drug is taken for (e.g. food noise, cravings, blood sugar) is a purpose instead.\n'
     '- surface, indication, reason and effect are copied character for character from the input (same spelling and capitalization), '
     'never normalized or translated; indication, reason and effect name the condition, goal, reason or symptom itself, never a pronoun '
     'such as "that" or "it" standing for it.'),
    ('- polarity (purposes): "affirmed"', '- polarity (purposes, medication decisions): "affirmed"'),
    ('Do not paraphrase, abbreviate with "..." or join separate passages.',
     'Do not paraphrase, abbreviate with "..." or join separate passages. reason_quote and anchor_quote follow the same rules, one '
     'passage each.'),
    ('If the input has no purpose, side effect or no-side-effect statement, return empty groups.',
     'If the input has no purpose, side effect, no-side-effect statement or medication decision, return empty groups.'),
]


def derive_prompt(text):
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError('role prompt replacement target must occur exactly once: ' + old[:60])
        text = text.replace(old, new)
    return text


INSTRUCTIONS = derive_prompt(PROMPT_V7)
Polarity = Optional[Literal['affirmed', 'negated', 'uncertain']]


class WireRolePurpose(Wire):
    drug: Handle
    reason_quote: Quote
    indication: Surface
    anchor_quote: Optional[Quote]
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    polarity: Polarity
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]

    @model_validator(mode='after')
    def _report_has_polarity(self):
        if self.statement_kind == 'report' and self.polarity is None:
            raise ValueError('a report must state polarity')
        return self


class WireDecision(Wire):
    drug: Handle
    from_drug: Optional[Handle]
    action: Literal['start', 'stop', 'switch', 'resume', 'dose_change', 'dose_hold', 'schedule_change', 'other']
    reason_quote: Quote
    reason: Surface
    reason_role: Literal['tolerability', 'dose_concern', 'access', 'other']
    anchor_quote: Optional[Quote]
    subject: Subject
    subject_label: Optional[Label]
    statement_kind: StatementKind
    polarity: Polarity
    use_status: UseStatus

    @model_validator(mode='after')
    def _decision_rules(self):
        if self.from_drug is not None and (self.action != 'switch' or self.from_drug == self.drug):
            raise ValueError('from_drug is only for a switch, and differs from drug')
        if self.statement_kind == 'report' and self.polarity is None:
            raise ValueError('a report must state polarity')
        return self


class WireRoleExtraction(Wire):
    drugs: list[WireDrug]
    purposes: list[WireRolePurpose]
    side_effects: list[WireSideEffect]
    no_side_effects: list[WireNoSideEffect]
    medication_decisions: list[WireDecision]


def _guard(answer):
    """compact-role1 answers are judged only by evaluate_response; the plain compact chain never accepts them."""
    raise ExtractionAnomaly('schema_invalid', 'compact-role1 replies are evaluated by role_single.evaluate_response only',
                            retriable=False, stage='structure')


SPEC = compact.register(compact.Spec(SPEC_NAME, INSTRUCTIONS, WireRoleExtraction, SCHEMA_VERSION, PROMPT_VERSION, schema_name=SCHEMA_NAME,
                                     to_canonical=_guard, text_verbosity=TEXT_VERBOSITY, contract=CONTRACT))
ITEM_MODELS = {'drugs': WireDrug, 'purposes': WireRolePurpose, 'side_effects': WireSideEffect, 'no_side_effects': WireNoSideEffect,
               'medication_decisions': WireDecision}
RULES = {'version': VERSION, 'spec': SPEC_NAME, 'localize': LOCALIZE, 'contract': CONTRACT,
         'held': 'no single parseable final JSON object (provider/protocol anomaly) or a top level that is not the five role groups',
         'item': 'each item validated alone with its model; any failed check rejects that item only, logged with reasons',
         'drug': 'unique handle; surface a token-bounded literal after localize-1.3.0; alias surfaces ref name (contract 2.1.0)',
         'side_effects/no_side_effects': 'compact-wire-1.0.0 item + localize-1.3.0 + compact-contract-2.1.0 checks, per item',
         'passage': 'reason/anchor quote: unique token-bounded exact passage, else unique passage equal after whitespace-run and case '
                    'normalization; no gap restoration, no join split',
         'indication/reason': 'token-bounded phrase inside its own reason passage (case/whitespace variant -> the one original phrase); '
                              'not a pronoun',
         'drug_named': 'the drug surface (and from_drug surface) occurs bounded in the reason or anchor passage, not as a condition '
                       'label (core_export heads)',
         'as_written': 'dose, duration, subject_label: localize-1.3.0 field rule, then a token-bounded literal',
         'pronouns': sorted(PRONOUNS), 'purpose_ev': 'ev[0] reason span; ev[1] anchor span unless inside the reason span',
         'decision_keys': ['drug', 'drug_surface', 'drug_ref', 'from_drug', 'from_drug_surface', 'from_drug_ref', 'action', 'reason',
                           'reason_role', 'subject', 'subject_label', 'statement_kind', 'polarity', 'use_status', 'ev', 'raw_index']}


def fingerprint():
    return {**SPEC.fingerprint(), 'candidate': VERSION, 'spec': SPEC_NAME, 'model': MODEL, 'reasoning_effort': REASONING,
            'max_output_tokens': MAX_OUTPUT_TOKENS, 'localize': LOCALIZE,
            'rules_sha256': hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()}


def ledger_kwargs():
    """Keyword arguments for runner.RunLedger.create that make ledger.request_body(inp) equal request_body(inp). The ledger's
    localize option stays unset: evaluate_response applies localize-1.3.0 itself."""
    return {'spec': SPEC, 'model': MODEL, 'reasoning_effort': REASONING, 'max_output_tokens': MAX_OUTPUT_TOKENS,
            'retry_policy': 'provider-only'}


def request_body(inp):
    """The exact Responses API body for one record (sent by the runner, never here)."""
    return compact.build_request(inp, spec=SPEC, model=MODEL, reasoning_effort=REASONING, max_output_tokens=MAX_OUTPUT_TOKENS)


def _span(hit):
    return '{}:{}-{}'.format(*hit)


def locate_passage(quote, fields):
    """(hit, how): the unique token-bounded exact passage ('exact'), else the unique passage equal after whitespace-run and case
    normalization ('normalized'); (None, 'ambiguous' | 'not_found') otherwise."""
    hits = contract2.literal_hits(quote, fields)
    if len(hits) == 1:
        return hits[0], 'exact'
    if hits:
        return None, 'ambiguous'
    loose = localize.normalized_spans(quote, fields, splits_token_v2)
    if len(loose) == 1:
        return loose[0], 'normalized'
    return None, 'ambiguous' if loose else 'not_found'


def phrase_in_passage(phrase, fields, hit):
    """(original phrase, how) for a phrase written inside the passage hit, on token boundaries; a case/whitespace variant becomes
    the one original phrase it matches there; (None, reason) otherwise."""
    if ' '.join(phrase.lower().split()) in PRONOUNS or not phrase.split():
        return None, 'pronoun'
    name, a, b = hit
    text = fields[name]
    i = text.find(phrase, a)
    while i != -1 and i + len(phrase) <= b:
        if not splits_token_v2(text, i, i + len(phrase)):
            return phrase, 'exact'
        i = text.find(phrase, i + 1)
    found = {text[s:e] for _, s, e in localize.normalized_spans(phrase, {name: text}, splits_token_v2) if a <= s and e <= b}
    if len(found) == 1:
        return found.pop(), 'normalized'
    return None, 'ambiguous' if found else 'not_in_passage'


def named_in(surface, fields, hits):
    """The drug surface occurs, on token boundaries and case-insensitively, inside one of the passages and not as a condition
    label ("Ozempic face")."""
    if core_export.CONDITION_SURFACE.search(surface):
        return False
    pattern = core_export.bounded(surface)
    for name, a, b in hits:
        text = fields[name]
        for m in pattern.finditer(text, a, b):
            if not core_export.CONDITION_LABEL.match(text[m.end():m.end() + 24]):
                return True
    return False


def _as_written(value, fields, name, path, repairs):
    """localize-1.3.0 field rule, then the contract-2.x literal check: (value, ok)."""
    if value in (None, ''):
        return value, True
    new, repair = localize.resolve_field(value, fields, name, LOCALIZE)
    if repair:
        repairs.append({'path': path, 'kind': 'field', **repair})
    return new, bool(contract2.literal_hits(new, fields, name in contract2.DOSE_FIELDS))


def _anomaly_reasons(error):
    out = error.as_dict()
    return [f"{out['kind']}" + (f": {', '.join(out.get('paths') or [])}" if out.get('paths') else f": {out['detail']}")]


class _Item:
    """Per-item evaluation state, so one item never changes another."""

    def __init__(self, group, index, raw):
        self.group, self.index, self.raw, self.reasons, self.repairs = group, index, raw, [], []

    @property
    def path(self):
        return f'{self.group}[{self.index}]'


def _drug_table(raw_drugs, fields, rejected, repairs):
    """{handle: (raw index, WireDrug)} of usable drugs (localized surface, contract-2.1.0 identity). A handle written more than once in
    the raw drugs group is unusable in every copy, schema-invalid copies included."""
    raw_handles = Counter(r.get('handle') for r in raw_drugs if isinstance(r.get('handle'), str))  # every raw copy counts
    parsed = []
    for i, raw in enumerate(raw_drugs):
        it = _Item('drugs', i, raw)
        try:
            drug = validate_model(WireDrug, raw)
        except ExtractionAnomaly as error:
            rejected.append({'group': 'drugs', 'index': i, 'reasons': ['schema: ' + error.detail], 'item': raw})
            continue
        surface, repair = localize.resolve_surface(drug.surface, fields, LOCALIZE)
        if repair:
            it.repairs.append({'path': f'drugs[{i}].surface', 'kind': 'surface', **repair})
        if not contract2.literal_hits(surface, fields):
            it.reasons.append('surface_not_found')
        parsed.append((it, drug.model_copy(update={'surface': surface})))
    table = {}
    for it, drug in parsed:
        if raw_handles[drug.handle] > 1:
            it.reasons.append('duplicate_handle')
        if it.reasons:
            rejected.append({'group': 'drugs', 'index': it.index, 'reasons': it.reasons, 'item': it.raw})
            continue
        repairs.extend(it.repairs)
        table[drug.handle] = (it.index, drug)
    wire = WireExtraction(drugs=[d for _, d in table.values()], purposes=[], side_effects=[], no_side_effects=[])
    normalized, changes = contract2.normalize_identity(wire)
    index_of = {d.handle: i for i, d in table.values()}
    for change in changes:
        handle = wire.drugs[int(change['path'][6:].split(']')[0])].handle
        repairs.append({**change, 'path': f'drugs[{index_of[handle]}].ref'})
    return {d.handle: (index_of[d.handle], d) for d in normalized.drugs}


def _one_item_wire(drug, group, item):
    groups = {'purposes': [], 'side_effects': [], 'no_side_effects': []}
    groups[group] = [item]
    return WireExtraction(drugs=[drug], **groups)


def _core_item(it, group, table, fields):
    """Side effect / no-side-effect item: compact-wire-1.0.0 model, localize-1.3.0, compact-contract-2.1.0 checks, per item."""
    try:
        item = validate_model(ITEM_MODELS[group], it.raw)
    except ExtractionAnomaly as error:
        it.reasons.append('schema: ' + error.detail)
        return None
    if item.drug not in table:
        it.reasons.append('drug_invalid')
        return None
    wire, found = localize.apply(_one_item_wire(table[item.drug][1], group, item), fields, LOCALIZE)
    it.repairs.extend({**r, 'path': r['path'].replace(f'{group}[0]', it.path, 1)} for r in found if r['path'].startswith(group))
    try:
        wire = validate_model(WireExtraction, wire.model_dump())
        contract2.check(wire, fields)
    except ExtractionAnomaly as error:
        it.reasons.extend(r.replace(f'{group}[0]', it.path) for r in _anomaly_reasons(error))
        return None
    return getattr(wire, group)[0]


def _passages(it, raw, fields, key, table, named):
    """Locate reason/anchor passages and the reason phrase; check the drug(s) are named. Returns (reason_hit, anchor_hit, phrase)."""
    reason_hit, how = locate_passage(raw.reason_quote, fields)
    if reason_hit is None:
        it.reasons.append(f'reason_quote_{how}')
    elif how != 'exact':
        it.repairs.append({'path': f'{it.path}.reason_quote', 'kind': 'quote', 'from': raw.reason_quote,
                           'to': fields[reason_hit[0]][reason_hit[1]:reason_hit[2]], 'span': _span(reason_hit)})
    anchor_hit = None
    if raw.anchor_quote is not None:
        anchor_hit, ahow = locate_passage(raw.anchor_quote, fields)
        if anchor_hit is None:
            it.reasons.append(f'anchor_quote_{ahow}')
        elif ahow != 'exact':
            it.repairs.append({'path': f'{it.path}.anchor_quote', 'kind': 'quote', 'from': raw.anchor_quote,
                               'to': fields[anchor_hit[0]][anchor_hit[1]:anchor_hit[2]], 'span': _span(anchor_hit)})
    phrase = None
    if reason_hit is not None:
        value = getattr(raw, key)
        phrase, phow = phrase_in_passage(value, fields, reason_hit)
        if phrase is None:
            it.reasons.append(f'{key}_{phow}')
        elif phow != 'exact':
            it.repairs.append({'path': f'{it.path}.{key}', 'kind': 'surface', 'from': value, 'to': phrase})
    hits = [h for h in (reason_hit, anchor_hit) if h is not None]
    for handle_field in named:
        handle = getattr(raw, handle_field)
        if handle is None:
            continue
        if handle not in table:
            it.reasons.append(f'{handle_field}_invalid')
        elif hits and not named_in(table[handle][1].surface, fields, hits):
            it.reasons.append(f'{handle_field}_not_named_in_passages')
    return reason_hit, anchor_hit, phrase


def _anchor_inside(reason_hit, anchor_hit):
    return anchor_hit is None or (anchor_hit[0] == reason_hit[0] and reason_hit[1] <= anchor_hit[1] and anchor_hit[2] <= reason_hit[2])


def _purpose(it, table, fields):
    try:
        raw = validate_model(WireRolePurpose, it.raw)
    except ExtractionAnomaly as error:
        it.reasons.append('schema: ' + error.detail)
        return None
    reason_hit, anchor_hit, indication = _passages(it, raw, fields, 'indication', table, ('drug',))
    values = {}
    for name in ('dose', 'duration', 'subject_label'):
        values[name], ok = _as_written(getattr(raw, name), fields, name, f'{it.path}.{name}', it.repairs)
        if not ok:
            it.reasons.append(f'field_not_found: {name}')
    if it.reasons:
        return None
    inside = _anchor_inside(reason_hit, anchor_hit)
    hits = [reason_hit] if inside else [reason_hit, anchor_hit]
    try:
        item = validate_model(WirePurpose, {'drug': raw.drug, 'indication': indication, 'subject': raw.subject, 'subject_label': values['subject_label'],
                                            'use_status': raw.use_status, 'dose': values['dose'], 'duration': values['duration'],
                                            'statement_kind': raw.statement_kind, 'polarity': raw.polarity,
                                            'quotes': [fields[n][a:b] for n, a, b in hits]})
        contract2.check(_one_item_wire(table[raw.drug][1], 'purposes', item), fields)
    except ExtractionAnomaly as error:
        it.reasons.extend(r.replace('purposes[0]', it.path) for r in _anomaly_reasons(error))
        return None
    return item, {'reason': _span(reason_hit), 'anchor': None if anchor_hit is None else _span(anchor_hit), 'anchor_in_reason': inside and anchor_hit is not None}


def _decision(it, table, fields):
    try:
        raw = validate_model(WireDecision, it.raw)
    except ExtractionAnomaly as error:
        it.reasons.append('schema: ' + error.detail)
        return None
    reason_hit, anchor_hit, reason = _passages(it, raw, fields, 'reason', table, ('drug', 'from_drug'))
    label, ok = _as_written(raw.subject_label, fields, 'subject_label', f'{it.path}.subject_label', it.repairs)
    if not ok:
        it.reasons.append('field_not_found: subject_label')
    if it.reasons:
        return None
    hits = [reason_hit] if _anchor_inside(reason_hit, anchor_hit) else [reason_hit, anchor_hit]
    drug, from_drug = table[raw.drug][1], table[raw.from_drug][1] if raw.from_drug else None
    return {'drug': raw.drug, 'drug_surface': drug.surface, 'drug_ref': drug.ref, 'from_drug': raw.from_drug,
            'from_drug_surface': from_drug.surface if from_drug else None, 'from_drug_ref': from_drug.ref if from_drug else None,
            'action': raw.action, 'reason': reason, 'reason_role': raw.reason_role, 'subject': raw.subject, 'subject_label': label,
            'statement_kind': raw.statement_kind, 'polarity': raw.polarity, 'use_status': raw.use_status, 'ev': [_span(h) for h in hits]}


def _held(record_id, run, anomaly):
    return {'record_id': record_id, 'run': run, 'version': VERSION, 'status': 'held', 'anomaly': anomaly, 'record': None, 'decisions': [],
            'rejected': [], 'repairs': [], 'purpose_evidence': [], 'retention': {'keep': False, 'basis': []},
            'counts': {'raw': {}, 'accepted': {}, 'rejected': {}}}


def evaluate_response(response, inp, run):
    """Raw Responses JSON -> evaluation dict (see the module docstring). Pure, offline and deterministic; never raises on model
    content."""
    try:
        value = parse_json_object(final_output_text(response))
    except ExtractionAnomaly as anomaly:
        return _held(inp.record_id, run, anomaly.as_dict())
    if set(value) != set(GROUPS) or not all(isinstance(value[g], list) and all(isinstance(x, dict) for x in value[g]) for g in GROUPS):
        return _held(inp.record_id, run, ExtractionAnomaly('schema_invalid', 'top level is not the five role groups', stage='structure').as_dict())
    return evaluate_value(value, inp, run)


def evaluate_value(value, inp, run):
    """The item-level chain of evaluate_response for an already parsed top-level object with the five role groups."""
    fields = dict(inp.fields)
    rejected, repairs = [], []
    table = _drug_table(value['drugs'], fields, rejected, repairs)
    accepted = {'purposes': [], 'side_effects': [], 'no_side_effects': []}
    evidence, decisions = [], []
    for group in ('purposes', 'side_effects', 'no_side_effects', 'medication_decisions'):
        for k, raw in enumerate(value[group]):
            it = _Item(group, k, raw)
            if group == 'purposes':
                out = _purpose(it, table, fields)
            elif group == 'medication_decisions':
                out = _decision(it, table, fields)
            else:
                out = _core_item(it, group, table, fields)
            if out is None:
                rejected.append({'group': group, 'index': k, 'reasons': it.reasons, 'item': raw})
                continue
            repairs.extend(it.repairs)
            if group == 'purposes':
                accepted[group].append(out[0])
                evidence.append({'index': len(accepted[group]) - 1, 'raw_index': k, **out[1]})
            elif group == 'medication_decisions':
                decisions.append({**out, 'raw_index': k})
            else:
                accepted[group].append(out)
    used = {item.drug for g in accepted for item in accepted[g]}
    wire = WireExtraction(drugs=[d for h, (_, d) in table.items() if h in used], **accepted)
    record = compact.to_compact(wire, contract2.check(wire, fields), inp.record_id, run)
    compact.validate_compact(record, inp, canonical_wire=WireExtraction, contract=CONTRACT)  # raises only on a bug here
    basis = [compact.RETENTION_BASIS[g] for g in compact.GROUPS if record[g]] + (['medication_decision'] if decisions else [])
    count = Counter(r['group'] for r in rejected)
    return {'record_id': inp.record_id, 'run': run, 'version': VERSION, 'status': 'ok', 'anomaly': None, 'record': record,
            'decisions': decisions, 'rejected': rejected, 'repairs': repairs, 'purpose_evidence': evidence,
            'retention': {'keep': bool(basis), 'basis': basis},
            'counts': {'raw': {g: len(value[g]) for g in GROUPS}, 'accepted': {**{g: len(record[g]) for g in compact.GROUPS},
                                                                                  'drugs': len(table), 'medication_decisions': len(decisions)},
                       'rejected': {g: count.get(g, 0) for g in GROUPS}}}


def _response_sha(response):
    return sha256_text(compact_json(response))


def export(inputs, responses, run):
    """(core_rows, decision_rows, provenance_rows, held_rows, summary) for every input record, in input order.

    responses: rows {'custom_id': '<record_id>@<attempt>', 'response': raw Responses JSON} (runner responses.jsonl). Per record
    the earliest attempt that evaluates 'ok' is used; otherwise the record is held with its latest anomaly (or no_response)."""
    by_record = {rid: [] for rid in inputs}
    seen = set()
    for row in responses:
        cid = row['custom_id']
        rid, _, attempt = cid.rpartition('@')
        if cid in seen:
            raise ValueError(f'duplicate response {cid}')
        if rid not in inputs:
            raise ValueError(f'response {cid} has no input record')
        seen.add(cid)
        by_record[rid].append((int(attempt), cid, row['response']))
    core, decisions, provenance, held = [], [], [], []
    for rid, inp in inputs.items():
        tried = []
        for attempt, cid, response in sorted(by_record[rid]):
            tried.append((attempt, cid, response, evaluate_response(response, inp, run)))
        oks = [t for t in tried if t[3]['status'] == 'ok']
        used = oks[0] if oks else None
        if used is not None:
            ev = used[3]
        elif tried:
            ev = tried[-1][3]
        else:
            ev = _held(rid, run, {'kind': 'no_response', 'detail': 'no saved reply for this record', 'retriable': True, 'stage': 'provider'})
        if ev['status'] == 'ok':
            core.append(ev['record'])
        else:
            held.append({'record_id': rid, 'anomaly': ev['anomaly'], 'attempts': [t[1] for t in tried]})
        decisions.append({'record_id': rid, 'run': run, 'version': VERSION, 'status': ev['status'], 'custom_id': used[1] if used else None,
                          'decisions': ev['decisions']})
        provenance.append({'record_id': rid, 'run': run, 'version': VERSION, 'status': ev['status'], 'custom_id': used[1] if used else None,
                           'response_sha256': _response_sha(used[2]) if used else None,
                           'attempts': [{'custom_id': t[1], 'status': t[3]['status'], 'anomaly': t[3]['anomaly'],
                                         'response_sha256': _response_sha(t[2])} for t in tried],
                           **{k: ev[k] for k in ('anomaly', 'rejected', 'repairs', 'purpose_evidence', 'retention', 'counts')}})
    reasons = Counter(r.split(':')[0] for p in provenance for x in p['rejected'] for r in x['reasons'])
    summary = {**fingerprint(), 'run': run, 'records': len(inputs), 'responses': len(seen),
               'status': dict(Counter(p['status'] for p in provenance)),
               'held_anomalies': dict(Counter(h['anomaly']['kind'] for h in held)),
               'accepted': {g: sum(len(r[g]) for r in core) for g in compact.GROUPS} | {'medication_decisions': sum(len(d['decisions']) for d in decisions)},
               'rejected_items': dict(Counter(x['group'] for p in provenance for x in p['rejected'])),
               'rejection_reasons': dict(reasons), 'repairs': dict(Counter(r['kind'] for p in provenance for r in p['repairs'])),
               'records_with_decisions_only': sum(1 for p, d in zip(provenance, decisions)
                                                  if d['decisions'] and p['status'] == 'ok' and p['retention']['basis'] == ['medication_decision']),
               'retry_records': sum(1 for p in provenance if len(p['attempts']) > 1), 'api_calls_here': 0}
    return core, decisions, provenance, held, summary


def _jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def load_inputs(path):
    rows = _jl(path)
    ids = [r['record_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate record ids in inputs')
    return {r['record_id']: compact.ExtractionInput(r['record_id'], r['fields'], r['kind']) for r in rows}


def _write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fingerprint')
    b = sub.add_parser('bodies')
    b.add_argument('--inputs', required=True)
    b.add_argument('--out', required=True)
    e = sub.add_parser('export')
    for name in ('--inputs', '--responses', '--run', '--out-dir'):
        e.add_argument(name, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'fingerprint':
        print(json.dumps(fingerprint(), indent=1))
        return 0
    if a.cmd == 'bodies':
        if Path(a.out).exists() or Path(a.out).resolve() == Path(a.inputs).resolve():
            raise SystemExit(f'{a.out} exists; outputs are written once')
        lines = []
        for rid, inp in load_inputs(a.inputs).items():
            body = request_body(inp)
            lines.append({'record_id': rid, 'custom_id': f'{rid}@1', 'candidate': VERSION, 'body_sha256': sha256_text(compact_json(body)),
                          'cap_usd': round(request_cap_usd(MODEL, body), 6), 'body': body})
        _write(a.out, lines)
        print(json.dumps({'bodies': len(lines), 'cap_usd_sum': round(sum(x['cap_usd'] for x in lines), 6), **fingerprint()}))
        return 0
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f'{out} exists; outputs are written once')
    inputs = load_inputs(a.inputs)
    core, decisions, provenance, held, summary = export(inputs, _jl(a.responses), a.run)
    out.mkdir(parents=True)
    names = {'core': core, 'decisions': decisions, 'provenance': provenance, 'held': held}
    for name, rows in names.items():
        _write(out/f'{name}.jsonl', rows)
    summary['sha256'] = {'inputs': hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
                         'responses': hashlib.sha256(Path(a.responses).read_bytes()).hexdigest(),
                         **{name: hashlib.sha256((out/f'{name}.jsonl').read_bytes()).hexdigest() for name in names}}
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('records', 'status', 'accepted', 'rejected_items')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
