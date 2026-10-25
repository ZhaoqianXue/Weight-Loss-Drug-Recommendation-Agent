"""Compact drug-statement extraction: API wire schema, local validation and the deterministic compact conversion.

Field meanings follow the accepted compact candidate and KG record schema v1.1 (results/schema-design-2026-09-27 in the
reddit-batch-budget worktree). The model answers in the wire format: every key present, null when the text does not
say, drugs as a list of handles, evidence as verbatim quotes. `to_compact` turns a validated answer into the stored
format: nulls and empty lists dropped, drugs keyed by handle, each quote replaced by the code-located span
"field:start-end" (Python str offsets, end exclusive) over the exact input text. Absence never means self, current,
affirmed or attributed; the Drug/Indication/SideEffect graph is derived later from these groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Annotated, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .structured import (ExtractionAnomaly, build_body, final_output_text, parse_json_object, schema_sha256,
                         sha256_text, strict_json_schema, validate_model)

SCHEMA_NAME = 'weightloss_compact_extraction'
SCHEMA_VERSION = 'compact-wire-1.0.0'
PROMPT_VERSION = 'compact-prompt-1.0.0'
DEFAULT_MODEL = 'gpt-6-luna'
DEFAULT_REASONING = 'none'
DEFAULT_MAX_OUTPUT_TOKENS = 4096
EVIDENCE_FIELDS = ('text', 'title', 'review', 'condition_field', 'drug_field')
SOURCE_KINDS = {'reddit': ('post', 'comment'), 'webmd': ('review',)}
RECORD_ID = re.compile(r'^(reddit|webmd):[A-Za-z0-9_-]+$')
GROUPS = ('purposes', 'side_effects', 'no_side_effects')
RETENTION_BASIS = {'purposes': 'purpose', 'side_effects': 'side_effect', 'no_side_effects': 'explicit_none'}

INSTRUCTIONS = """You extract drug statements from one social-media post or health review for a knowledge graph. Use only the given text: no outside knowledge, no guesses. Nothing is pre-filled and nothing has a default; never assume the writer, current use, an affirmed purpose or an attributed effect.

Groups (every key present; null when the text does not say; [] when a group is empty):
- drugs: one entry per medication that an item below refers to. handle: d1, d2, ... in order of first mention. surface: the drug words as written in the text. ref: "name" (brand or generic name), "class" (a drug class only, e.g. "GLP-1") or "unspecified" ("this med", "the shot"). Never upgrade a class or unspecified reference to a brand.
- purposes: what a drug is used for or would be used for, as stated. indication: the condition or goal as written in the text. An outcome alone (e.g. "I lost 20 lbs") is not a purpose.
- side_effects: side effects the text links to a drug, whether they occurred, did not occur, are uncertain, feared or asked about. effect: the symptom as written in the text.
- no_side_effects: statements that a drug caused no side effects. scope_text: the limiting words as written when the claim is narrower than none at all (e.g. "no serious side effects", "none so far"), else null. Never widen the scope.

surface, indication and effect are copied character for character from the text (same spelling and capitalization), never normalized or translated.

Item fields (independent of each other; never fill one from another):
- drug: the handle of the drug the statement is about.
- subject: "self" (the writer), "other" (a specific other person), "general" (people in general), or null when unclear. subject_label: a short label from the text only when subject is "other" (e.g. "sister"), else null.
- statement_kind: "report" (declarative), "question", or "hypothetical" (planned, considered, feared or conditional).
- polarity (purposes): "affirmed", "negated" or "uncertain"; null only for a pure question that asserts nothing. A report always has a polarity.
- occurrence (side effects): "occurred", "not_occurred", "uncertain", or "not_asserted" (no evidence either way, e.g. a pure fear or a question about a future effect). "I have nausea; could Ozempic be causing it?" is occurred + question + uncertain attribution.
- attribution (side effects; the author's view, not medical causality): "attributed", "temporal_only" (only a time relation is stated, e.g. "after each shot"), "uncertain", "other_cause" or "not_stated".
- use_status: "current", "past", "planned", "considering", "not_using" or "unknown" as stated for this drug and subject; null when the text does not say. Planned or considering is not actual use.
- dose: the general dose as written (e.g. "0.5 mg"). duration: the total time on the drug as written (e.g. "six weeks"); it never fills onset or event_duration. Repeat use_status, dose and duration on every item of the same use episode (same person, same drug, explicitly the same episode); keep different episodes apart.
- severity ("mild", "moderate", "severe"), event_dose (a dose the text ties to the event), onset (e.g. "two weeks after stopping"), event_duration, clinical_consequences ("medical_visit", "emergency_visit", "hospitalization", "other"; [] when none stated) and outcome ("resolved", "ongoing", "unknown"): only when stated for that effect. They stay null or [] when occurrence is "not_occurred" or "not_asserted".
- quotes: 1 to 8 passages copied exactly from the text, character for character, that support the item and its fields; each long enough to occur only once in the text. Do not paraphrase, abbreviate with "..." or join separate passages.

If the text has no purpose, side effect or no-side-effect statement, return empty groups."""

Handle = Annotated[str, Field(pattern=r'^d[1-9][0-9]*$')]
Surface = Annotated[str, Field(min_length=1, max_length=200)]
Label = Annotated[str, Field(min_length=1, max_length=40)]
DoseText = Annotated[str, Field(min_length=1, max_length=60)]
ShortText = Annotated[str, Field(min_length=1, max_length=100)]
Quote = Annotated[str, Field(min_length=1, max_length=1000)]
Quotes = Annotated[list[Quote], Field(min_length=1, max_length=8)]
Subject = Optional[Literal['self', 'other', 'general']]
UseStatus = Optional[Literal['current', 'past', 'planned', 'considering', 'not_using', 'unknown']]
StatementKind = Literal['report', 'question', 'hypothetical']
Consequence = Literal['medical_visit', 'emergency_visit', 'hospitalization', 'other']
EVENT_FIELDS = ('severity', 'event_dose', 'onset', 'event_duration', 'clinical_consequences', 'outcome')


class Wire(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    @model_validator(mode='after')
    def _subject_label_only_for_other(self):
        if 'subject' in type(self).model_fields and self.subject != 'other' and self.subject_label is not None:
            raise ValueError('subject_label is only allowed when subject is "other"')
        return self


class WireDrug(Wire):
    handle: Handle
    surface: Surface
    ref: Literal['name', 'class', 'unspecified']


class WirePurpose(Wire):
    drug: Handle
    indication: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    polarity: Optional[Literal['affirmed', 'negated', 'uncertain']]
    quotes: Quotes

    @model_validator(mode='after')
    def _report_has_polarity(self):
        if self.statement_kind == 'report' and self.polarity is None:
            raise ValueError('a report must state polarity')
        return self


class WireSideEffect(Wire):
    drug: Handle
    effect: Surface
    subject: Subject
    subject_label: Optional[Label]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    statement_kind: StatementKind
    occurrence: Literal['occurred', 'not_occurred', 'uncertain', 'not_asserted']
    attribution: Literal['attributed', 'temporal_only', 'uncertain', 'other_cause', 'not_stated']
    severity: Optional[Literal['mild', 'moderate', 'severe']]
    event_dose: Optional[DoseText]
    onset: Optional[ShortText]
    event_duration: Optional[ShortText]
    clinical_consequences: list[Consequence]
    outcome: Optional[Literal['resolved', 'ongoing', 'unknown']]
    quotes: Quotes

    @model_validator(mode='after')
    def _event_fields(self):
        if len(set(self.clinical_consequences)) != len(self.clinical_consequences):
            raise ValueError('clinical_consequences must not repeat')
        if self.occurrence in ('not_occurred', 'not_asserted'):
            filled = [name for name in EVENT_FIELDS if getattr(self, name) not in (None, [])]
            if filled:
                raise ValueError(f'{filled} must be empty when occurrence is {self.occurrence}')
        return self


class WireNoSideEffect(Wire):
    drug: Handle
    subject: Subject
    subject_label: Optional[Label]
    scope_text: Optional[ShortText]
    use_status: UseStatus
    dose: Optional[DoseText]
    duration: Optional[ShortText]
    quotes: Quotes


class WireExtraction(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffect]
    no_side_effects: list[WireNoSideEffect]


class WireSideEffectSev(WireSideEffect):
    """Canonical side effect, severity-v2 semantics (opt-in per spec; compact-wire-1.0.0 is unchanged): severity is the
    intensity the text gives the effect ("no severe pain", "afraid of severe nausea"), so it may stand with occurrence
    not_occurred or not_asserted without making the effect occur. Event dose, onset, event duration, clinical
    consequences and outcome still describe an event that happened and stay empty for those occurrences."""

    @model_validator(mode='after')
    def _event_fields(self):
        if len(set(self.clinical_consequences)) != len(self.clinical_consequences):
            raise ValueError('clinical_consequences must not repeat')
        if self.occurrence in ('not_occurred', 'not_asserted'):
            filled = [name for name in EVENT_FIELDS if name != 'severity' and getattr(self, name) not in (None, [])]
            if filled:
                raise ValueError(f'{filled} must be empty when occurrence is {self.occurrence}')
        return self


class WireExtractionSev(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffectSev]
    no_side_effects: list[WireNoSideEffect]



class WireSideEffectNegDose(WireSideEffect):
    """Canonical side effect, negated-dose semantics (opt-in per spec as compact-canonical-negdose-1.0.0; the wire JSON
    schema and compact-wire-1.0.0 are unchanged): a not_occurred effect may keep event_dose, the dose at which the text
    says the symptom did not happen ("didn't barf on 5" -> "5"). Severity, onset, event duration, clinical consequences
    and outcome describe an event that happened and stay empty for not_occurred; not_asserted keeps every event field
    empty. Every compact-wire-1.0.0 record is also valid here."""

    @model_validator(mode='after')
    def _event_fields(self):
        if len(set(self.clinical_consequences)) != len(self.clinical_consequences):
            raise ValueError('clinical_consequences must not repeat')
        if self.occurrence in ('not_occurred', 'not_asserted'):
            allowed = ('event_dose',) if self.occurrence == 'not_occurred' else ()
            filled = [name for name in EVENT_FIELDS if name not in allowed and getattr(self, name) not in (None, [])]
            if filled:
                raise ValueError(f'{filled} must be empty when occurrence is {self.occurrence}')
        return self


class WireExtractionNegDose(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffectNegDose]
    no_side_effects: list[WireNoSideEffect]


class WireSideEffectNegDoseSev(WireSideEffect):
    """Canonical side effect compact-canonical-negdose-sev-1.0.0 (opt-in; the wire schema is unchanged). It is
    negdose-1.0.0 plus severity-v2. A not_occurred effect may keep event_dose and severity. A not_asserted effect (a
    question, fear or hypothetical) may keep severity, the intensity the text gives the mentioned effect ("intense return
    of hunger?"), without that meaning the effect occurred. Onset, event duration, clinical consequences and outcome
    describe an event that happened and stay empty for both occurrences; event_dose also stays empty for not_asserted.
    Every compact-wire-1.0.0 and negdose-1.0.0 record is also valid here."""

    @model_validator(mode='after')
    def _event_fields(self):
        if len(set(self.clinical_consequences)) != len(self.clinical_consequences):
            raise ValueError('clinical_consequences must not repeat')
        if self.occurrence in ('not_occurred', 'not_asserted'):
            allowed = ('event_dose', 'severity') if self.occurrence == 'not_occurred' else ('severity',)
            filled = [name for name in EVENT_FIELDS if name not in allowed and getattr(self, name) not in (None, [])]
            if filled:
                raise ValueError(f'{filled} must be empty when occurrence is {self.occurrence}')
        return self


class WireExtractionNegDoseSev(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffectNegDoseSev]
    no_side_effects: list[WireNoSideEffect]


CANONICAL_MODELS = {None: WireExtraction, 'compact-canonical-negdose-1.0.0': WireExtractionNegDose,
                    'compact-canonical-negdose-sev-1.0.0': WireExtractionNegDoseSev}


WIRE_SCHEMA = strict_json_schema(WireExtraction)


@dataclass(frozen=True)
class Spec:
    """A frozen request/answer definition. Experimental specs may use another prompt or wire model; to_canonical maps
    their parsed answer to the canonical wire dict, so validation, spans and the stored compact format stay identical.
    A spec is never edited once a run used it: changes get a new name."""
    name: str
    instructions: str
    wire_model: type
    schema_version: str
    prompt_version: str
    schema_name: str = SCHEMA_NAME
    to_canonical: object = None
    render: object = None
    text_verbosity: object = None  # request parameter frozen with the spec (None = API default)
    canonical: object = None  # (version, canonical wire model) for an opt-in canonical version; None = compact-wire-1.0.0
    contract: object = None  # versioned local contract (e.g. contract2.CONTRACT_2); None = the historical rules
    schema: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'schema', WIRE_SCHEMA if self.wire_model is WireExtraction else strict_json_schema(self.wire_model))

    @property
    def canonical_wire(self):
        return self.canonical[1] if self.canonical else WireExtraction

    def fingerprint(self):
        out = {'schema_name': self.schema_name, 'schema_version': self.schema_version, 'schema_sha256': schema_sha256(self.schema),
               'prompt_version': self.prompt_version, 'prompt_sha256': sha256_text(self.instructions)}
        if self.text_verbosity is not None:
            out['text_verbosity'] = self.text_verbosity
        if self.canonical is not None:
            out['canonical_version'] = self.canonical[0]
        if self.contract is not None:
            out['contract_version'] = self.contract
        return out


CANONICAL = Spec('compact-v1', INSTRUCTIONS, WireExtraction, SCHEMA_VERSION, PROMPT_VERSION)
SPECS = {CANONICAL.name: CANONICAL}


def register(spec):
    known = SPECS.get(spec.name)
    if known is not None and known.fingerprint() != spec.fingerprint():
        raise ValueError(f'spec {spec.name} is already registered with a different fingerprint')
    SPECS[spec.name] = spec
    return spec


def get_spec(name):
    from . import variants  # noqa: F401  (registers experimental specs)
    if name not in SPECS and name.startswith('compact-r1@'):
        variants.review_spec(name.split('@', 1)[1])  # review spec bound to one frozen context file
    if name not in SPECS:
        raise ValueError(f'unknown extraction spec {name!r}')
    return SPECS[name]


@dataclass(frozen=True)
class ExtractionInput:
    """One source record: record_id '<platform>:<source id>' and the exact text of each evidence field."""
    record_id: str
    fields: Mapping[str, str]
    kind: str

    def __post_init__(self):
        if not RECORD_ID.match(self.record_id):
            raise ValueError(f'invalid record_id {self.record_id!r}; expected <reddit|webmd>:<id>')
        if self.kind not in SOURCE_KINDS[self.platform]:
            raise ValueError(f'invalid kind {self.kind!r} for {self.platform}')
        if not self.fields or any(name not in EVIDENCE_FIELDS for name in self.fields):
            raise ValueError(f'fields must be a non-empty subset of {EVIDENCE_FIELDS}')
        if any(not isinstance(v, str) or not v.strip() for v in self.fields.values()):
            raise ValueError(f'{self.record_id}: empty or non-text field')

    @property
    def platform(self):
        return self.record_id.split(':', 1)[0]

    @property
    def text_sha256(self):
        """Version of the exact text the evidence spans refer to."""
        return hashlib.sha256(json.dumps(dict(self.fields), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def render_input(inp):
    label = {'reddit': 'Reddit', 'webmd': 'WebMD'}[inp.platform]
    blocks = [f'{name}:\n<<<\n{text}\n>>>' for name, text in inp.fields.items()]
    return f'{label} {inp.kind}\n' + '\n'.join(blocks)


def build_request(inp, *, spec=None, model=DEFAULT_MODEL, reasoning_effort=DEFAULT_REASONING, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    spec = spec or CANONICAL
    return build_body(model=model, instructions=spec.instructions, user_content=(spec.render or render_input)(inp),
                      schema_name=spec.schema_name, schema=spec.schema, reasoning_effort=reasoning_effort,
                      max_output_tokens=max_output_tokens, store=False, text_verbosity=spec.text_verbosity)


def schema_fingerprint(spec=None):
    return (spec or CANONICAL).fingerprint()


def items(wire):
    for group in GROUPS:
        for index, item in enumerate(getattr(wire, group)):
            yield group, index, item


def check_references(wire):
    handles = [d.handle for d in wire.drugs]
    repeated = sorted({h for h in handles if handles.count(h) > 1})
    if repeated:
        raise ExtractionAnomaly('drug_ref_invalid', f'repeated handles {repeated}')
    unknown = [f'{group}[{i}].drug={item.drug}' for group, i, item in items(wire) if item.drug not in handles]
    if unknown:
        raise ExtractionAnomaly('drug_ref_invalid', 'unknown handles: ' + ', '.join(unknown[:5]))


def check_surfaces(wire, fields):
    """Surfaces are phrases as written: each must occur verbatim somewhere in the input (occurrence count is irrelevant)."""
    missing = [f'drugs[{i}].surface' for i, d in enumerate(wire.drugs) if not any(d.surface in t for t in fields.values())]
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect')):
        missing += [f'{group}[{i}].{name}' for i, item in enumerate(getattr(wire, group))
                    if not any(getattr(item, name) in t for t in fields.values())]
    if missing:
        raise ExtractionAnomaly('surface_not_found', 'not verbatim in the input: ' + ', '.join(missing[:5]))


def locate(quote, fields):
    """Every (field, start, end) where the quote occurs verbatim, overlapping occurrences included."""
    hits = []
    for name, text in fields.items():
        start = text.find(quote)
        while start != -1:
            hits.append((name, start, start + len(quote)))
            start = text.find(quote, start + 1)
    return hits


def locate_evidence(wire, fields):
    """Spans per item, in quote order. A quote found nowhere or in several places is an anomaly, never guessed."""
    spans, missing, ambiguous = {}, [], []
    for group, i, item in items(wire):
        found = []
        for q, quote in enumerate(item.quotes):
            hits = locate(quote, fields)
            if not hits:
                missing.append(f'{group}[{i}].quotes[{q}]')
            elif len(hits) > 1:
                ambiguous.append(f'{group}[{i}].quotes[{q}] ({len(hits)} matches)')
            else:
                found.append('{}:{}-{}'.format(*hits[0]))
        spans[group, i] = found
    if missing:
        raise ExtractionAnomaly('evidence_not_found', 'not verbatim in the input: ' + ', '.join(missing[:5]))
    if ambiguous:
        raise ExtractionAnomaly('evidence_ambiguous', 'several matches: ' + ', '.join(ambiguous[:5]))
    return spans


def to_compact(wire, spans, record_id, run):
    """Deterministic wire -> compact conversion. Drugs no item refers to are not stored (v1.1 has no carrier for them)."""
    used = {item.drug for _, _, item in items(wire)}
    record = {'record_id': record_id, 'run': run,
              'drugs': {d.handle: {'surface': d.surface, 'ref': d.ref} for d in wire.drugs if d.handle in used}}
    for group in GROUPS:
        record[group] = []
        for i, item in enumerate(getattr(wire, group)):
            entry = {k: v for k, v in item.model_dump().items() if k != 'quotes' and v is not None and v != []}
            entry['ev'] = spans[group, i]
            record[group].append(entry)
    return record


def retention(record):
    """Keep a record with any purpose, side-effect or explicit no-side-effect item, whatever its statement kind,
    polarity or occurrence."""
    basis = [RETENTION_BASIS[g] for g in GROUPS if record[g]]
    return {'keep': bool(basis), 'basis': basis}


def evaluate_response(response, inp, run, spec=None, localize_policy=None, repairs=None):
    """Raw Responses JSON -> validated compact record, or ExtractionAnomaly. Shared by sync and Batch results.

    localize_policy (opt-in, e.g. localize.LOCALIZE_1) first maps strings that are not verbatim but match the input
    uniquely after whitespace/case normalization back to their original substrings (see localize.py); each repair is
    appended to ``repairs``. Without a policy the behaviour is unchanged."""
    return evaluate_value(parse_json_object(final_output_text(response)), inp, run, spec, localize_policy, repairs)


def evaluate_value(value, inp, run, spec=None, localize_policy=None, repairs=None):
    """The validation chain of evaluate_response for an already parsed answer (also used by surface_repair to re-validate
    a repaired answer end to end)."""
    spec = spec or CANONICAL
    if spec.to_canonical is not None:
        value = spec.to_canonical(validate_model(spec.wire_model, value))
    wire = validate_model(spec.canonical_wire, value)
    check_references(wire)
    if localize_policy is not None:
        from . import localize
        wire, found = localize.apply(wire, inp.fields, localize_policy)
        if repairs is not None:
            repairs.extend(found)
        if found:  # repaired strings are original substrings; re-check the wire contract (e.g. length limits)
            wire = validate_model(spec.canonical_wire, wire.model_dump())
    if spec.contract is not None:
        from . import contract2
        if spec.contract not in contract2.CONTRACTS:
            raise ValueError(f'unknown contract {spec.contract!r}')
        if spec.contract == contract2.CONTRACT_2_1:
            wire, found = contract2.normalize_identity(wire)
            if repairs is not None:
                repairs.extend(found)
        return to_compact(wire, contract2.check(wire, inp.fields), inp.record_id, run)
    check_surfaces(wire, inp.fields)
    return to_compact(wire, locate_evidence(wire, inp.fields), inp.record_id, run)


def render_span(span, fields):
    name, _, offsets = span.rpartition(':')
    start, end = (int(x) for x in offsets.split('-'))
    if name not in fields or not 0 <= start < end <= len(fields[name]):
        raise ValueError(f'span {span} is outside the input text')
    return fields[name][start:end]


def compact_to_wire(record, fields):
    """Inverse view for audits and round-trip checks: nulls restored, spans rendered back to quotes."""
    names = {'purposes': WirePurpose, 'side_effects': WireSideEffect, 'no_side_effects': WireNoSideEffect}
    wire = {'drugs': [{'handle': h, **d} for h, d in record['drugs'].items()]}
    for group in GROUPS:
        wire[group] = []
        for entry in record[group]:
            full = {}
            for name in names[group].model_fields:
                if name == 'quotes':
                    full[name] = [render_span(s, fields) for s in entry['ev']]
                else:
                    full[name] = entry.get(name, [] if name == 'clinical_consequences' else None)
            wire[group].append(full)
    return wire


def validate_compact(record, inp, canonical_wire=None, contract=None):
    """Acceptance check for a stored compact record against its input text: exact key sets (no unknown keys, no
    stored nulls or empty lists), wire-schema validation, drug references, verbatim surfaces, and every span unique and
    verbatim. The record must equal the canonical conversion of its own content. canonical_wire is the canonical model
    of the spec that produced the record (default compact-wire-1.0.0)."""
    fields = {'purposes': WirePurpose, 'side_effects': WireSideEffect, 'no_side_effects': WireNoSideEffect}
    if set(record) != {'record_id', 'run', 'drugs', *GROUPS}:
        raise ValueError(f'record keys {sorted(record)} differ from the compact format')
    if record['record_id'] != inp.record_id:
        raise ValueError('record_id does not match the input')
    for handle, drug in record['drugs'].items():
        if not isinstance(drug, dict) or set(drug) != {'surface', 'ref'}:
            raise ValueError(f'drugs.{handle} must hold exactly surface and ref')
    for group in GROUPS:
        allowed = (set(fields[group].model_fields) - {'quotes'}) | {'ev'}
        for i, entry in enumerate(record[group]):
            unknown = set(entry) - allowed
            if unknown or 'ev' not in entry:
                raise ValueError(f'{group}[{i}] has unknown keys {sorted(unknown)} or no ev')
            if any(v is None or v == [] for v in entry.values()):
                raise ValueError(f'{group}[{i}] stores null or empty values; the compact format omits them')
    wire = validate_model(canonical_wire or WireExtraction, compact_to_wire(record, inp.fields))
    check_references(wire)
    if contract is not None:
        from . import contract2
        if contract == contract2.CONTRACT_2_1:  # an unnormalized alias makes the record differ from its canonical form
            wire, _ = contract2.normalize_identity(wire)
        spans = contract2.check(wire, inp.fields)
    else:
        check_surfaces(wire, inp.fields)
        spans = locate_evidence(wire, inp.fields)
    canonical = to_compact(wire, spans, record['record_id'], record['run'])
    if canonical != record:
        raise ValueError('record differs from the canonical compact form of its content (e.g. an unreferenced drug)')
    return wire
