"""Versioned field contract ``compact-contract-2.0.0`` (2026-09-30 engineering review; spec compact-p7t3, localize-1.2.0).

The request is unchanged, byte for byte: the compact-p7 prompt and wire schema. The contract keeps the p7 prompt's
semantics exactly and adds only literal checks the prompt already asks for.

1. **Event semantics are p7's, applied consistently.** For a not_occurred or not_asserted effect every event field
   stays empty: severity, event_dose, onset, event_duration, clinical_consequences and outcome. This is the canonical
   compact-wire-1.0.0 rule the prompt states. The local negdose and severity relaxations of p7t1 and p7t2 do NOT apply
   here. A reply that fills them fails at stage "contract", with the item paths.
2. **As-written fields must be literals.** dose, duration, event_dose, onset, event_duration, subject_label and
   scope_text must each occur somewhere in the input as a token-bounded literal. This is lexical provenance only; it does
   not show that the literal belongs to that item.
3. **Boundaries apply to exact matches too,** with the version-2 number boundaries:
   - "pain" inside "painting" does not count;
   - ".5" inside "10.5", "25" inside ".25", "5" inside "-5" or "2-5", "-5" inside "2-5" and ".5" inside "-.5" do not count either;
   - for dose and event_dose, "5 mg" inside "5 mg/kg" does not count (compound units stay whole).

   A quote must have exactly one bounded occurrence; a surface at least one.

Older specs keep their old rules, so every historical run, replay and export reproduces unchanged.

``compact-contract-2.1.0`` (specs compact-p7t4 and compact-p7c1, iteration-20260930-01a0f4c2) = 2.0.0 plus one
deterministic identity normalization, applied before the lexical checks and the compact conversion: a drug whose ref is
"unspecified" or "class" but whose surface, as a whole (case and punctuation aside), is a brand, generic or common
abbreviation of a study drug in the frozen alias table below (e.g. "MJ", "zep", "sema", "Oz") gets ref "name". The
surface is never changed, nothing else is touched, and no drug the text does not name is inferred. Each change is
reported with the localization repairs (kind "drug_ref").
"""
from __future__ import annotations

import re

from pydantic import model_validator

from .compact import EVENT_FIELDS, Wire, WireDrug, WireNoSideEffect, WirePurpose, WireSideEffect, items
from .localize import splits_token_v2
from .structured import ExtractionAnomaly

CONTRACT_2 = 'compact-contract-2.0.0'
CONTRACT_2_1 = 'compact-contract-2.1.0'
CONTRACTS = (CONTRACT_2, CONTRACT_2_1)
# Frozen alias table of contract 2.1.0: the study-drug aliases of the dual400/val200 scorer lexicon (score_d400.ALIASES and the
# val200 "ozempuc" spelling), without the ambiguous "tru". Whole-surface matches only.
NAME_ALIASES = {
    'semaglutide': ('ozempic', 'wegovy', 'rybelsus', 'rybellsus', 'semaglutide', 'sema', 'semi', 'wevovy', 'semiglutide', 'wagovy', 'wegovvy',
                    'wegoovy', 'weygovy', 'oz', 'ozem', 'ozempix', 'ozempuc'),
    'tirzepatide': ('mounjaro', 'zepbound', 'tirzepatide', 'tirz', 'tirzep', 'zep', 'mj', 'zb', 'zip bound', 'mounjar', 'tirzepetide', 'zepound',
                    'monjaro', 'mounjaru', 'moujaro', 'tirzepitide'),
    'liraglutide': ('saxenda', 'victoza', 'voctoza', 'liraglutide', 'lira'),
    'dulaglutide': ('trulicity', 'dulaglutide'),
    'retatrutide': ('retatrutide', 'reta'),
}
ALIAS_OF = {alias: molecule for molecule, aliases in NAME_ALIASES.items() for alias in aliases}


def _alias_key(surface):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', surface.lower()).split())


def normalize_identity(wire):
    """compact-contract-2.1.0: (wire with alias surfaces marked ref "name", [change])."""
    changes, drugs = [], []
    for i, d in enumerate(wire.drugs):
        molecule = ALIAS_OF.get(_alias_key(d.surface))
        if d.ref != 'name' and molecule:
            changes.append({'path': f'drugs[{i}].ref', 'kind': 'drug_ref', 'from': d.ref, 'to': 'name', 'surface': d.surface, 'alias_of': molecule})
            d = d.model_copy(update={'ref': 'name'})
        drugs.append(d)
    return (wire.model_copy(update={'drugs': drugs}) if changes else wire), changes
AS_WRITTEN = {'purposes': ('dose', 'duration', 'subject_label'),
              'side_effects': ('dose', 'duration', 'event_dose', 'onset', 'event_duration', 'subject_label'),
              'no_side_effects': ('dose', 'duration', 'subject_label', 'scope_text')}


class WireSideEffectLoose(WireSideEffect):
    """Analysis-only shape (path_repair): the p7 item without its occurrence rule, so the offending event fields of a
    contract failure can be located per path. Never used to accept a record."""

    @model_validator(mode='after')
    def _event_fields(self):
        if len(set(self.clinical_consequences)) != len(self.clinical_consequences):
            raise ValueError('clinical_consequences must not repeat')
        return self


class WireExtractionLoose(Wire):
    drugs: list[WireDrug]
    purposes: list[WirePurpose]
    side_effects: list[WireSideEffectLoose]
    no_side_effects: list[WireNoSideEffect]


def event_violations(wire):
    """[(path, field, value)] of event fields filled on not_occurred/not_asserted effects (the p7 rule)."""
    out = []
    for i, item in enumerate(wire.side_effects):
        if item.occurrence in ('not_occurred', 'not_asserted'):
            for name in EVENT_FIELDS:
                v = getattr(item, name)
                if v not in (None, []):
                    out.append((f'side_effects[{i}].{name}', name, v))
    return out


DOSE_FIELDS = ('dose', 'event_dose')  # compound units ("mg/kg", "mg/mL") stay whole for these


def literal_hits(value, fields, compound_units=False):
    """Every (field, start, end) where value occurs exactly and on version-2 token/number boundaries."""
    out = []
    if not value:
        return out
    for name, text in fields.items():
        start = text.find(value)
        while start != -1:
            if not splits_token_v2(text, start, start + len(value), compound_units):
                out.append((name, start, start + len(value)))
            start = text.find(value, start + 1)
    return out


def check_surfaces(wire, fields):
    missing = [f'drugs[{i}].surface' for i, d in enumerate(wire.drugs) if not literal_hits(d.surface, fields)]
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect')):
        missing += [f'{group}[{i}].{name}' for i, item in enumerate(getattr(wire, group)) if not literal_hits(getattr(item, name), fields)]
    if missing:
        raise ExtractionAnomaly('surface_not_found', 'not a token-bounded literal of the input: ' + ', '.join(missing[:5]),
                                stage='lexical', paths=missing)


def locate_evidence(wire, fields):
    spans, missing, ambiguous = {}, [], []
    for group, i, item in items(wire):
        found = []
        for q, quote in enumerate(item.quotes):
            hits = literal_hits(quote, fields)
            if not hits:
                missing.append(f'{group}[{i}].quotes[{q}]')
            elif len(hits) > 1:
                ambiguous.append(f'{group}[{i}].quotes[{q}]')
            else:
                found.append('{}:{}-{}'.format(*hits[0]))
        spans[group, i] = found
    if missing:
        raise ExtractionAnomaly('evidence_not_found', 'not a token-bounded literal of the input: ' + ', '.join(missing[:5]),
                                stage='lexical', paths=missing)
    if ambiguous:
        raise ExtractionAnomaly('evidence_ambiguous', 'several token-bounded matches: ' + ', '.join(ambiguous[:5]),
                                stage='lexical', paths=ambiguous)
    return spans


def check_fields(wire, fields):
    bad = []
    for group, i, item in items(wire):
        for name in AS_WRITTEN[group]:
            v = getattr(item, name, None)
            if v not in (None, '') and not literal_hits(v, fields, name in DOSE_FIELDS):
                bad.append(f'{group}[{i}].{name}')
    if bad:
        raise ExtractionAnomaly('field_not_found', 'as-written field is not a token-bounded literal of the input: ' + ', '.join(bad[:5]),
                                stage='lexical', paths=bad)


def check(wire, fields):
    """Lexical checks of contract 2.0.0 after the p7 canonical validation (which already enforces the event rule)."""
    check_surfaces(wire, fields)
    spans = locate_evidence(wire, fields)
    check_fields(wire, fields)
    return spans
