"""Candidate retrieval for one text: alias rule -> exact (folded) dictionary name -> fuzzy dictionary recall.

Retrieval only collects and orders candidates; it never accepts. Without a dictionary only rule candidates exist and
the result says so (dictionary_blocked), so the caller reports resource_blocked instead of an empty recall.
"""
from __future__ import annotations

from dataclasses import dataclass

from .knowledge import registry_alignment, resolve_target
from .models import Candidate, ExternalAlignment, TargetOut
from .textnorm import fold_key


@dataclass(frozen=True)
class Retrieval:
    text: str
    key: str
    rule: object
    candidates: tuple
    dictionary_blocked: bool

    @property
    def rule_candidates(self):
        return tuple(c for c in self.candidates if c.method in ('certified_rule', 'draft_rule', 'context_rule'))


def meddra_out(llt, dictionary):
    pt = dictionary.pt_of(llt)
    return TargetOut(kind='meddra_llt', llt_code=llt.code, llt_name=llt.name, pt_code=pt.code, pt_name=pt.name)


def target_out(target, status, payload, dictionary, registry=None):
    if target.kind == 'registry':
        if payload is None:
            return TargetOut(kind='registry', concept_id=target.concept_id)
        return TargetOut(kind='registry', concept_id=target.concept_id, concept_label=payload.label,
                         external_alignment=ExternalAlignment(**registry_alignment(payload, registry, dictionary)))
    if status in ('ok', 'not_current') and payload is not None:
        return meddra_out(payload, dictionary)
    return TargetOut(kind='meddra_llt', llt_name=target.llt_name)


def retrieve(text, *, rules, dictionary, registry, fuzzy_limit=5):
    key = fold_key(text)
    out = []
    rule = rules.match(text) if rules is not None else None
    if rule is not None:
        method = 'context_rule' if rule.context_required else ('certified_rule' if rule.certified else 'draft_rule')
        for target in rule.targets:
            status, payload = resolve_target(target, dictionary, registry)
            flags = []
            if target.kind == 'registry' and payload is not None and not payload.reviewed:
                flags.append('registry_concept_draft')
            if rule.specificity == 'nonspecific':
                flags.append('nonspecific')
            out.append(Candidate(method=method, rule_id=rule.rule_id, resolution=status,
                                 target=target_out(target, status, payload, dictionary, registry), flags=tuple(flags)))
    if dictionary is not None and key:
        hits, collision = dictionary.folded(text)
        seen = {c.target.llt_code for c in out if c.target.llt_code}
        for llt in hits:
            flags = tuple(f for f, on in (('noncurrent', not llt.current), ('case_collision', collision)) if on)
            if llt.code not in seen:
                out.append(Candidate(method='exact_dictionary', resolution='ok' if llt.current else 'not_current',
                                     target=meddra_out(llt, dictionary), flags=flags))
                seen.add(llt.code)
        for llt, score in dictionary.fuzzy(text, limit=fuzzy_limit):
            if llt.code not in seen:
                out.append(Candidate(method='fuzzy_dictionary', resolution='ok', target=meddra_out(llt, dictionary), score=score))
                seen.add(llt.code)
    return Retrieval(text=text, key=key, rule=rule, candidates=tuple(out), dictionary_blocked=dictionary is None)
