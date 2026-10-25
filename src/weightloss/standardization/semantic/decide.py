"""Per-node decision: components, semantic status, coding status and (for purposes) intent.

Acceptance needs all of: a certified context-free rule, a target that resolves (MedDRA name in the frozen file, or a
reviewed registry concept), deterministic source support in the node's own evidence, no context-required key, no
ambiguous coordination, compatible residuals and, for purposes, no intent conflict. Everything else stays a candidate
(or unmapped with a reason when the dictionary was searched and nothing was found). Without a dictionary nothing is
reported as unmapped: unresolved components are `blocked` and the coding status is resource_blocked.
"""
from __future__ import annotations

from . import context as ctx
from .models import Candidate, Component, NodeDecision, Residual, TargetOut
from .retrieval import retrieve
from .source import drug_category, scope_names
from .textnorm import fold_key

# relation-scope findings that make the KG drug -> purpose binding itself doubtful (candidate + upstream review);
# coverage / non-medication relations only leave the intent unclear
SCOPE_BLOCKING = frozenset({'other_drug', 'ambiguous_drug'})


def _residuals(items, offset):
    return tuple(Residual(kind=r['kind'], text=r['text'], start=r['start'] + offset, end=r['end'] + offset) for r in items)


def _concept_meta(rule, target, registry):
    """(concept_kind, inherent_direction) of a purpose target."""
    if target.kind == 'registry':
        concept = registry.get(target.concept_id) if registry is not None else None
        return (concept.kind, concept.inherent_direction) if concept is not None else (None, None)
    return rule.concept_kind, rule.inherent_direction


class Decider:
    def __init__(self, *, run_id, kg_sha256, rules, dictionary, registry, catalog):
        self.run_id, self.kg_sha256 = run_id, kg_sha256
        self.rules, self.dictionary, self.registry, self.catalog = rules, dictionary, registry, catalog

    # -- one text span -------------------------------------------------------------------------------------------
    def _retrieve(self, role, text):
        return retrieve(text, rules=self.rules[role], dictionary=self.dictionary, registry=self.registry)

    def _evaluate(self, binding, role, surface, start, end, reading, support, proposal_terms=(), names=None):
        text = surface[start:end]
        found = self._retrieve(role, text)
        core, residuals = text, ()
        if found.rule is None:
            stripped, items = ctx.strip_qualifiers(text)
            if stripped and stripped != text:
                second = self._retrieve(role, stripped)
                if second.rule is not None or (second.candidates and not found.candidates):
                    found, core, residuals = second, stripped, _residuals(items, start)
        candidates = list(found.candidates)
        for term in proposal_terms:
            extra = self._retrieve(role, term)
            seen = {(c.target.kind, c.target.llt_code or c.target.llt_name, c.target.concept_id) for c in candidates}
            for c in extra.candidates:
                key = (c.target.kind, c.target.llt_code or c.target.llt_name, c.target.concept_id)
                if key not in seen:
                    candidates.append(Candidate(method='llm_proposal', rule_id=c.rule_id, resolution=c.resolution,
                                                target=c.target, score=c.score, flags=c.flags + ('proposed_term',)))
                    seen.add(key)
        reasons, intent, direction, basis = [], None, None, None
        rule = found.rule
        quotes = binding.quotes
        dictionary_blocked = self.dictionary is None

        def done(semantic, coding, failure, target=None, method=None, rule_id=None, specificity=None, review='none'):
            return dict(text=text, start=start, end=end, reading=reading, core_text=core, residuals=residuals,
                        semantic_status=semantic, coding_status=coding, target=target, method=method, rule_id=rule_id,
                        specificity=specificity, intent=intent, goal_direction=direction, intent_basis=basis,
                        reasons=tuple(sorted(set(reasons))), candidates=tuple(candidates), retrieval_failure=failure,
                        review_basis=review)

        # a certified rule whose own key (a reviewed synonym) is literal in the evidence supports the component even when
        # the extracted wording is not ("nausea" extracted from "I get nauseous")
        variant = None
        if rule is not None and rule.certified and not ctx.deterministic_support(support):
            joined = ' '.join(quotes)
            variant = next((k for k in sorted(rule.keys, key=len, reverse=True) if ctx.mentions(k, joined)), None)
            if variant is not None:
                reasons.append(f'info:rule_variant_support:{variant}')
        anchor_text = variant or core
        # intent (purposes) for the rule's first target, whatever the final status
        if role == 'indication' and rule is not None:
            kind, inherent = _concept_meta(rule, rule.targets[0], self.registry)
            intent, direction, basis, conflict = ctx.decide_intent(kind, rule.intent_policy, anchor_text, quotes, inherent,
                                                                   names=names)
            if conflict:
                reasons.append(f'intent_conflict:{basis}')
            else:
                if basis.startswith('relation_out_of_scope:'):
                    why = basis.split(':', 1)[1]
                    if why in SCOPE_BLOCKING:  # the source gives this purpose relation to another / no unique drug
                        reasons += [f'relation_scope:{why}', 'upstream_review']
                    else:
                        reasons.append(f'info:relation_scope:{why}')
                if kind == 'goal' and intent != 'goal':
                    reasons.append('goal_relation_not_confirmed')  # a goal concept needs a stated goal, not a result
        if rule is None:
            if dictionary_blocked:
                reasons.append('dictionary_not_available')
                return done('blocked', 'resource_blocked', 'resource_blocked')
            if not candidates:
                reasons.append('no_candidate')
                return done('unmapped', 'none', 'recall_failure')
            reasons.append('no_certified_rule')
            if any('case_collision' in c.flags for c in candidates):
                reasons.append('case_collision')
            return done('candidate', 'none', 'disambiguation_failure')
        # a rule matched
        if rule.context_required:
            reasons.append('context_required_rule')
        elif not rule.certified:
            reasons.append('draft_rule')
        rule_cand = found.rule_candidates[0] if found.rule_candidates else None
        if rule_cand is not None and rule_cand.resolution not in ('ok', 'resource_blocked'):
            reasons.append(f'rule_target_{rule_cand.resolution}')
        if 'registry_concept_draft' in (rule_cand.flags if rule_cand else ()):
            reasons.append('registry_concept_draft')
        if ctx.context_required(role, core) or ctx.context_required(role, text):
            reasons.append('context_required_term')
        if reading == 'ambiguous':
            reasons.append('ambiguous_coordination')
        if reading == 'descriptive':
            reasons.append('descriptive_phrase')
        if not ctx.deterministic_support(support) and variant is None:
            reasons.append(f'source_support:{support}')
        if any(r.kind not in ctx.COMPATIBLE_RESIDUALS for r in residuals):
            reasons.append('incompatible_residual')
        if role == 'side_effect' and ctx.tolerability(text):
            reasons.append('tolerability_statement')
        if self.dictionary is not None and rule_cand is not None and rule_cand.target.kind == 'meddra_llt':
            # a folded key that is a case-collision bucket (e.g. Rh antigen E/e) is only accepted on an exact-case match
            names = {rule_cand.target.llt_name, core}
            if any(fold_key(n) in self.dictionary.collisions for n in names) and core != rule_cand.target.llt_name:
                reasons.append('case_collision')
            hits, collision = self.dictionary.folded(core)
            if not collision and any(h.current and h.name != rule_cand.target.llt_name for h in hits):
                reasons.append('exact_llt_differs')  # the reported words are themselves a different current LLT
        if rule_cand is not None and rule_cand.target.kind == 'registry' and self.registry is not None:
            concept = self.registry.get(rule_cand.target.concept_id)
            if concept is not None and concept.meddra_alignment.get('relation') == 'exact':
                reasons.append('registry_exact_alignment:migrate_to_meddra')
        if role == 'indication' and rule_cand is not None and rule_cand.target.kind == 'registry' and intent not in (None, 'unclear'):
            concept = self.registry.get(rule_cand.target.concept_id) if self.registry is not None else None
            if concept is not None and intent not in concept.allowed_intents:
                reasons.append('intent_not_allowed_for_concept')
        if reading in ('conjunctive', 'ambiguous') and not ctx.mentions(text, ' '.join(quotes)):
            reasons.append('component_not_in_evidence')
        specificity = rule.specificity
        blocking = [r for r in reasons if not r.startswith('info:')]
        if not blocking and rule.certified and not rule.context_required and rule_cand is not None:
            target = rule_cand.target
            if rule_cand.resolution == 'ok':
                coding = 'coded' if target.kind == 'meddra_llt' else 'not_applicable'
            else:  # resource_blocked: semantic decision stands, the code waits for the dictionary
                coding = 'resource_blocked'
            return done('accepted', coding, 'none', target=target, method='certified_rule', rule_id=rule.rule_id,
                        specificity=specificity, review='certified_rule')
        coding = 'resource_blocked' if dictionary_blocked else ('failed' if any(r.startswith('rule_target_') for r in reasons) else 'none')
        return done('candidate', coding, 'disambiguation_failure', rule_id=rule.rule_id, specificity=specificity)

    # -- one node --------------------------------------------------------------------------------------------------
    def decide(self, binding, record_fields=None, proposal=None):
        role, surface = binding.role, binding.node['surface']
        record_text = '\n'.join(v for v in (record_fields or {}).values() if isinstance(v, str)) or None
        support = ctx.source_support(surface, binding.quotes, record_text)
        terms = tuple(t for t in (proposal or {}).get('terms', ()) if t)
        names = scope_names(binding, self.catalog) if role == 'indication' else None
        comps = []
        if role == 'indication' and (ctx.dose_phase(surface) or ctx.dose_use(surface, binding.quotes)):
            comps.append(dict(text=surface, start=0, end=len(surface), reading='whole', core_text=surface, residuals=(),
                              semantic_status='not_target', coding_status='none', target=None, method=None, rule_id=None,
                              specificity=None, intent=None, goal_direction=None, intent_basis=None,
                              reasons=('not_a_purpose:dose_phase', 'upstream_review'), candidates=(),
                              retrieval_failure='none', review_basis='none'))
        elif role == 'side_effect' and ctx.generic_ae(surface) and self.rules[role].match(surface) is None:
            comps.append(dict(text=surface, start=0, end=len(surface), reading='whole', core_text=surface, residuals=(),
                              semantic_status='not_target', coding_status='none', target=None, method=None, rule_id=None,
                              specificity='nonspecific', intent=None, goal_direction=None, intent_basis=None,
                              reasons=('nonspecific_reference',), candidates=(), retrieval_failure='none', review_basis='none'))
        else:
            whole = self._evaluate(binding, role, surface, 0, len(surface), 'whole', support, terms, names)
            reading, parts = ctx.split_components(surface)
            # a whole-term rule or an exact dictionary term (a combination term such as "nausea and vomiting") wins
            # over splitting, as MTS:PTC prefers an existing combination term
            whole_term = whole['rule_id'] is not None or any(c.method == 'exact_dictionary' for c in whole['candidates'])
            if whole['semantic_status'] == 'accepted' or whole_term or reading in ('whole', 'descriptive'):
                if reading == 'descriptive' and whole['semantic_status'] != 'accepted':
                    whole = self._evaluate(binding, role, surface, 0, len(surface), 'descriptive', support, terms, names)
                comps.append(whole)
            else:
                for s, e in parts:
                    comps.append(self._evaluate(binding, role, surface, s, e, reading, support, names=names))
        components = tuple(Component(component_id=f'c{i}', **c) for i, c in enumerate(comps, 1))
        return NodeDecision(run_id=self.run_id, kg_sha256=self.kg_sha256, kg_line=binding.kg_line, record_id=binding.record_id,
                            node_id=binding.node['id'], item_path=binding.node['item_path'], role=role, surface=surface,
                            carrier_edge_id=binding.edge['id'], carrier_sha256=binding.carrier_sha256,
                            drug_node_id=binding.drug['id'], drug_category=drug_category(binding.drug, self.catalog),
                            kg_review_flags=tuple(binding.review_flags), source_support=support, components=components,
                            coverage=coverage(components), coding=coding(components),
                            intent_coverage=intent_coverage(role, components),
                            dictionary_class=self.dictionary.dictionary_class if self.dictionary is not None else None)


def coverage(components):
    states = [c.semantic_status for c in components]
    if all(s == 'not_target' for s in states):
        return 'not_target'
    if any(c.reading == 'ambiguous' for c in components):
        return 'ambiguous'
    if all(s == 'accepted' for s in states):
        return 'complete'
    if any(s == 'accepted' for s in states):
        return 'partial'
    return 'unresolved'


def intent_coverage(role, components):
    """Purposes only: concept + intent completeness, reported apart from concept coverage."""
    if role != 'indication':
        return 'not_applicable'
    accepted = [c for c in components if c.semantic_status == 'accepted']
    if not accepted:
        return 'none'
    clear = [c for c in accepted if c.intent not in (None, 'unclear')]
    if len(clear) == len(accepted) and all(c.semantic_status == 'accepted' for c in components if c.semantic_status != 'not_target'):
        return 'complete'
    return 'partial' if clear else 'intent_unclear'


def coding(components):
    accepted = [c for c in components if c.semantic_status == 'accepted']
    if any(c.coding_status == 'resource_blocked' for c in components):
        return 'resource_blocked'
    if any(c.coding_status == 'failed' for c in components):
        return 'failed'
    if not accepted:
        return 'none'
    states = {c.coding_status for c in accepted}
    if states == {'coded'}:
        return 'coded'
    if states == {'not_applicable'}:
        return 'not_applicable'
    return 'partially_coded'
