"""Reviewed mapping knowledge: alias rules (per role) and the project purpose-concept registry.

Rules name their MedDRA target by exact LLT name; the code is resolved locally against the frozen dictionary at run
time, so no code is ever written before the licensed file is present. Only certified rules can yield an accepted
mapping; draft rules and context-required rules yield candidates. Registry concepts live in the project namespace,
never pose as MedDRA codes, and record their MedDRA alignment status separately (not_checked / resource_blocked /
checked with a relation and a search log).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

from . import context as ctx
from .textnorm import fold_key

ROLES = ('side_effect', 'indication')
INTENTS = ('treat_condition', 'relieve_symptom', 'prevent', 'goal', 'other_explicit', 'unclear')
DIRECTIONS = ('decrease', 'maintain', 'increase', 'unspecified')
CONCEPT_KINDS = ('condition', 'symptom', 'goal', 'procedure', 'finding', 'other')
INTENT_MODES = ('surface_cue', 'evidence_default', 'context_required')
ALIGN_STATUS = ('not_checked', 'resource_blocked', 'checked')
ALIGN_RELATIONS = ('exact', 'target_broader', 'related', 'none')  # exact: migrate the mapping to MedDRA explicitly
ANCHORED_RELATIONS = ('exact', 'target_broader', 'related')
CONCEPT_ID = re.compile(r'^wlp:[a-z0-9_.]+$')


class KnowledgeError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SHA256_HEX = re.compile(r'^[0-9a-f]{64}$')


def _check_certification(owner, status, cert, base_dir):
    """A certified rule / reviewed concept names its review artifact; the file must exist (relative to the knowledge
    file) and hash to artifact_sha256, so every acceptance traces back to recorded review evidence."""
    if status not in ('draft', 'certified', 'reviewed'):
        raise KnowledgeError(f'{owner}: status {status!r}')
    if status in ('certified', 'reviewed'):
        if not isinstance(cert, dict) or not cert.get('reviewer') or not isinstance(cert.get('reviewed_records'), int) \
                or cert['reviewed_records'] < 1 or not SHA256_HEX.match(str(cert.get('artifact_sha256', ''))) \
                or not cert.get('artifact_path'):
            raise KnowledgeError(f'{owner}: {status} needs reviewer, reviewed_records >= 1, artifact_path and a 64-hex '
                                 f'artifact_sha256')
        artifact = (Path(base_dir) / cert['artifact_path']).resolve()
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != cert['artifact_sha256']:
            raise KnowledgeError(f'{owner}: review artifact {cert["artifact_path"]} missing or hash differs')


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    definition: str
    kind: str
    inherent_direction: str | None
    allowed_intents: tuple
    status: str
    meddra_alignment: dict

    @property
    def reviewed(self):
        return self.status == 'reviewed'


@dataclass
class Registry:
    registry_id: str
    version: str
    sha256: str
    concepts: dict = field(default_factory=dict)

    def get(self, concept_id):
        return self.concepts.get(concept_id)


def load_registry(path):
    data = json.loads(Path(path).read_text())
    if data.get('namespace') != 'wlp':
        raise KnowledgeError('registry namespace must be wlp')
    reg = Registry(registry_id=data['registry_id'], version=data['version'], sha256=_sha256(path))
    for item in data['concepts']:
        cid = item['concept_id']
        if not CONCEPT_ID.match(cid) or cid in reg.concepts:
            raise KnowledgeError(f'invalid or duplicate concept_id {cid!r}')
        if item['kind'] not in CONCEPT_KINDS:
            raise KnowledgeError(f'{cid}: kind {item["kind"]!r}')
        direction = item.get('inherent_direction')
        if direction is not None and direction not in DIRECTIONS[:3]:
            raise KnowledgeError(f'{cid}: inherent_direction {direction!r}')
        intents = tuple(item['allowed_intents'])
        if not intents or any(i not in INTENTS[:-1] for i in intents):
            raise KnowledgeError(f'{cid}: allowed_intents {intents}')
        if not item.get('definition') or not item.get('label'):
            raise KnowledgeError(f'{cid}: label and definition are required')
        _check_certification(cid, item['status'], item.get('review'), Path(path).parent)
        if item['status'] not in ('draft', 'reviewed'):
            raise KnowledgeError(f'{cid}: registry status is draft or reviewed')
        align = (item.get('external_alignment') or {}).get('meddra') or {}
        if align.get('status') not in ALIGN_STATUS:
            raise KnowledgeError(f'{cid}: meddra alignment status {align.get("status")!r}')
        if align['status'] == 'checked':
            if align.get('relation') not in ALIGN_RELATIONS or not align.get('search_log'):
                raise KnowledgeError(f'{cid}: a checked alignment needs a relation and its search log')
            if not SHA256_HEX.match(str(align.get('dictionary_snapshot_sha256', ''))):
                raise KnowledgeError(f'{cid}: a checked alignment names the dictionary snapshot it was checked on')
            if align['relation'] in ANCHORED_RELATIONS and not align.get('llt_name'):
                raise KnowledgeError(f'{cid}: an anchored alignment names the LLT it was checked against')
        elif align.get('relation') is not None:
            raise KnowledgeError(f'{cid}: relation only after the alignment is checked')
        reg.concepts[cid] = Concept(cid, item['label'], item['definition'], item['kind'], direction, intents, item['status'],
                                    align)
    return reg


@dataclass(frozen=True)
class Target:
    kind: str                 # meddra_llt | registry
    llt_name: str | None = None
    concept_id: str | None = None

    def key(self):
        return ('meddra_llt', self.llt_name) if self.kind == 'meddra_llt' else ('registry', self.concept_id)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    role: str
    keys: tuple
    targets: tuple
    status: str
    context_required: bool
    specificity: str
    concept_kind: str | None
    inherent_direction: str | None
    intent_policy: dict | None
    certification: dict | None

    @property
    def certified(self):
        return self.status == 'certified'


@dataclass
class RuleSet:
    rule_set_id: str
    role: str
    version: str
    sha256: str
    rules: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)

    def match(self, text):
        return self.index.get(fold_key(text))


def _target(raw, owner):
    if raw.get('kind') == 'meddra_llt' and raw.get('llt_name'):
        return Target('meddra_llt', llt_name=raw['llt_name'])
    if raw.get('kind') == 'registry' and CONCEPT_ID.match(raw.get('concept_id') or ''):
        return Target('registry', concept_id=raw['concept_id'])
    raise KnowledgeError(f'{owner}: invalid target {raw!r}')


def load_rules(path, role, registry=None):
    data = json.loads(Path(path).read_text())
    if role not in ROLES or data.get('role') != role:
        raise KnowledgeError(f'{path}: rule set role {data.get("role")!r} is not {role!r}')
    rs = RuleSet(rule_set_id=data['rule_set_id'], role=role, version=data['version'], sha256=_sha256(path))
    for item in data['rules']:
        rid = item['rule_id']
        if rid in rs.rules:
            raise KnowledgeError(f'duplicate rule_id {rid}')
        targets = tuple(_target(t, rid) for t in item['targets'])
        context_required = bool(item.get('context_required', False))
        if not targets or (not context_required and len(targets) != 1):
            raise KnowledgeError(f'{rid}: a context-free rule has exactly one target')
        if context_required and item['status'] == 'certified':
            raise KnowledgeError(f'{rid}: a context-required rule only proposes candidates and cannot be certified')
        _check_certification(rid, item['status'], item.get('certification'), Path(path).parent)
        if item['status'] == 'reviewed':
            raise KnowledgeError(f'{rid}: rule status is draft or certified')
        specificity = item.get('specificity', 'specific')
        if specificity not in ('specific', 'nonspecific'):
            raise KnowledgeError(f'{rid}: specificity {specificity!r}')
        kind = item.get('concept_kind')
        direction = item.get('inherent_direction')
        policy = item.get('intent_policy')
        if role == 'indication':
            for t in targets:
                if t.kind == 'meddra_llt' and kind not in CONCEPT_KINDS:
                    raise KnowledgeError(f'{rid}: a MedDRA purpose target needs concept_kind')
                if t.kind == 'registry' and registry is not None and registry.get(t.concept_id) is None:
                    raise KnowledgeError(f'{rid}: unknown registry concept {t.concept_id}')
            if not isinstance(policy, dict) or policy.get('mode') not in INTENT_MODES:
                raise KnowledgeError(f'{rid}: purpose rules need intent_policy.mode in {INTENT_MODES}')
            if policy['mode'] == 'surface_cue' and (policy.get('intent') not in INTENTS[:-1]):
                raise KnowledgeError(f'{rid}: surface_cue needs the intent the key itself expresses')
            if policy['mode'] == 'evidence_default' and policy.get('intent') not in INTENTS[:-1]:
                raise KnowledgeError(f'{rid}: evidence_default needs a default intent')
            if policy.get('direction') is not None and policy['direction'] not in DIRECTIONS:
                raise KnowledgeError(f'{rid}: direction {policy["direction"]!r}')
            if direction is not None and policy.get('direction') not in (None, direction):
                raise KnowledgeError(f'{rid}: policy direction contradicts the concept direction {direction}')
        elif policy is not None or kind is not None:
            raise KnowledgeError(f'{rid}: side-effect rules carry no intent or concept kind')
        keys = tuple(sorted({fold_key(k) for k in item['keys']}))
        if not keys or '' in keys:
            raise KnowledgeError(f'{rid}: empty key')
        rule = Rule(rid, role, keys, targets, item['status'], context_required, specificity, kind, direction, policy,
                    item.get('certification'))
        for key in keys:
            if key in rs.index:
                raise KnowledgeError(f'key {key!r} is claimed by {rs.index[key].rule_id} and {rid}; keep one rule per key')
            rs.index[key] = rule
        rs.rules[rid] = rule
    return rs


def registry_alignment(concept, registry, dictionary):
    """Run-time MedDRA alignment of one registry concept, reported next to (never instead of) its internal acceptance.

    resource_blocked: no dictionary in this run, so nothing about MedDRA (including absence) can be claimed.
    not_checked: a dictionary is present but the reviewed registry carries no check; a completed run does not anchor.
    stale: the recorded check was made on another dictionary snapshot.
    check_failed: the recorded LLT is not a single current LLT of this dictionary.
    checked: checked on this run's dictionary; anchored only for exact / target_broader / related (exact means the
    mapping itself must migrate to MedDRA)."""
    declared = concept.meddra_alignment or {}
    out = {'vocabulary': 'MedDRA', 'status': 'resource_blocked', 'relation': None, 'anchored': False, 'llt_name': None,
           'declared_status': declared.get('status', 'not_checked'), 'declared_relation': declared.get('relation'),
           'registry_sha256': registry.sha256, 'dictionary_snapshot_sha256': None}
    if dictionary is None:
        return out
    out['dictionary_snapshot_sha256'] = dictionary.snapshot_sha256
    if declared.get('status') != 'checked':
        return dict(out, status='not_checked')
    if declared.get('dictionary_snapshot_sha256') != dictionary.snapshot_sha256:
        return dict(out, status='stale')
    relation = declared['relation']
    if relation in ANCHORED_RELATIONS:
        hits = dictionary.exact(declared['llt_name'])
        if len(hits) != 1 or not hits[0].current:
            return dict(out, status='check_failed')
        return dict(out, status='checked', relation=relation, anchored=True, llt_name=declared['llt_name'])
    return dict(out, status='checked', relation=relation)


def resolve_target(target, dictionary, registry):
    """(status, payload) for one rule target. status: ok | resource_blocked | name_not_found | not_current |
    unknown_concept."""
    if target.kind == 'registry':
        concept = registry.get(target.concept_id) if registry is not None else None
        return ('ok', concept) if concept is not None else ('unknown_concept', None)
    if dictionary is None:
        return 'resource_blocked', None
    hits = dictionary.exact(target.llt_name)
    if not hits:
        return 'name_not_found', None
    if len(hits) > 1:
        return 'name_not_found', None
    llt = hits[0]
    return ('ok', llt) if llt.current else ('not_current', llt)


def exact_llt_conflicts(rule, dictionary):
    """Keys of a MedDRA rule that are themselves the exact name of a different current LLT (MTS:PTC: the LLT closest to
    the reported words is selected), e.g. key "nauseous" with an LLT "Nauseous" while the rule targets "Nausea"."""
    if dictionary is None or rule.targets[0].kind != 'meddra_llt' or rule.context_required:
        return []
    out = []
    for key in rule.keys:
        hits, collision = dictionary.folded(key)
        current = [h for h in hits if h.current and h.name != rule.targets[0].llt_name]
        if current and not collision:
            out.append((key, current[0].name))
    return out


def rule_problems(ruleset, dictionary, registry):
    """Rule targets that do not resolve against the frozen dictionary / registry, keys that are the exact name of a
    different current LLT, and certified keys that the context-required word list blocks (conflicts reported for review;
    nothing is resolved automatically)."""
    out = []
    for rule in ruleset.rules.values():
        for target in rule.targets:
            status, _ = resolve_target(target, dictionary, registry)
            if status != 'ok':
                out.append({'rule_id': rule.rule_id, 'target': target.llt_name or target.concept_id, 'status': status})
        for key, name in exact_llt_conflicts(rule, dictionary):
            out.append({'rule_id': rule.rule_id, 'target': rule.targets[0].llt_name, 'status': f'key_is_exact_llt:{key}->{name}'})
        if rule.certified and not rule.context_required:
            for key in rule.keys:
                if ctx.context_required(ruleset.role, key):
                    out.append({'rule_id': rule.rule_id, 'target': rule.targets[0].llt_name or rule.targets[0].concept_id,
                                'status': f'certified_key_context_required:{key}'})
    return out
