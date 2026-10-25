"""Typed output records (Pydantic v2). Every written line is validated against these models before it is written and
again by the validator. Model wire schemas for the optional proposer live in llm.py."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 'kg-semantic-map-0.1.2'
Role = Literal['side_effect', 'indication']
Method = Literal['certified_rule', 'draft_rule', 'context_rule', 'exact_dictionary', 'fuzzy_dictionary', 'llm_proposal']
Resolution = Literal['ok', 'resource_blocked', 'name_not_found', 'not_current', 'unknown_concept']
Intent = Literal['treat_condition', 'relieve_symptom', 'prevent', 'goal', 'other_explicit', 'unclear']
Direction = Literal['decrease', 'maintain', 'increase', 'unspecified']
AlignRelation = Literal['exact', 'target_broader', 'related', 'none']
SHA256 = r'^[0-9a-f]{64}$'


class Closed(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ExternalAlignment(Closed):
    """MedDRA status of a registry concept in this run (knowledge.registry_alignment); independent of acceptance."""
    vocabulary: Literal['MedDRA']
    status: Literal['resource_blocked', 'not_checked', 'stale', 'check_failed', 'checked']
    relation: Optional[AlignRelation] = None
    anchored: bool
    llt_name: Optional[str] = None
    declared_status: Literal['not_checked', 'resource_blocked', 'checked']
    declared_relation: Optional[AlignRelation] = None
    registry_sha256: str = Field(pattern=SHA256)
    dictionary_snapshot_sha256: Optional[str] = Field(default=None, pattern=SHA256)

    @model_validator(mode='after')
    def _states(self):
        if (self.status == 'resource_blocked') != (self.dictionary_snapshot_sha256 is None):
            raise ValueError('resource_blocked exactly when the run has no dictionary snapshot')
        if self.relation is not None and self.status != 'checked':
            raise ValueError('a relation is only reported for an alignment checked on this dictionary')
        if self.anchored != (self.status == 'checked' and self.relation in ('exact', 'target_broader', 'related')):
            raise ValueError('anchored only for a checked exact / target_broader / related alignment')
        if self.anchored != (self.llt_name is not None):
            raise ValueError('an anchored alignment names its LLT, and only then')
        return self


class TargetOut(Closed):
    kind: Literal['meddra_llt', 'registry']
    llt_code: Optional[str] = Field(default=None, pattern=r'^[0-9]{8}$')
    llt_name: Optional[str] = None
    pt_code: Optional[str] = Field(default=None, pattern=r'^[0-9]{8}$')
    pt_name: Optional[str] = None
    concept_id: Optional[str] = Field(default=None, pattern=r'^wlp:[a-z0-9_.]+$')
    concept_label: Optional[str] = None
    external_alignment: Optional[ExternalAlignment] = None

    @model_validator(mode='after')
    def _shape(self):
        if self.kind == 'registry':
            if not self.concept_id or any(v is not None for v in (self.llt_code, self.llt_name, self.pt_code, self.pt_name)):
                raise ValueError('a registry target carries only concept_id / concept_label / external_alignment')
            if (self.concept_label is None) != (self.external_alignment is None):
                raise ValueError('a known registry concept reports its external alignment (unknown concepts carry neither)')
        else:
            if not self.llt_name or self.concept_id is not None or self.external_alignment is not None:
                raise ValueError('a MedDRA target carries llt_name (codes only once resolved)')
            if (self.llt_code is None) != (self.pt_code is None):
                raise ValueError('llt_code and pt_code are resolved together')
        return self


class Candidate(Closed):
    method: Method
    rule_id: Optional[str] = None
    resolution: Resolution
    target: TargetOut
    score: Optional[float] = None
    flags: tuple[str, ...] = ()


class Residual(Closed):
    kind: Literal['determiner', 'quantity', 'severity', 'frequency', 'laterality', 'temporal', 'unconsumed']
    text: str
    start: int
    end: int


class Component(Closed):
    component_id: str = Field(pattern=r'^c[1-9][0-9]*$')
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    reading: Literal['whole', 'conjunctive', 'ambiguous', 'descriptive']
    core_text: str
    residuals: tuple[Residual, ...] = ()
    semantic_status: Literal['accepted', 'candidate', 'unmapped', 'not_target', 'blocked']
    coding_status: Literal['coded', 'resource_blocked', 'failed', 'not_applicable', 'none']
    target: Optional[TargetOut] = None
    method: Optional[Method] = None
    rule_id: Optional[str] = None
    specificity: Optional[Literal['specific', 'nonspecific']] = None
    intent: Optional[Intent] = None
    goal_direction: Optional[Direction] = None
    intent_basis: Optional[str] = None
    reasons: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    retrieval_failure: Literal['none', 'recall_failure', 'disambiguation_failure', 'resource_blocked']
    review_basis: Literal['certified_rule', 'agent_review', 'human_review', 'none']

    @model_validator(mode='after')
    def _consistent(self):
        if self.semantic_status == 'accepted':
            if self.target is None or self.method not in ('certified_rule',) and self.review_basis not in ('agent_review', 'human_review'):
                raise ValueError('accepted needs a target from a certified rule or a recorded review')
        elif self.target is not None:
            raise ValueError('only an accepted component carries a chosen target')
        if self.coding_status == 'coded' and (self.target is None or self.target.kind != 'meddra_llt' or self.target.llt_code is None):
            raise ValueError('coded needs a resolved MedDRA target')
        if self.target is not None and self.target.kind == 'meddra_llt' and self.target.llt_code is not None \
                and self.coding_status != 'coded':
            raise ValueError('a resolved MedDRA code is only written when coding_status is coded')
        if self.goal_direction is not None and self.intent != 'goal':
            raise ValueError('goal_direction only with intent goal')
        return self


class NodeDecision(Closed):
    mapping_schema_version: Literal['kg-semantic-map-0.1.2'] = SCHEMA_VERSION
    run_id: str
    kg_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    kg_line: int = Field(ge=1)
    record_id: str
    node_id: str
    item_path: str
    role: Role
    surface: str
    carrier_edge_id: str
    carrier_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    drug_node_id: str
    drug_category: Literal['catalog_brand', 'catalog_generic', 'name_not_in_catalog', 'class', 'unspecified', 'held']
    kg_review_flags: tuple[str, ...]
    source_support: Literal['own_evidence_literal', 'own_evidence_casefold', 'record_literal', 'not_literal']
    components: tuple[Component, ...] = Field(min_length=1)
    coverage: Literal['complete', 'partial', 'ambiguous', 'unresolved', 'not_target']
    coding: Literal['coded', 'partially_coded', 'resource_blocked', 'failed', 'not_applicable', 'none']
    intent_coverage: Literal['complete', 'partial', 'intent_unclear', 'none', 'not_applicable']
    dictionary_class: Optional[Literal['formal', 'unpinned', 'synthetic_test']] = None
    anomalies: tuple[str, ...] = ()

    @model_validator(mode='after')
    def _ids(self):
        ids = [c.component_id for c in self.components]
        if ids != [f'c{i}' for i in range(1, len(ids) + 1)]:
            raise ValueError('component ids are c1..cn in surface order')
        for c in self.components:
            if not (0 <= c.start < c.end <= len(self.surface)) or self.surface[c.start:c.end] != c.text:
                raise ValueError(f'{c.component_id}: text is not the surface slice at its offsets')
        if any(c.coding_status == 'coded' for c in self.components) and self.dictionary_class is None:
            raise ValueError('coded output names its dictionary class')
        return self
