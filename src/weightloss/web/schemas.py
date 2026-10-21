"""HTTP contract of the website API. The frontend's TypeScript types are generated from these models
(apps/web/src/api/openapi.json -> schema.d.ts), so a field change here surfaces as a type error there.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel

Platform = Literal['reddit', 'webmd']
Target = Literal['side_effect', 'purpose']
Level = Literal['ingredient', 'brand']
Show = Literal['side_effect', 'purpose', 'both']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


# ---------------------------------------------------------------------------------------------------- overview
class PlatformCounts(BaseModel):
    platform: Platform
    records_processed: Optional[int] = Field(None, description='Records given to extraction')
    records_with_side_effect: int
    side_effect_statements: int
    records_with_purpose: int
    purpose_statements: int


class Versions(BaseModel):
    kg: Optional[str] = None
    kg_built: Optional[str] = None
    graph_loaded: Optional[str] = None
    meddra: Optional[str] = None
    extraction_model: Optional[str] = None
    extraction_prompt: Optional[str] = None
    agent: Optional[str] = None
    agent_model: Optional[str] = None


class BrandInfo(BaseModel):
    brand: str
    ingredient: str


class Soc(BaseModel):
    code: str
    name: str
    side_effect: int = Field(description='Largest per-platform record count with a side effect in this SOC')
    purpose: int


class Overview(BaseModel):
    platforms: list[PlatformCounts]
    faers_reports: int = Field(description='FAERS reports in the primary configuration')
    faers_status: str
    versions: Versions
    ingredients: list[str]
    brands: list[BrandInfo]
    age_groups: list[str]
    sexes: list[str]
    socs: list[Soc]
    notes: list[str]


# ---------------------------------------------------------------------------------------------------- graph explorer
class GraphQuery(Strict):
    platform: Platform = 'reddit'
    level: Level = 'brand'
    show: Show = 'both'
    drugs: Optional[list[str]] = Field(None, description='Ingredients or brands (by level) to keep; omitted: all')
    age_groups: Optional[list[str]] = None
    sexes: Optional[list[str]] = None
    soc: Optional[str] = Field(None, description='MedDRA SOC code: keep only PTs under it')
    top_n: int = Field(4, ge=1, le=10, description='PTs kept per drug and role')


class GraphNode(BaseModel):
    id: str
    kind: Literal['drug', 'side_effect', 'purpose']
    label: str
    code: Optional[str] = None
    soc: Optional[str] = None
    ingredient: Optional[str] = None
    records: dict[str, int] = Field(description='Records per role: for a drug, records with at least one statement of that '
                                                'role; for a PT, records reporting it for any shown drug')
    denominator: Optional[int] = None


class GraphLink(BaseModel):
    source: str
    target: str
    kind: Target
    records: int
    denominator: int = Field(description="Records of the drug with at least one statement of this role")
    percent: Optional[float] = None
    sparse: bool = Field(description='Denominator below 10: too few records to compare')
    small: bool = Field(False, description='Denominator below 30: a small group, so the share is unstable')


class GraphSummary(BaseModel):
    target: Target
    universe_records: Optional[int] = None
    records_before_age_sex_filter: Optional[int] = None
    excluded_missing_age: Optional[int] = None
    excluded_missing_sex: Optional[int] = None
    universe_records_with_uncoded_mention: Optional[int] = None
    universe_records_with_community_ingredient: Optional[int] = None


class GraphView(BaseModel):
    platform: Platform
    level: Level
    show: Show
    top_n: int
    nodes: list[GraphNode]
    links: list[GraphLink]
    summary: list[GraphSummary]
    rows_not_shown: dict[str, int]
    notes: list[str]


# ---------------------------------------------------------------------------------------------------- patterns
class PatternRow(BaseModel):
    platform: Platform
    label: str
    code: Optional[str] = None
    records: int
    denominator: int
    percent: Optional[float] = None
    sparse: bool
    small: bool = False


class PatternSection(BaseModel):
    key: str
    title: str
    unit: str
    definition: Optional[str] = None
    rows: list[PatternRow]


class PopulationCell(BaseModel):
    platform: Platform
    age_group: str
    sex: str
    records: int
    denominator: int
    percent: Optional[float] = None


class PopulationSummary(BaseModel):
    platform: Platform
    records: int
    age_and_sex_stated: int
    neither_stated: int


class Patterns(BaseModel):
    sections: list[PatternSection]
    population: list[PopulationCell]
    population_summary: list[PopulationSummary]
    notes: list[str]


# ---------------------------------------------------------------------------------------------------- FAERS
class FaersQuery(Strict):
    scope: str = Field('all', pattern=r'^(all|ingredient:[a-z]+|brand:[A-Za-z]+)$')
    rank_by: Literal['faers', 'reddit', 'webmd'] = 'faers'
    top_n: int = Field(12, ge=1, le=30)
    sex: Optional[Literal['female', 'male']] = None
    age_group: Optional[str] = None


class FaersRow(BaseModel):
    concept: str
    code: Optional[str] = Field(None, description='MedDRA PT code (ranked rows only)')
    faers_reports: Optional[int] = None
    faers_denominator: Optional[int] = None
    faers_percent: Optional[float] = None
    reddit_records: Optional[int] = None
    reddit_denominator: Optional[int] = None
    reddit_percent: Optional[float] = None
    webmd_records: Optional[int] = None
    webmd_denominator: Optional[int] = None
    webmd_percent: Optional[float] = None
    reddit_small: Optional[bool] = Field(None, description='Reddit denominator below 30')
    webmd_small: Optional[bool] = Field(None, description='WebMD denominator below 30')


class FaersView(BaseModel):
    scope: str
    rank_by: str
    sex: Optional[str] = None
    age_group: Optional[str] = None
    status: Optional[str] = None
    window: Optional[str] = None
    summary: dict[str, Any]
    rows: list[FaersRow]
    notes: list[str]


# ---------------------------------------------------------------------------------------------------- evidence
class EvidenceQuery(Strict):
    target: Target
    platform: Platform
    level: Optional[Level] = None
    drug: Optional[str] = None
    code: Optional[str] = Field(None, description='PT code')
    limit: int = Field(6, ge=1, le=10)
    age_groups: Optional[list[str]] = None
    sexes: Optional[list[str]] = None
    soc: Optional[str] = Field(None, description='MedDRA SOC code, used when no PT code is given')


class Statement(BaseModel):
    edge_id: str
    platform: Optional[str] = None
    record_id: Optional[str] = None
    date: Optional[str] = None
    ingredient: Optional[str] = None
    brand: Optional[str] = None
    ingredient_basis: Optional[str] = None
    pt: Optional[str] = None
    pt_code: Optional[str] = None
    soc: Optional[str] = None
    wording: Optional[str] = None
    writer_age_group: Optional[str] = None
    writer_sex: Optional[str] = None
    dose: Optional[str] = None
    duration: Optional[str] = None
    intensity: Optional[str] = None
    impact: Optional[str] = None
    action_taken: list[str] = []
    evidence: list[str] = []
    evidence_withheld: Optional[str] = Field(None, description='Why the source sentences are not shown (writers aged 13-18)')
    writer_evidence: Optional[list[str]] = None


class EvidenceView(BaseModel):
    target: Target
    platform: Platform
    drug: Optional[str] = None
    code: Optional[str] = None
    matching_records: int
    statements: list[Statement]
    notes: list[str]


# ---------------------------------------------------------------------------------------------------- agent
class AskRequest(Strict):
    question: str = Field(min_length=1, max_length=500)


class AskStatus(BaseModel):
    available: bool
    reason: Optional[str] = None
    agent_version: str
    spent_usd: float
    budget_usd: float


class Citation(BaseModel):
    id: str
    call: str
    row: int
    field: str
    value: str
    tool: Optional[str] = None
    arguments: dict[str, Any] = {}
    source: Optional[dict[str, Any]] = Field(None, description='The cited tool-output row')


class Quote(BaseModel):
    edge_id: str
    text: str
    statement: Optional[Statement] = None


class ToolCall(BaseModel):
    call: str
    tool: str
    seconds: Optional[float] = None
    arguments: dict[str, Any] = {}
    error: Optional[str] = None
    rows: int


class AskResult(BaseModel):
    run_id: Optional[str] = None
    question: Optional[str] = None
    verified: bool
    answer_type: Optional[Literal['answered', 'limited', 'declined']] = None
    answer: str
    limitations: list[str]
    citations: list[Citation]
    quotes: list[Quote]
    tool_calls: list[ToolCall]
    problems: list[str]
    repair_used: bool
    repairs: int = Field(0, description='Repair rounds after failed verification (0-2)')
    stop_reason: Optional[str] = None
    cost_usd: float
    seconds: Optional[float] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    agent_version: Optional[str] = None


class StartEvent(BaseModel):
    type: Literal['start']
    question: str


class QueuedEvent(BaseModel):
    type: Literal['queued']
    ahead: int = Field(description='Questions from other visitors that start before this one')


class ModelEvent(BaseModel):
    type: Literal['model']
    round: int
    status: Literal['running', 'done']
    tool_calls: Optional[int] = None


class ToolEvent(BaseModel):
    type: Literal['tool']
    step: str
    tool: str
    status: Literal['running', 'done', 'error']
    arguments: dict[str, Any] = {}
    seconds: Optional[float] = None
    rows: Optional[int] = None
    detail: Optional[str] = None


class ResultEvent(BaseModel):
    type: Literal['result']
    result: AskResult


class ErrorEvent(BaseModel):
    type: Literal['error']
    message: str


class AskEvent(RootModel[Annotated[Union[StartEvent, QueuedEvent, ModelEvent, ToolEvent, ResultEvent, ErrorEvent],
                                     Field(discriminator='type')]]):
    """One server-sent event of POST /api/ask (each `data:` line is one of these, as JSON)."""


class ErrorResponse(BaseModel):
    detail: str
