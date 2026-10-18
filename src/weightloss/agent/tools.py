"""Graph tools of the agent: deterministic, parameterized Cypher over the local Neo4j knowledge graph. No LLM.

The graph (weightloss.graph.neo4j_db) is the Section 3.4 graph plus the MedDRA 29.1 primary path and the validated
FAERS reports. Four tools:
- link_terms      entity linking: question words -> PT and MedDRA group nodes, through full-text indexes over PT
                  names, MedDRA group names and the verbatim wording on the statement edges;
- query_graph     graph pattern query: match statements by drug, population, required co-statements of the same drug
                  in the same record and target concepts (PT or HLT/HLGT/SOC), and count records with denominators;
- get_evidence    provenance: example statements with record, drug, PT path and the cited source sentences;
- compare_faers   the same concepts and drugs in the validated FAERS reports, next to the patient records under the
                  RQ2 comparison rules (aligned window, clinical-event PTs only, validated primary configuration).

Counting rules (project document Section 3.6): the unit is a record (Reddit post or comment, WebMD review); a PT, or a
MedDRA group, counts once per record; the denominator of a row is the number of records with at least one target edge
in the filtered universe and the row's population group; Reddit and WebMD are reported separately and never pooled;
brand values are only brands named in the text or the WebMD profile; named_drugs_only drops community-assigned
ingredients; a missing age group or sex is 'not stated'. A required co-statement ("who take it for X") must be on the
same drug in the same record. Group levels (HLT, HLGT, SOC) follow each PT's MedDRA primary path, so every PT sits in
exactly one group per level.

Cypher is assembled only from fixed fragments; every value from the model is passed as a query parameter.
Every output is a JSON-ready dict whose citable rows carry a `row` number unique within the output.
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import re
import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field

from weightloss.graph import neo4j_db
from weightloss.graph.build import AGE_GROUPS, BRAND_INGREDIENT, UNSPECIFIED

PLATFORMS = ('reddit', 'webmd')
INGREDIENTS = ('semaglutide', 'tirzepatide', 'liraglutide', 'dulaglutide')
BRANDS = tuple(BRAND_INGREDIENT)
SEXES = ('female', 'male', 'other')
INTENSITIES = ('mild', 'moderate', 'severe')
IMPACTS = ('daily_life', 'medical_visit', 'emergency_visit', 'hospitalization')
ACTIONS = ('dose_reduced', 'escalation_paused', 'dose_skipped', 'stopped')
TARGETS = ('side_effect', 'purpose')
LEVELS = ('pt', 'hlt', 'hlgt', 'soc')
NOT_STATED = 'not stated'
NONE_STATED = 'none stated'   # action_taken: empty list
UNCODED = 'uncoded'           # code of the uncoded node (mentions without an adequate MedDRA term)
UNCODED_LABEL = '[uncoded: no adequate MedDRA term]'
POPULATION_DIMS = ('ingredient', 'brand', 'age_group', 'sex')
DETAIL_DIMS = ('pt', 'hlt', 'hlgt', 'soc', 'intensity', 'impact', 'action_taken')
GROUP_DIMS = ('hlt', 'hlgt', 'soc')
SPARSE_BELOW = 10             # too few records to support any statement
SMALL_BELOW = 30              # few records: a share is unstable and must be called small in words
MAX_ROWS = 300
MAX_TIES = 20                 # rows tied with the last kept row of a group that are kept beyond top_n
WINDOW_END = '2026-06-30'     # last FAERS receivedate in the release; the comparison window ends here on both sides
STOPWORDS = frozenset('a an the of for and or to in on with my i is are was be by at as it from'.split())
# Words that name no concept by themselves: link_terms refuses a query made only of these (and drug names), because
# side effects or purposes in general need no target and drugs are tool arguments, not graph concepts to link.
GENERIC_WORDS = frozenset('side effect effects adverse experience experiences event events reaction reactions complaint '
                          'complaints problem problems issue issues symptom symptoms reason reasons purpose purposes use uses '
                          'common commonly reported report reports people patients drug drugs medication medications glp '
                          'glp1 shot shots injection injections pill pills women woman men man female females male males '
                          'older younger age aged teen teens adults writers reviewers'.split())
DRUG_WORDS = frozenset(w.lower() for w in INGREDIENTS + tuple(BRAND_INGREDIENT))
SCHEMA = {'side_effect': ('SIDE_EFFECT', 'SideEffect'), 'purpose': ('USED_FOR', 'Purpose')}
FAERS_STATUS = ('FAERS data validated and frozen (openFDA release 2026-07-30, primary configuration); spontaneous '
                'reporting counts, not incidence')

ORDER = {
    'platform': PLATFORMS,
    'ingredient': INGREDIENTS + (UNSPECIFIED,),
    'brand': BRANDS + (UNSPECIFIED,),
    'age_group': AGE_GROUPS + (NOT_STATED,),
    'sex': SEXES + (NOT_STATED,),
    'intensity': INTENSITIES + (NOT_STATED,),
    'impact': IMPACTS + (NOT_STATED,),
    'action_taken': ACTIONS + (NONE_STATED,),
}
BASE_NOTES = [
    'Counts are records (Reddit posts or comments, WebMD reviews), not patients; one person can write several records.',
    'Shares describe what writers chose to report in each corpus; they are not incidence, prevalence or risk and carry no causal meaning.',
    'Reddit and WebMD are reported separately and never pooled.',
]


class QueryError(ValueError):
    """Invalid tool arguments; the message is returned to the model."""


# ---------------------------------------------------------------------------------------------------- tool arguments
Target = Literal['side_effect', 'purpose']
Platform = Literal[PLATFORMS]
Ingredient = Literal[INGREDIENTS + (UNSPECIFIED,)]
Brand = Literal[BRANDS + (UNSPECIFIED,)]
AgeGroup = Literal[AGE_GROUPS + (NOT_STATED,)]
Sex = Literal['female', 'male', 'other', NOT_STATED]
GroupDim = Literal[POPULATION_DIMS + DETAIL_DIMS]
Scope = Literal[('all',) + tuple(f'ingredient:{i}' for i in INGREDIENTS) + tuple(f'brand:{b}' for b in BRANDS)
                + tuple(f'unspecified_brand:{i}' for i in INGREDIENTS)]


class Concept(BaseModel):
    concept: str = Field(description='Short name of the concept this node counts toward, e.g. "weight loss". Nodes given the same '
                                     'name form one concept: a record counts once for it, whichever of its nodes it mentions.')
    level: Literal[LEVELS] = Field(description='pt for a single Preferred Term; hlt, hlgt or soc for a MedDRA group (all PTs '
                                                'whose primary path passes through it)')
    code: str = Field(description='MedDRA code from link_terms, or "uncoded" (level pt) for mentions without an adequate term')


class LinkTermsArgs(BaseModel):
    text: str = Field(description='A condition, goal or symptom in plain words, e.g. "type 2 diabetes", "throwing up" or '
                                  '"stomach problems". One concept per call.')
    kind: Target = Field(description='side_effect or purpose: the role the concept plays in the question')


class FilterArgs(BaseModel):
    target: Target = Field(description='side_effect (SIDE_EFFECT statements) or purpose (USED_FOR statements)')
    platforms: list[Platform] = Field(description='Platforms to report; each is reported separately, never pooled')
    ingredients: Optional[list[Ingredient]] = Field(description='Drug ingredients to keep, or null for all')
    brands: Optional[list[Brand]] = Field(description="Named brands to keep, or null for all; 'unspecified' = no brand named")
    named_drugs_only: bool = Field(description='true drops statements whose ingredient was assigned from the Reddit community '
                                               '(robustness check)')
    purposes_from_text_only: bool = Field(description='true keeps only purpose statements written in the text, dropping WebMD '
                                                      'profile conditions (for purpose targets and require_purposes)')
    age_groups: Optional[list[AgeGroup]] = Field(description='Writer age groups to keep, or null for no age filter')
    sexes: Optional[list[Sex]] = Field(description='Writer sex/gender values to keep, or null for no sex filter')
    require_purposes: Optional[list[Concept]] = Field(description='Keep only statements whose record states one of these '
                                                                  'purposes for the same drug, or null')
    require_side_effects: Optional[list[Concept]] = Field(description='Keep only statements whose record reports one of these '
                                                                      'side effects for the same drug, or null')
    targets: Optional[list[Concept]] = Field(description='Target nodes to count (numerator), grouped into concepts by their '
                                                         'concept name, or null for every PT')
    intensities: Optional[list[Literal[INTENSITIES + (NOT_STATED,)]]] = Field(
        description='Side effects only: keep statements with these intensities (the universe becomes those statements), or null')
    impacts: Optional[list[Literal[IMPACTS + (NOT_STATED,)]]] = Field(
        description='Side effects only: keep statements with these impacts, or null')
    actions: Optional[list[Literal[ACTIONS + (NONE_STATED,)]]] = Field(
        description='Side effects only: keep statements with one of these actions taken with the drug (e.g. ["stopped"] for '
                    '"led people to stop"), or null')


class QueryGraphArgs(FilterArgs):
    group_by: list[GroupDim] = Field(description='Dimensions to split rows by; platform is always a dimension. pt ranks PTs; '
                                                 'hlt/hlgt/soc rank MedDRA groups; intensity, impact and action_taken exist '
                                                 'only for side effects')
    top_n: int = Field(description='Rows kept per platform and population group (1-50), highest record counts first')


class GetEvidenceArgs(FilterArgs):
    wording: Optional[str] = Field(description='Words the statement wording must contain (e.g. "sulfur burps"), or null')
    limit: int = Field(description='Number of example statements, 1-10')


class CompareFaersArgs(BaseModel):
    targets: Optional[list[Concept]] = Field(description='Side-effect nodes to compare, one row per concept name, or null to rank PTs')
    scopes: list[Scope] = Field(description="Drug scopes, each reported separately: 'all', 'ingredient:<name>' (every product "
                                            "of the ingredient), 'brand:<name>', 'unspecified_brand:<ingredient>'")
    sexes: Optional[list[Literal['female', 'male']]] = Field(description='Sex filter applied to both sides, or null')
    age_groups: Optional[list[Literal[AGE_GROUPS]]] = Field(description='Age-group filter applied to both sides, or null')
    rank_by: Literal['faers', 'reddit', 'webmd'] = Field(description='Source whose counts rank PTs when targets is null')
    top_n: int = Field(description='PTs per scope when targets is null (1-30)')
    include_outcomes: bool = Field(description='true adds hospitalization and action-taken counts per scope')


TOOL_ARGS = {'link_terms': LinkTermsArgs, 'query_graph': QueryGraphArgs, 'get_evidence': GetEvidenceArgs,
             'compare_faers': CompareFaersArgs}


# ---------------------------------------------------------------------------------------------------- helpers
def percent(num, den):
    return round(100.0 * num / den, 1) if den else None


def wilson(num, den, z=1.96):
    """95% Wilson score interval of a share, in percent (low, high), or (None, None) without a denominator. Records are
    treated as independent, which they are not quite (one writer can post several times), so the interval is approximate."""
    if not den:
        return None, None
    p = num / den
    centre = (p + z * z / (2 * den)) / (1 + z * z / den)
    half = z * math.sqrt(p * (1 - p) / den + z * z / (4 * den * den)) / (1 + z * z / den)
    return round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)


def _sort_value(dim, value):
    order = ORDER.get(dim)
    if order and value in order:
        return (0, order.index(value), '')
    return (1, 0, str(value))


def _clip(text, limit, focus=None):
    """At most `limit` characters of `text`, centred on `focus` when it occurs in the text."""
    if text is None or len(text) <= limit:
        return text
    start = 0
    if focus:
        i = text.lower().find(focus.lower())
        if i > 0:
            start = max(0, min(i - limit // 3, len(text) - limit))
    piece = text[start:start + limit]
    return ('…' if start > 0 else '') + piece + ('…' if start + limit < len(text) else '')


def number_rows(output, *sections):
    """Give every dict in the named list sections a `row` number, unique within the output, in section order."""
    n = 0
    for name in sections:
        for row in output.get(name, []):
            n += 1
            row['row'] = n
    return output


def interval_order(row, x, y):
    """'higher', 'lower' or 'similar' for source x against source y by their 95% intervals ('insufficient' when a
    denominator is below SPARSE_BELOW)."""
    if (row[f'{x}_denominator'] or 0) < SPARSE_BELOW or (row[f'{y}_denominator'] or 0) < SPARSE_BELOW:
        return 'insufficient'
    if row[f'{x}_ci95_low'] > row[f'{y}_ci95_high']:
        return 'higher'
    if row[f'{x}_ci95_high'] < row[f'{y}_ci95_low']:
        return 'lower'
    return 'similar'


def clear_differences(rows, popd, limit=40):
    """Pairs of rows that differ only in their population values and whose 95% intervals do not overlap (neither row
    sparse), as {higher_row, lower_row, higher, lower}; at most `limit` pairs, largest gaps between intervals first."""
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get('sparse') or r.get('ci95_low') is None:
            continue
        key = tuple((k, v) for k, v in r.items() if k not in popd and k not in (
            'row', 'records', 'denominator', 'percent', 'ci95_low', 'ci95_high', 'sparse', 'small', 'common_wordings'))
        groups[key].append(r)
    pairs = []
    for members in groups.values():
        for a, b in itertools.combinations(members, 2):
            hi, lo = (a, b) if a['percent'] >= b['percent'] else (b, a)
            gap = lo['ci95_high'] < hi['ci95_low'] and hi['ci95_low'] - lo['ci95_high']
            if gap:
                pairs.append((gap, {'higher_row': hi['row'], 'lower_row': lo['row'],
                                    'higher': ', '.join(str(hi[d]) for d in popd), 'lower': ', '.join(str(lo[d]) for d in popd)}))
    pairs.sort(key=lambda x: -x[0])
    return [p for _, p in pairs[:limit]]


def _public_filters(f):
    return {k: v for k, v in f.items() if not k.startswith('_')}


def _row(**fields):
    out = collections.OrderedDict(row=None)
    out.update(fields)
    return out


def lucene_query(text, mode='AND'):
    words = [w for w in re.findall(r'[a-z0-9]+', (text or '').lower()) if w not in STOPWORDS]
    if not words:
        raise QueryError('text has no searchable words')
    return f' {mode} '.join(f'({w} OR {w}*)' if len(w) > 2 else w for w in words), words


def _check_list(name, values, allowed):
    if values is None:
        return None
    values = list(dict.fromkeys(values))
    bad = [v for v in values if v not in allowed]
    if bad:
        raise QueryError(f'{name}: unknown values {bad}; allowed {list(allowed)}')
    return values


# Population and detail expressions over the statement pattern (d:Drug)-[e]->(n)
POP_EXPR = {'ingredient': 'd.ingredient', 'brand': 'd.brand', 'age_group': f"coalesce(e.writer_age_group, '{NOT_STATED}')",
            'sex': f"coalesce(e.writer_sex, '{NOT_STATED}')"}
DETAIL_EXPR = {'pt': f"coalesce(n.pt_code, '{UNCODED}')", 'hlt': f"coalesce(h.code, '{UNCODED}')",
               'hlgt': f"coalesce(g.code, '{UNCODED}')", 'soc': f"coalesce(s.code, '{UNCODED}')",
               'intensity': f"coalesce(e.intensity, '{NOT_STATED}')", 'impact': f"coalesce(e.impact, '{NOT_STATED}')",
               'action_taken': 'action'}
LABEL_EXPR = {'pt': f"coalesce(n.pt, '{UNCODED_LABEL}')", 'hlt': f"coalesce(h.name, '{UNCODED_LABEL}')",
              'hlgt': f"coalesce(g.name, '{UNCODED_LABEL}')", 'soc': f"coalesce(s.name, '{UNCODED_LABEL}')"}
PATH = 'OPTIONAL MATCH (n)-[:IN_HLT]->(h:HLT)-[:IN_HLGT]->(g:HLGT)-[:IN_SOC]->(s:SOC)'
ACTION_UNWIND = (f"UNWIND CASE WHEN size(coalesce(e.action_taken, [])) = 0 THEN ['{NONE_STATED}'] "
                 "ELSE e.action_taken END AS action")


# ---------------------------------------------------------------------------------------------------- the graph
class KnowledgeGraph:
    """Read-only access to the local Neo4j graph loaded by weightloss.graph.neo4j_db."""

    def __init__(self, driver=None):
        self.driver = driver or neo4j_db.connect()
        self.manifest = neo4j_db.manifest() or {}
        self._lock = threading.Lock()
        self._overview = None
        dates = neo4j_db.approval_dates()
        self.brand_start = dict(dates)
        self.ingredient_start = {i: min(d for b, d in dates.items() if BRAND_INGREDIENT[b] == i) for i in INGREDIENTS}
        self.all_start = min(dates.values())

    def close(self):
        self.driver.close()

    def run(self, cypher, **params):
        with self.driver.session(default_access_mode='READ') as s:
            return s.run(cypher, **params).data()

    # ------------------------------------------------------------------ concepts
    def resolve(self, kind, concepts):
        """Node ids of `kind` under the concepts, the resolved concepts (name, PT count) and codes not in the graph."""
        if len(concepts) > 1 and any(c['level'] == 'pt' and str(c['code']).strip() == UNCODED for c in concepts):
            raise QueryError("'uncoded' selects every mention without an adequate MedDRA term, whatever it describes; do not "
                             'combine it with other concepts. Report uncoded wordings from link_terms instead.')
        _, label = SCHEMA[kind]
        ids, resolved, missing = set(), [], []
        for c in concepts:
            level, code = c['level'], str(c['code']).strip()
            if level == 'pt' and code == UNCODED:
                rows = self.run(f'MATCH (n:{label}) WHERE n.pt_code IS NULL RETURN n.id AS id, $u AS name', u=UNCODED_LABEL)
            elif level == 'pt':
                rows = self.run(f'MATCH (n:{label} {{pt_code: $code}}) RETURN n.id AS id, n.pt AS name', code=code)
            else:
                path = {'hlt': '(n)-[:IN_HLT]->(x:HLT)', 'hlgt': '(n)-[:IN_HLT]->(:HLT)-[:IN_HLGT]->(x:HLGT)',
                        'soc': '(n)-[:IN_HLT]->(:HLT)-[:IN_HLGT]->(:HLGT)-[:IN_SOC]->(x:SOC)'}[level]
                rows = self.run(f'MATCH {path.replace("(n)", f"(n:{label})")} WHERE x.code = $code RETURN n.id AS id, x.name AS name',
                                code=code)
            if not rows:
                missing.append(f'{level}:{code}')
                continue
            ids.update(r['id'] for r in rows)
            resolved.append({'level': level, 'code': code, 'name': rows[0]['name'], 'pts': len(rows)})
        return sorted(ids), resolved, missing

    def resolve_sets(self, kind, concepts):
        """Concepts by name: [{concept, ids, members}] in the order the names first appear, plus codes not in the graph.
        A record counts once for a concept whichever of its member nodes it mentions."""
        if len(concepts) > 1 and any(c['level'] == 'pt' and str(c['code']).strip() == UNCODED for c in concepts):
            self.resolve(kind, concepts)  # raises the uncoded-mixing error
        sets, missing = collections.OrderedDict(), []
        for c in concepts:
            ids, res, miss = self.resolve(kind, [c])
            missing += miss
            name = (c.get('concept') or '').strip() or f"{c['level']}:{c['code']}"
            s = sets.setdefault(name, {'concept': name, 'ids': set(), 'members': []})
            s['ids'].update(ids)
            s['members'] += [{k: r[k] for k in ('level', 'code', 'name', 'pts')} for r in res]
        out = [{'concept': s['concept'], 'ids': sorted(s['ids']), 'members': s['members']} for s in sets.values()]
        return out, missing

    def pt_codes(self, kind, ids):
        """PT codes of node ids (uncoded nodes give 'uncoded')."""
        _, label = SCHEMA[kind]
        return {r['c'] for r in self.run(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN coalesce(n.pt_code, '{UNCODED}') AS c",
                                         ids=list(ids))}

    # ------------------------------------------------------------------ filters
    def _filters(self, a):
        if a['target'] not in TARGETS:
            raise QueryError(f'target must be one of {TARGETS}')
        platforms = _check_list('platforms', a['platforms'], PLATFORMS)
        if not platforms:
            raise QueryError('platforms must name at least one platform')
        f = {
            'target': a['target'],
            'platforms': [p for p in PLATFORMS if p in platforms],
            'ingredients': _check_list('ingredients', a.get('ingredients'), INGREDIENTS + (UNSPECIFIED,)),
            'brands': _check_list('brands', a.get('brands'), BRANDS + (UNSPECIFIED,)),
            'named_drugs_only': bool(a.get('named_drugs_only')),
            'age_groups': _check_list('age_groups', a.get('age_groups'), AGE_GROUPS + (NOT_STATED,)),
            'sexes': _check_list('sexes', a.get('sexes'), SEXES + (NOT_STATED,)),
        }
        missing, resolved = {}, {}
        params = {'platforms': f['platforms']}
        where = ['e.platform IN $platforms']
        if f['ingredients'] is not None:
            where.append('d.ingredient IN $ingredients')
            params['ingredients'] = f['ingredients']
        if f['brands'] is not None:
            where.append('d.brand IN $brands')
            params['brands'] = f['brands']
        if f['named_drugs_only']:
            where.append("e.ingredient_basis IN ['name', 'profile']")
        f['purposes_from_text_only'] = bool(a.get('purposes_from_text_only'))
        if f['purposes_from_text_only'] and f['target'] == 'purpose':
            where.append("e.source = 'text'")
        details = {k: a.get(k) for k in ('intensities', 'impacts', 'actions')}
        if f['target'] != 'side_effect' and any(details.values()):
            raise QueryError('intensities, impacts and actions exist only for side effects')
        f['intensities'] = _check_list('intensities', details['intensities'], INTENSITIES + (NOT_STATED,))
        f['impacts'] = _check_list('impacts', details['impacts'], IMPACTS + (NOT_STATED,))
        f['actions'] = _check_list('actions', details['actions'], ACTIONS + (NONE_STATED,))
        if f['intensities'] is not None:
            where.append(f"{DETAIL_EXPR['intensity']} IN $intensities")
            params['intensities'] = f['intensities']
        if f['impacts'] is not None:
            where.append(f"{DETAIL_EXPR['impact']} IN $impacts")
            params['impacts'] = f['impacts']
        if f['actions'] is not None:
            where.append(f"(any(x IN coalesce(e.action_taken, []) WHERE x IN $actions) OR "
                         f"('{NONE_STATED}' IN $actions AND size(coalesce(e.action_taken, [])) = 0))")
            params['actions'] = f['actions']
        counted = {'purpose': set(), 'side_effect': set()}
        for kind, key in (('purpose', 'require_purposes'), ('side_effect', 'require_side_effects')):
            concepts = a.get(key)
            if concepts is None:
                continue
            concepts = [dict(c) for c in concepts]
            sets, miss = self.resolve_sets(kind, concepts)
            ids = sorted({i for s in sets for i in s['ids']})
            resolved[key] = [{'concept': s['concept'], 'members': s['members']} for s in sets]
            f[key] = concepts
            counted[kind] |= self.pt_codes(kind, ids)
            if miss:
                missing[key] = miss
            rel, label = SCHEMA[kind]
            text_only = " AND r.source = 'text'" if kind == 'purpose' and f['purposes_from_text_only'] else ''
            keys = self.run(f'MATCH (d:Drug)-[r:{rel}]->(m:{label}) WHERE m.id IN $ids AND r.platform IN $platforms{text_only} '
                            "RETURN collect(DISTINCT r.record_key + '|' + d.id) AS k", ids=ids, platforms=f['platforms'])[0]['k']
            where.append(f"(e.record_key + '|' + d.id) IN $req_{kind}")
            params[f'req_{kind}'] = keys
        target_ids = None
        f['_target_sets'] = None
        if a.get('targets') is not None:
            concepts = [dict(c) for c in a['targets']]
            sets, miss = self.resolve_sets(f['target'], concepts)
            target_ids = sorted({i for s in sets for i in s['ids']})
            resolved['targets'] = [{'concept': s['concept'], 'members': s['members']} for s in sets]
            f['targets'], f['_target_sets'] = concepts, sets
            counted[f['target']] |= self.pt_codes(f['target'], target_ids)
            if miss:
                missing['targets'] = miss
            params['target_ids'] = target_ids
        f['_counted'] = counted
        pop = []
        if f['age_groups'] is not None:
            pop.append(f"{POP_EXPR['age_group']} IN $age_groups")
            params['age_groups'] = f['age_groups']
        if f['sexes'] is not None:
            pop.append(f"{POP_EXPR['sex']} IN $sexes")
            params['sexes'] = f['sexes']
        return f, where, pop, params, target_ids, resolved, missing

    def _match(self, f):
        rel, label = SCHEMA[f['target']]
        return f'MATCH (d:Drug)-[e:{rel}]->(n:{label})'

    def _summary_rows(self, f, where, pop, params, dims=()):
        base = f"{self._match(f)} WHERE {' AND '.join(where)}"
        before = {r['platform']: r for r in self.run(
            f"{base} RETURN e.platform AS platform, count(DISTINCT e.record_key) AS records, "
            "count(DISTINCT CASE WHEN e.writer_age_group IS NULL THEN e.record_key END) AS age_missing, "
            "count(DISTINCT CASE WHEN e.writer_sex IS NULL THEN e.record_key END) AS sex_missing", **params)}
        after = {r['platform']: r for r in self.run(
            f"{self._match(f)} WHERE {' AND '.join(where + pop)} RETURN e.platform AS platform, "
            "count(DISTINCT e.record_key) AS records, "
            "count(DISTINCT CASE WHEN e.ingredient_basis = 'community' THEN e.record_key END) AS community, "
            "count(DISTINCT CASE WHEN n.pt_code IS NULL THEN e.record_key END) AS uncoded, "
            "count(DISTINCT CASE WHEN e.source = 'profile' THEN e.record_key END) AS profile", **params)}
        age = f['age_groups'] is not None or 'age_group' in dims
        sex = f['sexes'] is not None or 'sex' in dims
        rows = []
        for p in f['platforms']:
            b, a = before.get(p, {}), after.get(p, {})
            n = b.get('records', 0)
            row = _row(platform=p, records_before_age_sex_filter=n)
            if age:
                row['records_with_age_stated'] = n - b.get('age_missing', 0)
            if sex:
                row['records_with_sex_stated'] = n - b.get('sex_missing', 0)
            row.update(
                excluded_missing_age=b.get('age_missing', 0) if f['age_groups'] is not None and NOT_STATED not in f['age_groups'] else None,
                excluded_missing_sex=b.get('sex_missing', 0) if f['sexes'] is not None and NOT_STATED not in f['sexes'] else None,
                universe_records=a.get('records', 0),
                universe_records_with_community_ingredient=a.get('community', 0),
                universe_records_with_uncoded_mention=a.get('uncoded', 0))
            if f['target'] == 'purpose':
                row['universe_records_with_profile_purpose'] = a.get('profile', 0)
            rows.append(row)
        return rows

    def _notes(self, f, dims):
        notes = list(BASE_NOTES)
        if f['target'] == 'side_effect':
            notes.append('Side effects are symptoms the writer reports having and links to the drug; feared, asked-about or '
                         'second-hand effects and statements of no side effects are not in the graph.')
        else:
            notes.append("Purposes are the writer's stated reasons for using the drug. Most WebMD purpose statements are the condition "
                         "the reviewer chose from the site's list in the profile (universe_records_with_profile_purpose); Reddit "
                         "purposes all come from free text. Purpose shares therefore measure different things on the two platforms; "
                         "purposes_from_text_only=true restricts both to purposes written in the text.")
        if f.get('purposes_from_text_only'):
            notes.append('purposes_from_text_only=true: WebMD profile conditions are left out.')
        if not f['named_drugs_only'] and (f['ingredients'] is not None or 'ingredient' in dims):
            notes.append("Ingredient-level counts include references that name no drug but come from a single-ingredient Reddit "
                         "community (ingredient_basis 'community'); named_drugs_only=true is the robustness check restricted to named drugs.")
        if f['brands'] is not None or 'brand' in dims:
            notes.append("Brand-level values use only brands named in the text or the WebMD profile; 'unspecified' means no brand was "
                         "named and is never attributed to a brand.")
        if f['age_groups'] is not None or f['sexes'] is not None or 'age_group' in dims or 'sex' in dims:
            notes.append("Reddit age group and sex are known only when the writer states them about themselves, so the records "
                         "with these values are a self-disclosed subset; WebMD values come from the reviewer profile. A filter "
                         "drops records with a missing value (excluded_missing_age / excluded_missing_sex).")
        if f.get('require_purposes') is not None or f.get('require_side_effects') is not None:
            notes.append('require_* conditions are statements for the same drug in the same record.')
        if any(f.get(k) is not None for k in ('intensities', 'impacts', 'actions')):
            notes.append('intensities/impacts/actions restrict the universe to side-effect statements with those details, so the '
                         'denominator counts records with at least one such statement (for actions ["stopped"]: records in which '
                         'the writer stopped the drug because of a side effect).')
        if set(dims) & set(GROUP_DIMS) or any(c['level'] != 'pt' for c in f.get('targets') or []):
            notes.append('MedDRA groups (HLT, HLGT, SOC) follow each PT\'s primary path, so a PT sits in one group per level; a '
                         'record counts once per group.')
        if f.get('targets') is not None:
            notes.append('Each concept row counts records with at least one statement to any of its member nodes (a record counts '
                         'once per concept); resolved_concepts lists the members.')
        notes.append(f"'{UNCODED_LABEL}' collects mentions without an adequate MedDRA term; it is not a PT. "
                     'universe_records_with_uncoded_mention tells how many records hold such a mention.')
        notes.append(f'sparse = denominator below {SPARSE_BELOW} (too few records to support a statement); small = denominator '
                     f'below {SMALL_BELOW} (say in words that the group is small).')
        notes.append('ci95_low/ci95_high: approximate 95% Wilson interval of percent (records treated as independent). Call one '
                     'share higher than another only when their intervals do not overlap.')
        return notes

    # ------------------------------------------------------------------ tools
    def overview(self):
        """Size of the graph per platform and of the FAERS layer (computed once per KnowledgeGraph)."""
        with self._lock:
            if self._overview is None:
                self._overview = self._compute_overview()
        return json.loads(json.dumps(self._overview), object_pairs_hook=collections.OrderedDict)

    def _compute_overview(self):
        out = collections.OrderedDict(tool='overview', graph=self.manifest.get('kg_version'),
                                      meddra_version='29.1', loaded_utc=self.manifest.get('loaded_utc'))
        counts = (self.manifest.get('kg_build') or {}).get('counts') or {}
        rows = []
        for p in PLATFORMS:
            se = self.run('MATCH ()-[e:SIDE_EFFECT {platform: $p}]->() RETURN count(DISTINCT e.record_key) AS r, count(e) AS e', p=p)[0]
            uf = self.run('MATCH ()-[e:USED_FOR {platform: $p}]->() RETURN count(DISTINCT e.record_key) AS r, count(e) AS e', p=p)[0]
            rows.append(_row(platform=p, records_processed=counts.get(f'records_{p}'),
                             records_with_side_effect=se['r'], side_effect_statements=se['e'],
                             records_with_purpose=uf['r'], purpose_statements=uf['e']))
        faers = self.run('MATCH (r:Report) WHERE r.in_primary_analysis RETURN count(r) AS c')[0]['c']
        out['rows'] = rows
        out['faers'] = [_row(source='FAERS', reports_primary_configuration=faers, status=FAERS_STATUS)]
        out['notes'] = ['records_processed counts every record given to extraction; records without any side effect or purpose '
                        'have no statements.'] + BASE_NOTES[:1]
        return number_rows(out, 'rows', 'faers')

    def link_terms(self, text, kind):
        """PT and MedDRA group nodes whose name, or the wording of their statements, matches every query word."""
        if kind not in TARGETS:
            raise QueryError(f'kind must be one of {TARGETS}')
        rel, label = SCHEMA[kind]
        q, words = lucene_query(text)
        if all(w in GENERIC_WORDS or w in DRUG_WORDS or w.isdigit() for w in words):
            raise QueryError(f'"{text}" names no specific condition, goal or symptom (only general words or drug names). '
                             'For side effects or purposes in general, call query_graph with targets null; pass drugs as '
                             'ingredients or brands.')

        def search(query):
            names = self.run("CALL db.index.fulltext.queryNodes('pt_names', $q) YIELD node, score "
                             f"WHERE '{label}' IN labels(node) RETURN node.id AS id, score", q=query)
            verb = self.run("CALL db.index.fulltext.queryRelationships('verbatims', $q) YIELD relationship AS r "
                            f"WHERE type(r) = '{rel}' WITH endNode(r) AS n, r "
                            "WITH n, toLower(trim(r.text)) AS t, r.platform AS p, r.record_key AS k "
                            "WITH n, t, collect(DISTINCT CASE WHEN p = 'reddit' THEN k END) AS rr, "
                            "collect(DISTINCT CASE WHEN p = 'webmd' THEN k END) AS ww "
                            "RETURN n.id AS id, collect({t: t, c: size(rr) + size(ww)}) AS texts, "
                            "reduce(a = [], x IN collect(rr) | a + x) AS rr, reduce(a = [], x IN collect(ww) | a + x) AS ww",
                            q=query)
            return names, verb

        names, verb = search(q)
        partial = not names and not verb and len(words) > 1
        if partial:
            names, verb = search(lucene_query(text, 'OR')[0])
        hit = {r['id']: {'name': True} for r in names}
        for r in verb:
            texts = sorted(r['texts'], key=lambda x: (-x['c'], x['t']))
            hit.setdefault(r['id'], {})['verbatim'] = {
                'reddit': len(set(r['rr'])), 'webmd': len(set(r['ww'])), 'examples': [x['t'] for x in texts[:4]]}
        ids = list(hit)
        totals = {}
        if ids:
            for r in self.run(f'UNWIND $ids AS id MATCH (n:{label} {{id: id}}) OPTIONAL MATCH (n)<-[e:{rel}]-() '
                              'RETURN id, n.pt AS pt, n.pt_code AS code, '
                              "count(DISTINCT CASE WHEN e.platform = 'reddit' THEN e.record_key END) AS reddit, "
                              "count(DISTINCT CASE WHEN e.platform = 'webmd' THEN e.record_key END) AS webmd", ids=ids):
                totals[r['id']] = r
        faers = {}
        if kind == 'side_effect' and ids:
            for r in self.run('UNWIND $ids AS id MATCH (n:SideEffect {id: id})<-[:REACTION]-(r:Report) '
                              'WHERE r.in_primary_analysis RETURN id, count(r) AS c', ids=ids):
                faers[r['id']] = r['c']
        rows = []
        for node_id, h in hit.items():
            t = totals.get(node_id)
            if t is None or (t['reddit'] + t['webmd'] == 0 and not faers.get(node_id)):
                continue  # PT known only from FAERS with no FAERS primary report either
            v = h.get('verbatim') or {}
            row = _row(level='pt', code=t['code'] or UNCODED, name=t['pt'] or UNCODED_LABEL,
                       match='+'.join(m for m in ('name', 'verbatim') if m in h),
                       records_reddit=t['reddit'], records_webmd=t['webmd'],
                       matching_wording_records_reddit=v.get('reddit', 0), matching_wording_records_webmd=v.get('webmd', 0),
                       example_wordings=v.get('examples', []))
            if kind == 'side_effect':
                row['faers_reports'] = faers.get(node_id, 0)
            rows.append(row)
        # PTs by the records whose wording matched (not by their total size, so a large PT with a few incidental matches does
        # not outrank the PT the words describe); a matching PT name breaks ties.
        rows.sort(key=lambda r: (r['code'] == UNCODED, -(r['matching_wording_records_reddit'] + r['matching_wording_records_webmd']),
                                 'name' not in r['match'], -(r['records_reddit'] + r['records_webmd']), r['name']))
        # Group candidates: MedDRA groups whose name matches every word, then the primary-path parents (HLT, HLGT, SOC)
        # of the three best coded PT matches.
        level_rank = {'SOC': 0, 'HLGT': 1, 'HLT': 2}
        groups = sorted(self.run("CALL db.index.fulltext.queryNodes('group_names', $q) YIELD node, score "
                                 "RETURN node.code AS code, node.name AS name, labels(node)[0] AS level ORDER BY score DESC LIMIT 6",
                                 q=q), key=lambda g: level_rank[g['level']])
        top = [f"{kind}:pt:{r['code']}" for r in rows if r['code'] != UNCODED][:3]
        if top:
            for r in self.run(f'UNWIND $ids AS id MATCH (n:{label} {{id: id}})-[:IN_HLT]->(h:HLT)-[:IN_HLGT]->(g:HLGT)-[:IN_SOC]->(s:SOC) '
                              'RETURN h.code AS hc, h.name AS hn, g.code AS gc, g.name AS gn, s.code AS sc, s.name AS sn', ids=top):
                groups += [{'code': r['hc'], 'name': r['hn'], 'level': 'HLT'}, {'code': r['gc'], 'name': r['gn'], 'level': 'HLGT'},
                           {'code': r['sc'], 'name': r['sn'], 'level': 'SOC'}]
        groups = list({(g['level'], g['code']): g for g in groups}.values())
        group_rows = []
        for g in groups:
            level = g['level'].lower()
            path = {'hlt': '(n)-[:IN_HLT]->(x:HLT)', 'hlgt': '(n)-[:IN_HLT]->(:HLT)-[:IN_HLGT]->(x:HLGT)',
                    'soc': '(n)-[:IN_HLT]->(:HLT)-[:IN_HLGT]->(:HLGT)-[:IN_SOC]->(x:SOC)'}[level]
            c = self.run(f'MATCH {path.replace("(n)", f"(n:{label})")} WHERE x.code = $code '
                         f'OPTIONAL MATCH (n)<-[e:{rel}]-() RETURN count(DISTINCT n) AS pts, '
                         "count(DISTINCT CASE WHEN e.platform = 'reddit' THEN e.record_key END) AS reddit, "
                         "count(DISTINCT CASE WHEN e.platform = 'webmd' THEN e.record_key END) AS webmd", code=g['code'])[0]
            if c['reddit'] + c['webmd']:
                group_rows.append(_row(level=level, code=g['code'], name=g['name'], pts_with_statements=c['pts'],
                                       records_reddit=c['reddit'], records_webmd=c['webmd']))
        related = []
        shown = {r['code'] for r in rows[:15]}
        if top:  # other PTs of the best matches' HLTs, whose wordings did not contain the query words (synonyms, abbreviations)
            for r in self.run(f'UNWIND $ids AS id MATCH (m:{label} {{id: id}})-[:IN_HLT]->(h:HLT)<-[:IN_HLT]-(n:{label}) '
                              f'WHERE n.id <> m.id MATCH (n)<-[e:{rel}]-() WITH n, h, '
                              "count(DISTINCT CASE WHEN e.platform = 'reddit' THEN e.record_key END) AS reddit, "
                              "count(DISTINCT CASE WHEN e.platform = 'webmd' THEN e.record_key END) AS webmd "
                              'WHERE reddit + webmd >= 10 RETURN DISTINCT n.pt_code AS code, n.pt AS name, h.name AS hlt, reddit, webmd, '
                              'n.common_wordings AS w ORDER BY reddit + webmd DESC', ids=top):
                if r['code'] not in shown and len(related) < 8:
                    shown.add(r['code'])
                    related.append(_row(level='pt', code=r['code'], name=r['name'], hlt=r['hlt'], records_reddit=r['reddit'],
                                        records_webmd=r['webmd'], common_wordings=(r['w'] or [])[:2]))
        out = collections.OrderedDict(tool='link_terms', text=text, kind=kind, words=words)
        out['rows'] = rows[:15]
        out['related'] = related
        out['groups'] = group_rows
        out['total_pt_matches'] = len(rows)
        out['notes'] = [
            'Rows are PT nodes whose name or statement wording contains every query word (English stemming); records_* count '
            'all records with a statement to that PT, matching_wording_records_* only those whose wording matched.',
            'One meaning is often coded to several PTs (the rows show where writers\' wordings went). Give every row that '
            'expresses the asked meaning the same concept name in query_graph or compare_faers, so a record counts once for it.',
            'related are other PTs in the same MedDRA HLT as the best matches whose wordings did not contain the query words; check '
            'whether any expresses the asked meaning in other words (a synonym, an abbreviation, a more specific term).',
            'groups are MedDRA HLT/HLGT/SOC nodes whose name matches, plus the primary-path groups above the best PT matches; use '
            'one as a concept member (level hlt/hlgt/soc) when it matches a broad category as a whole, such as "stomach problems".',
            f"'{UNCODED_LABEL}' (code 'uncoded') is not a PT; a match there means some wordings received no term.",
            'Use code values as concepts in query_graph, get_evidence and compare_faers.']
        if partial:
            out['notes'].insert(0, 'No PT matched every word; these rows match at least one word. Check that they fit the question.')
        if not rows and not group_rows:
            out['notes'].insert(0, 'Nothing matched; try a single key word or a synonym.')
        out['_kind'] = kind
        return number_rows(out, 'rows', 'related', 'groups')

    def query_graph(self, **a):
        f, where, pop, params, target_ids, resolved, missing = self._filters(a)
        dims = list(dict.fromkeys(a.get('group_by') or ()))
        bad = [d for d in dims if d not in POPULATION_DIMS + DETAIL_DIMS]
        if bad:
            raise QueryError(f'group_by: unknown dimensions {bad}')
        if f['target'] == 'purpose' and set(dims) & {'intensity', 'impact', 'action_taken'}:
            raise QueryError('intensity, impact and action_taken exist only for side effects')
        top_n = max(1, min(int(a.get('top_n') or 15), 50))
        popd = [d for d in dims if d in POPULATION_DIMS]
        det = [d for d in dims if d in DETAIL_DIMS]
        sets = f['_target_sets']
        names = [s['concept'] for s in sets] if sets is not None else []
        cond = ' AND '.join(where + pop)
        pop_ret = ''.join(f', {POP_EXPR[d]} AS {d}' for d in popd)
        den_rows = self.run(f"{self._match(f)} WHERE {cond} RETURN e.platform AS platform{pop_ret}, "
                            "count(DISTINCT e.record_key) AS n", **params)
        universe = {r['platform']: 0 for r in den_rows}
        dens = {}
        for r in den_rows:
            dens[(r['platform'], tuple(r[d] for d in popd))] = r['n']
        summary = self._summary_rows(f, where, pop, params, dims)
        for s in summary:
            universe[s['platform']] = s['universe_records']
        distribution = not det and sets is None
        labels = {}
        rows = []  # (platform, population key, concept or None, detail key, records, denominator)
        if distribution:
            for (p, pk), n in dens.items():
                rows.append((p, pk, None, (), n, universe.get(p, 0)))
        else:
            pre = PATH if set(det) & set(GROUP_DIMS) else ''
            unwind = ACTION_UNWIND if 'action_taken' in det else ''
            det_ret = ''.join(f', {DETAIL_EXPR[d]} AS {d}' for d in det)
            lab_ret = ''.join(f', {LABEL_EXPR[d]} AS {d}_label' for d in det if d in LABEL_EXPR)
            if sets is not None:
                head = 'UNWIND $target_sets AS ts '
                tcond = cond + ' AND n.id IN ts.ids'
                cret = ', ts.name AS concept'
                params['target_sets'] = [{'name': s['concept'], 'ids': s['ids']} for s in sets]
            else:
                head, tcond, cret = '', cond, ', null AS concept'
            num_rows = self.run(f"{head}{self._match(f)} WHERE {tcond} {pre} {unwind} RETURN e.platform AS platform{pop_ret}"
                                f"{cret}{det_ret}{lab_ret}, count(DISTINCT e.record_key) AS n", **params)
            nums = {}
            for r in num_rows:
                key = (r['platform'], tuple(r[d] for d in popd), r['concept'], tuple(r[d] for d in det))
                nums[key] = r['n']
                for d in det:
                    if d in LABEL_EXPR:
                        labels[(d, r[d])] = r[f'{d}_label']
            if sets is not None:  # concepts (and member PTs) with no record still get a row
                for (p, pk) in dens:
                    for s in sets:
                        if not det:
                            nums.setdefault((p, pk, s['concept'], ()), 0)
                        elif det == ['pt'] and all(m['level'] == 'pt' for m in s['members']):
                            for m in s['members']:
                                labels.setdefault(('pt', m['code']), m['name'])
                                nums.setdefault((p, pk, s['concept'], (m['code'],)), 0)
            for (p, pk, c, dk), n in nums.items():
                rows.append((p, pk, c, dk, n, dens.get((p, pk), 0)))

        def key(r):
            p, pk, c, dk, n, _ = r
            return ((PLATFORMS.index(p),) + tuple(_sort_value(d, v) for d, v in zip(popd, pk))
                    + ((names.index(c),) if c is not None else ()) + (-n,)
                    + tuple(_sort_value(d, labels.get((d, v), v)) for d, v in zip(det, dk)))

        rows.sort(key=key)
        group_of = (lambda r: (r[0],)) if distribution else (lambda r: (r[0], r[1], r[2]))
        n_groups = len({group_of(r) for r in rows})
        top_applied = min(top_n, max(1, MAX_ROWS // max(1, n_groups)))
        kept, truncated, seen, last, ties = [], collections.Counter(), collections.Counter(), {}, 0
        for r in rows:
            g = group_of(r)
            seen[g] += 1
            if seen[g] > top_applied and not (r[4] == last.get(g) and seen[g] <= top_applied + MAX_TIES):
                truncated[r[0]] += 1
                continue
            if seen[g] > top_applied:
                ties += 1
            last[g] = r[4]
            kept.append(r)
        wordings = {}
        pt_codes = [dk[det.index('pt')] for (_, _, _, dk, _, _) in kept if 'pt' in det]
        if pt_codes:
            _, label = SCHEMA[f['target']]
            wordings = {r['c']: r['w'] for r in self.run(f'MATCH (n:{label}) WHERE n.pt_code IN $codes '
                                                          'RETURN n.pt_code AS c, n.common_wordings AS w', codes=pt_codes)}
        out_rows = []
        for p, pk, c, dk, n, den in kept:
            row = _row(platform=p)
            for d, v in zip(popd, pk):
                row[d] = v
            if c is not None:
                row['concept'] = c
            for d, v in zip(det, dk):
                if d in LABEL_EXPR:
                    row[d] = labels.get((d, v), v)
                    row[f'{d}_code'] = v
                else:
                    row[d] = v
            if 'pt' in det and wordings.get(dk[det.index('pt')]):
                row['common_wordings'] = wordings[dk[det.index('pt')]][:2]
            row['records'] = n
            row['denominator'] = den
            row['percent'] = percent(n, den)
            row['ci95_low'], row['ci95_high'] = wilson(n, den)
            row['sparse'] = den < SPARSE_BELOW
            row['small'] = den < SMALL_BELOW
            out_rows.append(row)
        if distribution:
            den_def = ('No PT, group or detail dimension and no targets: records counts the records in each group and denominator '
                       'is all records in the platform universe, so percent is the distribution across groups'
                       + (' (a record with several drugs can fall in several groups).' if set(popd) & {'ingredient', 'brand'} else '.'))
        else:
            den_def = ('records = records in the group with at least one '
                       + ('statement to a member of the row\'s concept' if sets is not None else f'{f["target"]} statement')
                       + (' with the row values' if det else '')
                       + f'; denominator = records with at least one {f["target"]} statement in the filtered universe and the '
                       'same population group' + (f" ({', '.join(popd)})" if popd else '') + '.')
        out = collections.OrderedDict(tool='query_graph', filters={**_public_filters(f), 'group_by': dims, 'top_n': top_n,
                                                                    'top_n_applied': top_applied})
        if resolved:
            out['resolved_concepts'] = resolved
        out['summary'] = summary
        out['rows'] = out_rows
        out['denominator_definition'] = den_def
        if truncated:
            out['rows_not_shown'] = dict(truncated)
        if ties:
            out['rows_kept_for_ties'] = ties
        if top_applied < top_n:
            out['top_n_reduced'] = f'top_n lowered from {top_n} to {top_applied} so that {n_groups} groups fit in {MAX_ROWS} rows'
        if missing:
            out['codes_not_in_graph'] = missing
        out['notes'] = self._notes(f, dims)
        if ties:
            out['notes'].append('Rows tied with the last row within top_n are also shown, so a cut never splits equal counts.')
        if 'pt' in det:
            out['notes'].append('common_wordings are the most frequent writer wordings coded to the PT (all records), to show what '
                                'the MedDRA term stands for in these data.')
        out['_counted'] = {k: sorted(v) for k, v in f['_counted'].items()}
        number_rows(out, 'summary', 'rows')
        if popd:
            out['clear_differences'] = clear_differences(out['rows'], popd)
            out['notes'].append('clear_differences lists, for the same platform, concept and detail values, the population groups '
                                'whose 95% intervals do not overlap (neither group sparse); every other pair is similar or unclear.')
        return out

    def get_evidence(self, **a):
        f, where, pop, params, target_ids, resolved, missing = self._filters(a)
        se = f['target'] == 'side_effect'
        limit = max(1, min(int(a.get('limit') or 5), 10))
        cond = where + pop
        if target_ids is not None:
            cond.append('n.id IN $target_ids')
        wording = (a.get('wording') or '').strip() or None
        if wording:
            rel, _ = SCHEMA[f['target']]
            params['wording_ids'] = self.run(
                "CALL db.index.fulltext.queryRelationships('verbatims', $q) YIELD relationship AS r "
                f"WHERE type(r) = '{rel}' RETURN collect(r.id) AS ids", q=lucene_query(wording)[0])[0]['ids']
            cond.append('e.id IN $wording_ids')
        base = f"{self._match(f)} WHERE {' AND '.join(cond)}"
        counts = {r['platform']: r for r in self.run(
            f"{base} RETURN e.platform AS platform, count(e) AS edges, count(DISTINCT e.record_key) AS records", **params)}
        queues = {}
        for p in f['platforms']:
            cand = self.run(f"{base} AND e.platform = $p {PATH} RETURN e {{.*}} AS e, d.ingredient AS ingredient, d.brand AS brand, "
                            f"coalesce(n.pt, $u) AS pt, coalesce(n.pt_code, '{UNCODED}') AS pt_code, s.name AS soc "
                            "ORDER BY e.order_key LIMIT $cap", p=p, u=UNCODED_LABEL, cap=limit * 40, **params)
            q, seen = [], set()
            for r in cand:  # one statement per record, in sha256(edge id) order: deterministic, never chosen by the model
                if r['e']['record_key'] not in seen:
                    seen.add(r['e']['record_key'])
                    q.append(r)
            queues[p] = q
        chosen = []
        while len(chosen) < limit and any(queues.values()):
            for p in f['platforms']:
                if queues[p] and len(chosen) < limit:
                    chosen.append(queues[p].pop(0))
        rows, full, record_ids = [], {}, set()
        for r in chosen:
            e = r['e']
            spans = json.loads(e['evidence'])
            w_spans = json.loads(e['writer_evidence'])
            row = _row(edge_id=e['id'], platform=e['platform'], date=e.get('date'),
                       ingredient=r['ingredient'], brand=r['brand'], ingredient_basis=e.get('ingredient_basis'),
                       pt=r['pt'], pt_code=r['pt_code'], soc=r['soc'], wording=e['text'],
                       writer_age_group=e.get('writer_age_group') or NOT_STATED, writer_sex=e.get('writer_sex') or NOT_STATED)
            if se:
                row.update(dose=e.get('dose'), duration=e.get('duration'), intensity=e.get('intensity'),
                           impact=e.get('impact'), action_taken=e.get('action_taken') or [])
            row['evidence'] = [_clip(s['text'], 600, e['text']) for s in spans[:3]]
            if len(spans) > 3:
                row['evidence_spans_not_shown'] = len(spans) - 3
            if e.get('writer_age_group') or e.get('writer_sex'):
                row['writer_evidence'] = [_clip(s['text'], 300) for s in w_spans[:2]]
            full[e['id']] = [s['text'] for s in spans + w_spans]
            record_ids.add(e['record_id'])
            rows.append(row)
        summary = [_row(platform=p, matching_statements=counts.get(p, {}).get('edges', 0),
                        matching_records=counts.get(p, {}).get('records', 0)) for p in f['platforms']]
        out = collections.OrderedDict(tool='get_evidence', filters={**_public_filters(f), 'wording': wording, 'limit': limit})
        if resolved:
            out['resolved_concepts'] = resolved
        out['summary'] = summary
        out['rows'] = rows
        if missing:
            out['codes_not_in_graph'] = missing
        out['notes'] = ['Examples are drawn in a fixed pseudo-random order (sha256 of the statement id), one statement per record, '
                        'platforms in turn; they illustrate and are not representative counts.',
                        'evidence holds the cited source sentences; WebMD profile values appear as profile fields. Quote only '
                        'from these sentences and give the edge_id in the quotes list; never write ids in the answer.'] + BASE_NOTES[:1]
        out['_full_evidence'] = full
        out['_record_ids'] = sorted(record_ids)
        out['_counted'] = {k: sorted(v) for k, v in f['_counted'].items()}
        number_rows(out, 'summary', 'rows')
        if popd:
            out['clear_differences'] = clear_differences(out['rows'], popd)
            out['notes'].append('clear_differences lists, for the same platform, concept and detail values, the population groups '
                                'whose 95% intervals do not overlap (neither group sparse); every other pair is similar or unclear.')
        return out

    # ------------------------------------------------------------------ FAERS
    def _scope(self, scope):
        """FAERS and patient patterns for one scope string: (anchored MATCH of the scope's primary reports, the same
        condition as a predicate on r, pattern of the stratum relationships y carrying action codes, patient drug
        condition, params, window start). The all-drug scope is the primary analysis set and takes the action codes of
        the report's ingredient strata, as the frozen FAERS tables do."""
        if scope == 'all':  # as the frozen RQ2 tables: every record from the first approval of any brand
            return ('MATCH (r:Report) WHERE r.in_primary_analysis', 'r.in_primary_analysis', '(r)-[y:REPORTS_INGREDIENT]->()',
                    'true', {}, self.all_start)
        kind, name = scope.split(':', 1)
        if kind == 'ingredient' and name in INGREDIENTS:
            pattern, dcond, start = ('(r)-[{v}:REPORTS_INGREDIENT]->(:Ingredient {{name: $scope_name}})',
                                     'd.ingredient = $scope_name', self.ingredient_start[name])
        elif kind == 'brand' and name in BRANDS:
            pattern, dcond, start = ("(r)-[{v}:REPORTS_DRUG {{level: 'brand'}}]->(:Drug {{brand: $scope_name}})",
                                     'd.brand = $scope_name', self.brand_start[name])
        elif kind == 'unspecified_brand' and name in INGREDIENTS:
            pattern, dcond, start = ("(r)-[{v}:REPORTS_DRUG {{level: 'unspecified'}}]->(:Drug {{ingredient: $scope_name}})",
                                     f"d.ingredient = $scope_name AND d.brand = '{UNSPECIFIED}'", self.ingredient_start[name])
        else:
            raise QueryError(f'unknown scope {scope!r}')
        anchored = pattern.format(v='x').replace('(r)', '(r:Report)')
        return (f'MATCH {anchored} WHERE x.in_primary', f"EXISTS {{ MATCH {pattern.format(v='x')} WHERE x.in_primary }}",
                pattern.format(v='y'), dcond, {'scope_name': name}, start)

    def compare_faers(self, targets=None, scopes=(), sexes=None, age_groups=None, rank_by='faers', top_n=10,
                      include_outcomes=False):
        scopes = list(dict.fromkeys(scopes or []))
        if not scopes:
            raise QueryError('scopes must name at least one drug scope')
        if sexes is not None and (not sexes or set(sexes) - {'female', 'male'}):
            raise QueryError("sexes must be null or a non-empty subset of ['female', 'male']")
        age_groups = _check_list('age_groups', age_groups, AGE_GROUPS)
        top_n = max(1, min(int(top_n or 10), 30))
        concepts, resolved, missing = [], [], []
        excluded_nonclinical = []
        counted = set()
        if targets is not None:
            sets, missing = self.resolve_sets('side_effect', [dict(c) for c in targets])
            for st in sets:
                if not st['ids']:
                    continue
                clin = [r['id'] for r in self.run('MATCH (n:SideEffect) WHERE n.id IN $ids AND n.nonclinical_group IS NULL '
                                                   'RETURN n.id AS id', ids=st['ids'])]
                if not clin:
                    excluded_nonclinical.append(st['concept'])
                    continue
                concepts.append({'key': st['concept'], 'ids': clin})
                resolved.append({'concept': st['concept'], 'members': st['members'], 'clinical_pts': len(clin)})
                counted |= self.pt_codes('side_effect', clin)
        rfil, efil, fparams = [], [], {}
        if sexes is not None:
            rfil.append('r.sex IN $sexes')
            efil.append('e.writer_sex IN $sexes')
            fparams['sexes'] = list(sexes)
        if age_groups is not None:
            rfil.append('r.age_group IN $age_groups')
            efil.append('e.writer_age_group IN $age_groups')
            fparams['age_groups'] = age_groups
        rf = ''.join(f' AND {x}' for x in rfil)
        ef = ''.join(f' AND {x}' for x in efil)
        epred = ' AND '.join(efil) or 'true'
        cc = [{'key': c['key'], 'ids': c['ids']} for c in concepts]
        summary, rows, outcomes, ties = [], [], [], 0
        for scope in scopes:
            rmatch, rpred, ypat, dcond, sp, start = self._scope(scope)
            p = {**fparams, **sp, 'start': start, 'end': WINDOW_END}
            fr = self.run(f"{rmatch} WITH DISTINCT r RETURN count(r) AS reports, "
                          "count(CASE WHEN r.sex IN ['female', 'male'] THEN 1 END) AS sex_known, "
                          "count(CASE WHEN r.age_group IS NOT NULL AND r.age_group <> 'under 13' THEN 1 END) AS age_known", **p)[0]
            fden = self.run(f'{rmatch}{rf} AND r.has_clinical_pt WITH DISTINCT r RETURN count(r) AS c', **p)[0]['c']
            ecore = (f'MATCH (d:Drug)-[e:SIDE_EFFECT]->(n:SideEffect) WHERE {dcond} '
                     'AND e.date >= $start AND e.date <= $end AND n.nonclinical_group IS NULL')
            pstat = {r['platform']: r for r in self.run(
                f'{ecore} RETURN e.platform AS platform, count(DISTINCT CASE WHEN {epred} THEN e.record_key END) AS den, '
                'count(DISTINCT CASE WHEN e.writer_sex IS NULL THEN e.record_key END) AS sex_missing, '
                'count(DISTINCT CASE WHEN e.writer_age_group IS NULL THEN e.record_key END) AS age_missing', **p)}
            pden = {plat: pstat.get(plat, {}).get('den', 0) for plat in PLATFORMS}
            summary.append(_row(
                scope=scope, faers_reports_in_scope=fr['reports'],
                faers_excluded_missing_sex=fr['reports'] - fr['sex_known'] if sexes is not None else None,
                faers_excluded_missing_age=fr['reports'] - fr['age_known'] if age_groups is not None else None,
                faers_denominator=fden, reddit_denominator=pden['reddit'], webmd_denominator=pden['webmd'],
                reddit_excluded_missing_sex=pstat.get('reddit', {}).get('sex_missing', 0) if sexes is not None else None,
                webmd_excluded_missing_sex=pstat.get('webmd', {}).get('sex_missing', 0) if sexes is not None else None,
                reddit_excluded_missing_age=pstat.get('reddit', {}).get('age_missing', 0) if age_groups is not None else None,
                webmd_excluded_missing_age=pstat.get('webmd', {}).get('age_missing', 0) if age_groups is not None else None,
                record_window=f'{start} to {WINDOW_END}'))

            def add(name, code, fn, rn, wn):
                row = _row(scope=scope, concept=name)
                if code:
                    row['pt_code'] = code
                for src, k, d in (('faers', fn, fden), ('reddit', rn, pden['reddit']), ('webmd', wn, pden['webmd'])):
                    lo, hi = wilson(k, d)
                    row.update({f'{src}_' + ('reports' if src == 'faers' else 'records'): k, f'{src}_denominator': d,
                                f'{src}_percent': percent(k, d), f'{src}_ci95_low': lo, f'{src}_ci95_high': hi})
                row.update(reddit_small=pden['reddit'] < SMALL_BELOW, webmd_small=pden['webmd'] < SMALL_BELOW)
                for x, y in (('faers', 'reddit'), ('faers', 'webmd'), ('reddit', 'webmd')):
                    row[f'{x}_vs_{y}'] = interval_order(row, x, y)
                rows.append(row)

            if targets is not None:
                fn = {r['key']: r['c'] for r in self.run(
                    f'UNWIND $cc AS cc MATCH (c:SideEffect)<-[:REACTION]-(r:Report) WHERE c.id IN cc.ids AND {rpred}{rf} '
                    'RETURN cc.key AS key, count(DISTINCT r) AS c', cc=cc, **p)}
                pn = {(r['key'], r['platform']): r['c'] for r in self.run(
                    f'UNWIND $cc AS cc MATCH (n:SideEffect)<-[e:SIDE_EFFECT]-(d:Drug) WHERE n.id IN cc.ids AND {dcond} '
                    f'AND e.date >= $start AND e.date <= $end{ef} '
                    'RETURN cc.key AS key, e.platform AS platform, count(DISTINCT e.record_key) AS c', cc=cc, **p)}
                for c in concepts:
                    k = c['key']
                    add(k, None, fn.get(k, 0), pn.get((k, 'reddit'), 0), pn.get((k, 'webmd'), 0))
            else:
                fcounts = {r['code']: r for r in self.run(
                    f'{rmatch}{rf} WITH DISTINCT r MATCH (r)-[:REACTION]->(c:SideEffect) WHERE c.nonclinical_group IS NULL '
                    'RETURN c.pt_code AS code, c.pt AS name, count(DISTINCT r) AS n', **p)}
                pcounts = {plat: {} for plat in PLATFORMS}
                for r in self.run(f'{ecore}{ef} AND n.pt_code IS NOT NULL RETURN e.platform AS platform, n.pt_code AS code, '
                                  'n.pt AS name, count(DISTINCT e.record_key) AS n', **p):
                    pcounts[r['platform']][r['code']] = r
                source = fcounts if rank_by == 'faers' else pcounts[rank_by]
                ranked = sorted(source.values(), key=lambda r: (-r['n'], r['name']))
                cut = ranked[top_n - 1]['n'] if len(ranked) >= top_n else None
                chosen = ranked[:top_n] + [r for r in ranked[top_n:top_n + MAX_TIES] if r['n'] == cut]
                ties += len(chosen) - min(top_n, len(ranked))
                for r in chosen:
                    code = r['code']
                    add(r['name'], code, fcounts.get(code, {}).get('n', 0), pcounts['reddit'].get(code, {}).get('n', 0),
                        pcounts['webmd'].get(code, {}).get('n', 0))
            if include_outcomes:
                o = self.run(f"{rmatch}{rf} WITH DISTINCT r OPTIONAL MATCH {ypat} WHERE y.in_primary "
                             "WITH r, collect(y.actions) AS acts WITH r, reduce(a = [], l IN acts | a + coalesce(l, [])) AS acts "
                             "RETURN count(r) AS reports, count(CASE WHEN r.hospitalization THEN 1 END) AS hosp, "
                             "count(CASE WHEN any(c IN acts WHERE c IN ['1', '2', '3', '4']) THEN 1 END) AS known, "
                             "count(CASE WHEN '1' IN acts THEN 1 END) AS withdrawn, count(CASE WHEN '2' IN acts THEN 1 END) AS reduced",
                             **p)[0]
                po = {r['platform']: r for r in self.run(
                    f"MATCH (d:Drug)-[e:SIDE_EFFECT]->(n:SideEffect) WHERE {dcond} AND e.date >= $start AND e.date <= $end{ef} "
                    "RETURN e.platform AS platform, count(DISTINCT e.record_key) AS records, "
                    "count(DISTINCT CASE WHEN e.impact = 'hospitalization' THEN e.record_key END) AS hosp, "
                    "count(DISTINCT CASE WHEN 'stopped' IN coalesce(e.action_taken, []) THEN e.record_key END) AS stopped, "
                    "count(DISTINCT CASE WHEN 'dose_reduced' IN coalesce(e.action_taken, []) THEN e.record_key END) AS reduced",
                    **p)}
                z = {'records': 0, 'hosp': 0, 'stopped': 0, 'reduced': 0}
                rr, ww = po.get('reddit', z), po.get('webmd', z)
                outcomes.append(_row(scope=scope, faers_reports=o['reports'], faers_hospitalization=o['hosp'],
                                     faers_action_known_reports=o['known'], faers_drug_withdrawn=o['withdrawn'],
                                     faers_dose_reduced=o['reduced'],
                                     reddit_records=rr['records'], reddit_hospitalization=rr['hosp'],
                                     reddit_stopped=rr['stopped'], reddit_dose_reduced=rr['reduced'],
                                     webmd_records=ww['records'], webmd_hospitalization=ww['hosp'],
                                     webmd_stopped=ww['stopped'], webmd_dose_reduced=ww['reduced']))
        out = collections.OrderedDict(tool='compare_faers', status=FAERS_STATUS, scopes=scopes, sexes=sexes,
                                      age_groups=age_groups, rank_by=None if targets is not None else rank_by)
        if resolved:
            out['resolved_concepts'] = resolved
        out['summary'] = summary
        out['rows'] = rows
        if include_outcomes:
            out['outcomes'] = outcomes
        if missing:
            out['codes_not_in_graph'] = missing
        if excluded_nonclinical:
            out['excluded_nonclinical_concepts'] = excluded_nonclinical
        out['notes'] = [
            FAERS_STATUS + '.',
            'faers_reports = primary-configuration FAERS reports in the scope (suspect or interacting role, validated drug identity, '
            'window from the FDA approval date to 2026-06-30) that list a PT of the concept; faers_denominator = reports in the scope '
            'with at least one clinical-event PT. The Reddit and WebMD columns count records (a concept once per record) in the '
            'same window (record_window); their denominators are records with at least one clinical-event side effect (uncoded '
            'ones included). A concept row counts a report or record once if it lists any member PT.',
            'Only clinical-event PTs are compared: product issues, medication errors, off-label use and misuse, therapeutic responses '
            'and interactions, overdoses and underdoses, and pregnancy exposures are excluded on both sides (official MedDRA groups).',
            'FAERS is spontaneous reporting by patients, clinicians and manufacturers; a report names suspect drugs but does not '
            'establish that the drug caused the reaction. Reddit and WebMD shares come from what writers chose to post, and the '
            'writers are not verified patients. Describe the sources side by side; never compute ratios or differences between '
            'them or call any of them a rate.',
            'FAERS age is the age at reaction onset; the Reddit or WebMD age is the writer\'s stated age when writing. FAERS has no '
            '"other" sex.',
            'Scopes: ingredient scopes cover every product of the ingredient (Reddit includes community-assigned ingredients); '
            'brand scopes need the brand named; unspecified_brand covers reports or records that name only the ingredient; for '
            'all, the Reddit and WebMD window starts at the first approval of any brand.',
            f'reddit_small / webmd_small = denominator below {SMALL_BELOW}: say in words that the group is small.',
            '*_ci95_low/high: approximate 95% Wilson intervals of each percent; x_vs_y says whether source x\'s share is higher, '
            'lower or similar (overlapping intervals) compared with source y. Describe comparisons with these words and never '
            'compute a ratio or difference.']
        if include_outcomes:
            out['notes'].append('outcomes: FAERS hospitalization = reports with the hospitalization seriousness criterion; action '
                                'codes come from the study-drug entries (withdrawn, dose reduced; known = any of withdrawn, reduced, '
                                'increased, not changed). Reddit and WebMD hospitalization = records with a side effect whose impact '
                                'is hospitalization; stopped and dose_reduced come from action_taken. Their outcome counts use all '
                                'side effects in the window (no clinical-event restriction).')
        if ties:
            out['rows_kept_for_ties'] = ties
            out['notes'].append('PTs tied with the last ranked PT are also shown, so a cut never splits equal counts.')
        out['_counted'] = {'side_effect': sorted(counted), 'purpose': []}
        return number_rows(out, 'summary', 'rows', 'outcomes')
