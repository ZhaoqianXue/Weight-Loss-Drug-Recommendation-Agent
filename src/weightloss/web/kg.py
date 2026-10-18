"""Knowledge-graph views for the website.

Every count comes from the agent's graph tools (weightloss.agent.tools.KnowledgeGraph), so the site and the agent apply
the same counting rules (project document Section 3.6): records, not patients; a concept once per record; Reddit and
WebMD never pooled; brand values only where a brand is named. This module only selects and reshapes tool outputs.
The graph is frozen while the server runs, so results are cached in memory and on disk under a directory named after the
loaded graph's manifest (a reload of the graph starts a new cache).
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from weightloss.agent import VERSION as AGENT_VERSION
from weightloss.agent.agent import EFFORT as AGENT_EFFORT, MODEL as AGENT_MODEL
from weightloss.agent import tools as T
from weightloss.graph import neo4j_db
from weightloss.graph.build import AGE_GROUPS, BRAND_INGREDIENT, UNSPECIFIED

PLATFORMS = T.PLATFORMS
INGREDIENTS = T.INGREDIENTS
BRANDS = T.BRANDS
TARGETS = T.TARGETS
SEXES = T.SEXES
NOT_STATED = T.NOT_STATED
FILTER_DEFAULTS = dict(ingredients=None, brands=None, named_drugs_only=False, age_groups=None, sexes=None,
                       require_purposes=None, require_side_effects=None, targets=None, intensities=None, impacts=None,
                       actions=None)


MAX_TOP_N = 10  # largest "terms per drug" the page offers


def _cut(view, top_n):
    """The graph view with each drug's first top_n PTs per role (links are stored largest first within a drug)."""
    seen, links = {}, []
    for link in view['links']:
        key = (link['source'], link['kind'])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= top_n:
            links.append(link)
    used = {link['source'] for link in links} | {link['target'] for link in links}
    return {**view, 'top_n': top_n, 'links': links, 'nodes': [n for n in view['nodes'] if n['id'] in used]}


class GraphUnavailable(RuntimeError):
    """The local Neo4j instance is not running."""


def _notes(*outputs):
    return list(dict.fromkeys(note for out in outputs for note in out.get('notes') or ()))


def _pt(code):
    return {'level': 'pt', 'code': code}


def statement(row):
    """An example statement with its record id, which the agent's rows leave out (answers must not show ids, the page
    may). Statement ids are '<platform>:<record id>:d<drug group>:<p|s><index>' (weightloss.graph.build)."""
    if row is None:
        return None
    parts = (row.get('edge_id') or '').split(':')
    return {**row, 'record_id': ':'.join(parts[1:-2]) if len(parts) >= 4 else None}


class KGService:
    """Read-only, cached access to the knowledge graph for the website."""

    def __init__(self, graph=None, connect: Optional[Callable[[], object]] = None, cache_dir: Optional[Path] = None):
        self._graph = graph
        self._connect = connect or self._default_connect
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._key_locks: dict = {}
        self._cache_root = cache_dir

    @staticmethod
    def _default_connect():
        if not neo4j_db.is_running():
            raise GraphUnavailable('The knowledge graph database is not running. Start it with '
                                   '`python -m weightloss.graph.neo4j_db start`.')
        return T.KnowledgeGraph()

    @property
    def graph(self):
        with self._lock:
            if self._graph is None:
                self._graph = self._connect()
            return self._graph

    def close(self):
        with self._lock:
            if self._graph is not None and hasattr(self._graph, 'close'):
                self._graph.close()
            self._graph = None

    def _disk_dir(self):
        if self._cache_root is None:
            return None
        # Keyed by the loaded graph and the agent version, since either changes what the tools return.
        manifest = getattr(self.graph, 'manifest', None) or {}
        stamp = hashlib.sha256(json.dumps([manifest, AGENT_VERSION], sort_keys=True).encode()).hexdigest()[:16]
        return Path(self._cache_root) / stamp

    def _cached(self, key, compute):
        """One computation per key: concurrent callers wait for it; results persist across restarts."""
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if key in self._cache:
                return self._cache[key]
            directory = self._disk_dir()
            path = directory / (hashlib.sha256(repr(key).encode()).hexdigest()[:24] + '.json') if directory else None
            if path is not None and path.is_file():
                value = json.loads(path.read_text())
            else:
                value = compute()
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(value))
                    tmp.replace(path)
            self._cache[key] = value
            return value

    def _query(self, target, platforms, group_by, top_n, **filters):
        args = {**FILTER_DEFAULTS, **filters}
        return self.graph.query_graph(target=target, platforms=list(platforms), group_by=list(group_by), top_n=top_n, **args)

    # ------------------------------------------------------------------ overview
    def overview(self):
        def compute():
            g = self.graph
            out = g.overview()
            manifest = getattr(g, 'manifest', None) or {}
            build = manifest.get('kg_build') or {}
            extraction = build.get('extraction') or {}
            return {
                'platforms': [{k: r[k] for k in ('platform', 'records_processed', 'records_with_side_effect',
                                                 'side_effect_statements', 'records_with_purpose', 'purpose_statements')}
                              for r in out['rows']],
                'faers_reports': out['faers'][0]['reports_primary_configuration'],
                'faers_status': out['faers'][0]['status'],
                'versions': {
                    'kg': out.get('graph') or manifest.get('kg_version'),
                    'kg_built': build.get('built_utc'),
                    'graph_loaded': out.get('loaded_utc'),
                    'meddra': out.get('meddra_version'),
                    'extraction_model': extraction.get('model'),
                    'extraction_prompt': extraction.get('prompt_version'),
                    'agent': AGENT_VERSION,
                    'agent_model': f'{AGENT_MODEL} ({AGENT_EFFORT} reasoning)',
                },
                'ingredients': list(INGREDIENTS),
                'brands': [{'brand': b, 'ingredient': BRAND_INGREDIENT[b]} for b in BRANDS],
                'age_groups': list(AGE_GROUPS),
                'sexes': list(SEXES),
                'notes': _notes(out) + [n for n in T.BASE_NOTES if n not in _notes(out)],
            }
        return self._cached(('overview',), compute)

    def socs(self):
        """MedDRA SOCs that hold side-effect or purpose statements, by records on either platform."""
        def compute():
            found = {}
            for target in TARGETS:
                out = self._query(target, PLATFORMS, ['soc'], 50)
                for r in out['rows']:
                    if r['soc_code'] == T.UNCODED:
                        continue
                    entry = found.setdefault(r['soc_code'], {'code': r['soc_code'], 'name': r['soc'], 'side_effect': 0, 'purpose': 0})
                    entry[target] = max(entry[target], r['records'])
            return sorted(found.values(), key=lambda s: -(s['side_effect'] + s['purpose']))
        return self._cached(('socs',), compute)

    # ------------------------------------------------------------------ graph explorer
    def graph_view(self, platform, level, show, drugs=None, age_groups=None, sexes=None, soc=None, top_n=4):
        """Drug -> purpose / side-effect PT edges for one platform: each drug's top PTs by records.

        Each role is computed once per filter combination at the largest top_n the page offers (the two roles in
        parallel) and cached; the view for "both" merges them and each smaller top_n is cut from the result, so moving
        the "terms per drug" slider or switching roles never waits for the database.
        """
        targets = list(TARGETS) if show == 'both' else [show]
        args = (platform, level, drugs, age_groups, sexes, soc)
        if len(targets) == 1:
            parts = [self._role_part(targets[0], *args)]
        else:
            with ThreadPoolExecutor(max_workers=len(targets)) as pool:
                parts = list(pool.map(lambda target: self._role_part(target, *args), targets))
        nodes, links, summaries, notes, hidden = {}, [], [], [], {}
        for target, part in zip(targets, parts):
            for node in part['nodes']:
                if node['id'] in nodes:  # a drug shows a record count per role
                    nodes[node['id']]['records'].update(node['records'])
                else:
                    nodes[node['id']] = {**node, 'records': dict(node['records'])}
            links += part['links']
            summaries += part['summary']
            notes += part['notes']
            hidden[target] = part['hidden']
        order = {f'drug:{d}': i for i, d in enumerate(INGREDIENTS if level == 'ingredient' else BRANDS)}
        node_list = sorted(nodes.values(), key=lambda n: (n['kind'] != 'drug', order.get(n['id'], 99), n['label']))
        base = {'platform': platform, 'level': level, 'show': show, 'top_n': MAX_TOP_N, 'nodes': node_list, 'links': links,
                'summary': summaries, 'rows_not_shown': hidden, 'notes': list(dict.fromkeys(notes))}
        return _cut(base, min(top_n, MAX_TOP_N))

    def _role_part(self, target, platform, level, drugs, age_groups, sexes, soc):
        frozen = lambda values: None if values is None else tuple(values)  # noqa: E731 - None (all) differs from () (none)
        key = ('graph-role', target, platform, level, frozen(drugs), frozen(age_groups), frozen(sexes), soc, MAX_TOP_N)
        return self._cached(key, lambda: self._compute_role(target, platform, level, drugs, age_groups, sexes, soc, MAX_TOP_N))

    def _compute_role(self, target, platform, level, drugs, age_groups, sexes, soc, top_n):
        """One role's drug -> PT links with each drug's top_n coded PTs, node totals over the shown drugs, and the summary."""
        filters = {'age_groups': None if age_groups is None else list(age_groups), 'sexes': None if sexes is None else list(sexes)}
        drug_key = 'ingredients' if level == 'ingredient' else 'brands'
        named = list(INGREDIENTS if level == 'ingredient' else BRANDS)
        if drugs is not None:
            filters[drug_key] = list(drugs)
        # Node totals count only the drugs a reader can see: every named drug of the level, never "unspecified".
        shown_drugs = {**filters, drug_key: filters.get(drug_key, named)}
        soc_filter = {'targets': [{'level': 'soc', 'code': soc}]} if soc else {}
        # Two queries at once: each drug's top rows (one extra, because the uncoded bucket can be among them and is not a
        # PT), and records per PT over the shown drugs for the 50 largest PTs, which covers nearly every PT drawn.
        with ThreadPoolExecutor(max_workers=2) as pool:
            main = pool.submit(self._query, target, [platform], [level, 'pt', 'soc'], min(top_n + 1, 50), **filters, **soc_filter)
            wide = pool.submit(self._query, target, [platform], ['pt'], 50, **shown_drugs, **soc_filter)
            out, totals_wide = main.result(), wide.result()
        summary = [{'target': target, **{k: s.get(k) for k in (
            'universe_records', 'records_before_age_sex_filter', 'excluded_missing_age', 'excluded_missing_sex',
            'universe_records_with_uncoded_mention', 'universe_records_with_community_ingredient')}} for s in out.get('summary') or []]
        nodes, links, kept, codes = {}, [], {}, []
        for r in out['rows']:  # rows come largest first within each drug
            drug = r[level]
            if drug == UNSPECIFIED or r['pt_code'] == T.UNCODED:
                continue
            kept[drug] = kept.get(drug, 0) + 1
            if kept[drug] > top_n:
                continue
            drug_id = f'drug:{drug}'
            node = nodes.setdefault(drug_id, {'id': drug_id, 'kind': 'drug', 'label': drug, 'code': None, 'soc': None,
                                              'ingredient': BRAND_INGREDIENT.get(drug, drug), 'records': {}})
            node['records'][target] = r['denominator']
            pt_id = f'{target}:{r["pt_code"]}'
            if pt_id not in nodes:
                nodes[pt_id] = {'id': pt_id, 'kind': target, 'label': r['pt'], 'code': r['pt_code'], 'soc': r.get('soc'),
                                'ingredient': None, 'records': {}}
                codes.append(r['pt_code'])
            links.append({'source': drug_id, 'target': pt_id, 'kind': target, 'records': r['records'],
                          'denominator': r['denominator'], 'percent': r['percent'], 'sparse': r['sparse'],
                          'small': r['small']})
        # Records per PT across the shown drugs (a record naming two drugs counts once); PTs outside the 50 largest get
        # their own query.
        rows = [r for r in totals_wide['rows'] if r['pt_code'] in codes]
        missing = [c for c in codes if c not in {r['pt_code'] for r in rows}]
        if missing:
            rows += self._query(target, [platform], ['pt'], 50, **shown_drugs, targets=[_pt(c) for c in missing])['rows']
        for r in rows:
            node = nodes.get(f'{target}:{r["pt_code"]}')
            if node is not None:
                node['records'][target] = r['records']
                node['denominator'] = r['denominator']
        return {'nodes': list(nodes.values()), 'links': links, 'summary': summary,
                'hidden': (out.get('rows_not_shown') or {}).get(platform, 0), 'notes': _notes(out)}

    # ------------------------------------------------------------------ patterns
    def patterns(self):
        """Side-by-side distributions for both platforms (never pooled)."""
        def compute():
            def rows(out, label_key, code_key=None):
                return [{'platform': r['platform'], 'label': r[label_key], 'code': r.get(code_key) if code_key else None,
                         'records': r['records'], 'denominator': r['denominator'], 'percent': r['percent'], 'sparse': r['sparse'],
                         'small': r['small']}
                        for r in out['rows'] if r.get(code_key or label_key) not in (T.UNCODED,)]
            se_pt = self._query('side_effect', PLATFORMS, ['pt'], 10)
            pu_pt = self._query('purpose', PLATFORMS, ['pt'], 10)
            se_soc = self._query('side_effect', PLATFORMS, ['soc'], 10)
            drugs = self._query('side_effect', PLATFORMS, ['ingredient'], 10)
            brands = self._query('side_effect', PLATFORMS, ['brand'], 10)
            population = self._query('side_effect', PLATFORMS, ['age_group', 'sex'], 50)
            impacts = self._query('side_effect', PLATFORMS, ['impact'], 10)
            return {
                'sections': [
                    {'key': 'side_effects', 'title': 'Side effects', 'unit': 'records with a side effect',
                     'definition': se_pt.get('denominator_definition'), 'rows': rows(se_pt, 'pt', 'pt_code')},
                    {'key': 'purposes', 'title': 'Purposes', 'unit': 'records with a purpose',
                     'definition': pu_pt.get('denominator_definition'), 'rows': rows(pu_pt, 'pt', 'pt_code')},
                    {'key': 'socs', 'title': 'Side effects by MedDRA SOC', 'unit': 'records with a side effect',
                     'definition': se_soc.get('denominator_definition'), 'rows': rows(se_soc, 'soc', 'soc_code')},
                    {'key': 'ingredients', 'title': 'Ingredients', 'unit': 'records with a side effect',
                     'definition': drugs.get('denominator_definition'), 'rows': rows(drugs, 'ingredient')},
                    {'key': 'brands', 'title': 'Brands named', 'unit': 'records with a side effect',
                     'definition': brands.get('denominator_definition'), 'rows': rows(brands, 'brand')},
                    {'key': 'impacts', 'title': 'Reported impact', 'unit': 'records with a side effect',
                     'definition': impacts.get('denominator_definition'), 'rows': rows(impacts, 'impact')},
                ],
                'population': [{'platform': r['platform'], 'age_group': r['age_group'], 'sex': r['sex'],
                                'records': r['records'], 'denominator': r['denominator'], 'percent': r['percent']}
                               for r in population['rows']],
                # Each record has one writer, so the age x sex cells partition the records of a platform.
                'population_summary': [
                    {'platform': p,
                     'records': sum(r['records'] for r in population['rows'] if r['platform'] == p),
                     'age_and_sex_stated': sum(r['records'] for r in population['rows']
                                               if r['platform'] == p and NOT_STATED not in (r['age_group'], r['sex'])),
                     'neither_stated': sum(r['records'] for r in population['rows']
                                           if r['platform'] == p and r['age_group'] == NOT_STATED and r['sex'] == NOT_STATED)}
                    for p in PLATFORMS],
                'notes': _notes(se_pt, pu_pt, population),
            }
        return self._cached(('patterns',), compute)

    # ------------------------------------------------------------------ FAERS
    def faers(self, scope='all', rank_by='faers', top_n=12, sex=None, age_group=None):
        key = ('faers', scope, rank_by, top_n, sex, age_group)

        def compute():
            out = self.graph.compare_faers(targets=None, scopes=[scope], sexes=[sex] if sex else None,
                                           age_groups=[age_group] if age_group else None, rank_by=rank_by, top_n=top_n,
                                           include_outcomes=False)
            keep = ('concept', 'faers_reports', 'faers_denominator', 'faers_percent', 'reddit_records', 'reddit_denominator',
                    'reddit_percent', 'webmd_records', 'webmd_denominator', 'webmd_percent', 'reddit_small', 'webmd_small')
            summary = (out.get('summary') or [{}])[0]
            return {'scope': scope, 'rank_by': rank_by, 'sex': sex, 'age_group': age_group, 'status': out.get('status'),
                    'window': summary.get('record_window'), 'summary': summary,
                    'rows': [{**{k: r.get(k) for k in keep}, 'code': r.get('pt_code')} for r in out['rows']],
                    'notes': _notes(out)}
        return self._cached(key, compute)

    # ------------------------------------------------------------------ evidence
    def evidence(self, target, platform, level=None, drug=None, code=None, limit=6, age_groups=None, sexes=None, soc=None):
        """Example statements for a node or link, under the explorer's drug level, population filters and SOC."""
        frozen = lambda values: None if values is None else tuple(values)  # noqa: E731
        key = ('evidence', target, platform, level, drug, code, limit, frozen(age_groups), frozen(sexes), soc)

        def compute():
            filters = {'age_groups': None if age_groups is None else list(age_groups), 'sexes': None if sexes is None else list(sexes)}
            if level in ('ingredient', 'brand'):
                # A PT node in the brand view stands for named brands only, so its examples do too.
                filters['ingredients' if level == 'ingredient' else 'brands'] = (
                    [drug] if drug else list(INGREDIENTS if level == 'ingredient' else BRANDS))
            if code:
                filters['targets'] = [_pt(code)]
            elif soc:
                filters['targets'] = [{'level': 'soc', 'code': soc}]
            out = self.graph.get_evidence(target=target, platforms=[platform], wording=None, limit=limit,
                                          **{**FILTER_DEFAULTS, **filters})
            keep = ('edge_id', 'platform', 'date', 'ingredient', 'brand', 'ingredient_basis', 'pt', 'pt_code', 'soc',
                    'wording', 'writer_age_group', 'writer_sex', 'dose', 'duration', 'intensity', 'impact', 'action_taken',
                    'evidence', 'writer_evidence')
            match = next((s for s in out.get('summary') or [] if s.get('platform') == platform), {})
            return {'target': target, 'platform': platform, 'drug': drug, 'code': code,
                    'matching_records': match.get('matching_records', 0),
                    'statements': [statement({k: r.get(k) for k in keep}) for r in out['rows']], 'notes': _notes(out)}
        return self._cached(key, compute)

    # ------------------------------------------------------------------ warm-up
    def warm(self):
        """Compute the default views first, then every view one click away, so the page never waits on them."""
        self.overview()
        self.socs()
        self.graph_view('reddit', 'brand', 'both')
        self.patterns()
        self.faers()
        for platform in PLATFORMS:
            for level in ('brand', 'ingredient'):
                self.graph_view(platform, level, 'both')
        scopes = ['all'] + [f'ingredient:{i}' for i in INGREDIENTS] + [f'brand:{b}' for b in BRANDS]
        for scope in scopes:
            for rank_by in ('faers', 'reddit', 'webmd'):
                self.faers(scope, rank_by)
            for sex in ('female', 'male'):
                self.faers(scope, 'faers', sex=sex)
        # Single-choice graph filters: each SOC, age group, sex, brand and ingredient on its own, for both sources.
        for platform in PLATFORMS:
            for level in ('brand', 'ingredient'):
                for soc in [s['code'] for s in self.socs()]:
                    self.graph_view(platform, level, 'both', soc=soc)
                for age in list(AGE_GROUPS) + [NOT_STATED]:
                    self.graph_view(platform, level, 'both', age_groups=[age])
                for sex in list(SEXES) + [NOT_STATED]:
                    self.graph_view(platform, level, 'both', sexes=[sex])
                for drug in (BRANDS if level == 'brand' else INGREDIENTS):
                    self.graph_view(platform, level, 'both', drugs=[drug])
