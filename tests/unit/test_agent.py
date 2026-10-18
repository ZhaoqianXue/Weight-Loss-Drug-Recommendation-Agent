"""Agent graph tools, verifier and loop.

Graph-tool expectations come from independent SQL over kg.sqlite and from the frozen FAERS validation tables
(results/fda-validation-20261017/frozen); those tests need the local Neo4j instance (python -m weightloss.graph.neo4j_db
start) and are skipped when it does not answer. The verifier and loop tests use a stub graph and a scripted fake client,
so no server and no API call are needed.
"""
import csv
import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from weightloss.agent import agent
from weightloss.agent.evaluate import check_calls
from weightloss.agent.tools import QueryError, lucene_query

ROOT = Path(__file__).resolve().parents[2]
KG = ROOT / '.cache/reextraction-20261016/kg/full-v7/kg.sqlite'
FROZEN = ROOT / 'results/fda-validation-20261017/frozen'
NAUSEA, VOMITING, T2DM, PANCREATITIS, GI_SOC = '10028813', '10047700', '10067585', '10033645', '10017947'


def neo4j_graph():
    try:
        from weightloss.agent.tools import KnowledgeGraph
        from weightloss.graph import neo4j_db
        if not (KG.exists() and FROZEN.exists() and neo4j_db.MANIFEST.exists()):
            return None
        return KnowledgeGraph(neo4j_db.connect(timeout=3))
    except Exception:
        return None


GRAPH = neo4j_graph()


def args(**kw):
    base = dict(target='side_effect', platforms=['reddit', 'webmd'], ingredients=None, brands=None, named_drugs_only=False,
                purposes_from_text_only=False, age_groups=None, sexes=None, require_purposes=None, require_side_effects=None, targets=None, intensities=None,
                impacts=None, actions=None)
    base.update(kw)
    return base


def pt(code, concept=None):
    return {'concept': concept or code, 'level': 'pt', 'code': code}


class PureTest(unittest.TestCase):
    def test_tool_schemas_are_strict(self):
        self.assertEqual([t['name'] for t in agent.TOOLS], ['link_terms', 'query_graph', 'get_evidence', 'compare_faers'])
        self.assertTrue(all(t['strict'] and t['parameters']['additionalProperties'] is False for t in agent.TOOLS))

    def test_lucene_query_keeps_only_words(self):
        q, words = lucene_query('Throwing up!! "the" (Ozempic) AND OR *')
        self.assertEqual(words, ['throwing', 'up', 'ozempic'])
        self.assertEqual(q, '(throwing OR throwing*) AND up AND (ozempic OR ozempic*)')
        with self.assertRaises(QueryError):
            lucene_query('the of ?')

    def test_intervals_and_clear_differences(self):
        from weightloss.agent.tools import clear_differences, interval_order, wilson
        self.assertEqual(wilson(0, 0), (None, None))
        lo, hi = wilson(59, 115)
        self.assertTrue(lo < 51.3 < hi)
        rows = [{'row': 1, 'platform': 'webmd', 'ingredient': 'semaglutide', 'percent': 46.6, 'ci95_low': 35.6, 'ci95_high': 57.9},
                {'row': 2, 'platform': 'webmd', 'ingredient': 'tirzepatide', 'percent': 27.3, 'ci95_low': 19.5, 'ci95_high': 36.8},
                {'row': 3, 'platform': 'webmd', 'ingredient': 'liraglutide', 'percent': 51.3, 'ci95_low': 42.3, 'ci95_high': 60.2},
                {'row': 4, 'platform': 'webmd', 'ingredient': 'dulaglutide', 'percent': 40.0, 'ci95_low': 20.0, 'ci95_high': 60.0,
                 'sparse': True}]
        self.assertEqual(clear_differences(rows, ['ingredient']),
                         [{'higher_row': 3, 'lower_row': 2, 'higher': 'liraglutide', 'lower': 'tirzepatide'}])
        row = {'faers_denominator': 100, 'faers_ci95_low': 10, 'faers_ci95_high': 20, 'reddit_denominator': 5,
               'reddit_ci95_low': 30, 'reddit_ci95_high': 40, 'webmd_denominator': 50, 'webmd_ci95_low': 25, 'webmd_ci95_high': 35}
        self.assertEqual((interval_order(row, 'faers', 'webmd'), interval_order(row, 'faers', 'reddit')), ('lower', 'insufficient'))

    def test_session_hides_private_keys(self):
        out = agent.public({'tool': 'get_evidence', 'rows': [], '_full_evidence': {'e': ['s']}})
        self.assertNotIn('_full_evidence', out)


# ---------------------------------------------------------------------------------------------------- graph tools
@unittest.skipUnless(GRAPH, 'local Neo4j graph not running or inputs missing')
class GraphToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GRAPH
        cls.db = sqlite3.connect(f'file:{KG}?mode=ro', uri=True)
        cls.soc = {}
        with open(FROZEN / 'meddra_pt_hierarchy.csv', newline='') as f:
            for r in csv.DictReader(f):
                cls.soc[r['pt_code']] = (r['soc_code'], r['soc_name'])

    def sql(self, q, params=()):
        return self.db.execute(q, params).fetchall()

    def test_graph_counts_match_kg_sqlite(self):
        for rel, table in (('SIDE_EFFECT', 'edge_side_effect'), ('USED_FOR', 'edge_used_for')):
            n = self.g.run(f'MATCH ()-[e:{rel}]->() RETURN count(e) AS c')[0]['c']
            self.assertEqual(n, self.sql(f'SELECT COUNT(*) FROM {table}')[0][0])
        self.assertEqual(self.g.run('MATCH (d:Drug) RETURN count(d) AS c')[0]['c'], self.sql('SELECT COUNT(*) FROM drug')[0][0])

    def test_pt_ranking_counts_records_once_per_pt(self):
        out = self.g.query_graph(**args(platforms=['reddit'], brands=['Zepbound'], sexes=['female']), group_by=['pt'], top_n=5)
        den = self.sql("SELECT COUNT(DISTINCT e.record_id) FROM edge_side_effect e JOIN drug d ON d.id = e.drug_id "
                       "WHERE e.platform = 'reddit' AND d.brand = 'Zepbound' AND e.writer_sex = 'female'")[0][0]
        num = dict(self.sql("SELECT COALESCE(s.pt_code, 'uncoded'), COUNT(DISTINCT e.record_id) FROM edge_side_effect e "
                            "JOIN drug d ON d.id = e.drug_id JOIN side_effect s ON s.id = e.side_effect_id "
                            "WHERE e.platform = 'reddit' AND d.brand = 'Zepbound' AND e.writer_sex = 'female' GROUP BY 1"))
        self.assertEqual(len(out['rows']), 5)
        for r in out['rows']:
            self.assertEqual(r['denominator'], den)
            self.assertEqual(r['records'], num[r['pt_code']])
        self.assertEqual(out['rows'][0]['records'], max(num.values()))

    def test_required_purpose_is_same_drug_same_record(self):
        out = self.g.query_graph(**args(require_purposes=[pt(T2DM)], targets=[pt(NAUSEA)]), group_by=[], top_n=5)
        req = ("WITH req AS (SELECT DISTINCT platform, record_id, drug_id FROM edge_used_for WHERE purpose_id = 'purpose:pt:10067585') "
               "SELECT e.platform, COUNT(DISTINCT e.record_id) FROM edge_side_effect e JOIN req USING (platform, record_id, drug_id) ")
        den = dict(self.sql(req + 'GROUP BY 1'))
        num = dict(self.sql(req + "WHERE e.side_effect_id = 'side_effect:pt:10028813' GROUP BY 1"))
        rows = {r['platform']: r for r in out['rows']}
        for p in ('reddit', 'webmd'):
            self.assertEqual((rows[p]['records'], rows[p]['denominator']), (num[p], den[p]))

    def test_age_filter_reports_missing_values(self):
        out = self.g.query_graph(**args(brands=['Ozempic'], age_groups=['45-54']), group_by=[], top_n=5)
        for s in out['summary']:
            before, missing, kept = self.sql(
                "SELECT COUNT(DISTINCT e.record_id), COUNT(DISTINCT CASE WHEN e.writer_age_group IS NULL THEN e.record_id END), "
                "COUNT(DISTINCT CASE WHEN e.writer_age_group = '45-54' THEN e.record_id END) FROM edge_side_effect e "
                "JOIN drug d ON d.id = e.drug_id WHERE e.platform = ? AND d.brand = 'Ozempic'", (s['platform'],))[0]
            self.assertEqual((s['records_before_age_sex_filter'], s['excluded_missing_age'], s['universe_records']), (before, missing, kept))
            self.assertIsNone(s['excluded_missing_sex'])

    def test_soc_grouping_follows_primary_path(self):
        out = self.g.query_graph(**args(platforms=['webmd']), group_by=['soc'], top_n=3)
        counts = {}
        for code, rec in self.sql("SELECT DISTINCT COALESCE(s.pt_code, 'uncoded'), e.record_id FROM edge_side_effect e "
                                  "JOIN side_effect s ON s.id = e.side_effect_id WHERE e.platform = 'webmd'"):
            counts.setdefault(self.soc[code][0] if code != 'uncoded' else 'uncoded', set()).add(rec)
        den = self.sql("SELECT COUNT(DISTINCT record_id) FROM edge_side_effect WHERE platform = 'webmd'")[0][0]
        for r in out['rows']:
            self.assertEqual((r['records'], r['denominator']), (len(counts[r['soc_code']]), den))
        self.assertEqual(out['rows'][0]['soc'], 'Gastrointestinal disorders')

    def test_group_target_counts_a_record_once(self):
        out = self.g.query_graph(**args(platforms=['webmd'], targets=[{'concept': 'gi', 'level': 'soc', 'code': GI_SOC}]),
                                 group_by=[], top_n=5)
        recs = {rec for code, rec in self.sql("SELECT s.pt_code, e.record_id FROM edge_side_effect e JOIN side_effect s "
                                              "ON s.id = e.side_effect_id WHERE e.platform = 'webmd' AND s.pt_code IS NOT NULL")
                if self.soc[code][0] == GI_SOC}
        self.assertEqual(out['rows'][0]['records'], len(recs))
        self.assertEqual(out['resolved_concepts']['targets'][0]['members'][0]['name'], 'Gastrointestinal disorders')

    def test_named_concept_counts_a_record_once_across_its_nodes(self):
        weight = [pt('10047895', 'weight loss'), pt('10050300', 'weight loss')]
        out = self.g.query_graph(**args(target='purpose', age_groups=['19-24', '55-64'], targets=weight + [pt(T2DM, 'type 2')]),
                                 group_by=['age_group'], top_n=10)
        rows = {(r['platform'], r['age_group'], r['concept']): r for r in out['rows']}
        for p in ('reddit', 'webmd'):
            for age in ('19-24', '55-64'):
                base = "FROM edge_used_for WHERE platform = ? AND writer_age_group = ?"
                den = self.sql(f'SELECT COUNT(DISTINCT record_id) {base}', (p, age))[0][0]
                num = self.sql(f"SELECT COUNT(DISTINCT record_id) {base} AND purpose_id IN ('purpose:pt:10047895', "
                               "'purpose:pt:10050300')", (p, age))[0][0]
                self.assertEqual((rows[(p, age, 'weight loss')]['records'], rows[(p, age, 'weight loss')]['denominator']), (num, den))
        self.assertEqual([c['concept'] for c in out['resolved_concepts']['targets']], ['weight loss', 'type 2'])
        self.assertEqual(set(out['_counted']['purpose']), {'10047895', '10050300', T2DM})
        out = self.g.query_graph(**args(target='purpose', platforms=['reddit'], targets=weight), group_by=['pt'], top_n=10)
        self.assertEqual({r['pt'] for r in out['rows']}, {'Weight decreased', 'Weight control'})
        self.assertTrue(all(r['common_wordings'] for r in out['rows']))

    def test_summary_states_known_values_and_profile_purposes(self):
        out = self.g.query_graph(**args(target='purpose', platforms=['webmd'], sexes=['female']), group_by=[], top_n=5)
        s = out['summary'][0]
        n, missing = self.sql("SELECT COUNT(DISTINCT record_id), COUNT(DISTINCT CASE WHEN writer_sex IS NULL THEN record_id END) "
                              "FROM edge_used_for WHERE platform = 'webmd'")[0]
        self.assertEqual((s['records_with_sex_stated'], s['excluded_missing_sex']), (n - missing, missing))
        self.assertGreater(s['universe_records_with_profile_purpose'], 0)
        text = self.g.query_graph(**args(target='purpose', platforms=['webmd'], purposes_from_text_only=True), group_by=[], top_n=5)
        want = self.sql("SELECT COUNT(DISTINCT record_id) FROM edge_used_for WHERE platform = 'webmd' "
                        "AND evidence NOT LIKE '%profile_condition%'")[0][0]
        self.assertEqual(text['summary'][0]['universe_records'], want)
        self.assertEqual(text['summary'][0]['universe_records_with_profile_purpose'], 0)

    def test_ties_at_the_cut_are_kept(self):
        out = self.g.query_graph(**args(platforms=['webmd'], brands=['Zepbound']), group_by=['pt'], top_n=3)
        counts = [r['records'] for r in out['rows']]
        rest = self.sql("SELECT COUNT(DISTINCT e.record_id) n FROM edge_side_effect e JOIN drug d ON d.id = e.drug_id "
                        "JOIN side_effect s ON s.id = e.side_effect_id WHERE e.platform = 'webmd' AND d.brand = 'Zepbound' "
                        "GROUP BY COALESCE(s.pt_code, 'uncoded') ORDER BY n DESC")
        self.assertEqual(len(counts), sum(1 for (n,) in rest if n >= rest[2][0]))

    def test_detail_dimensions(self):
        out = self.g.query_graph(**args(platforms=['reddit'], targets=[pt(NAUSEA)]), group_by=['impact'], top_n=10)
        want = dict(self.sql("SELECT COALESCE(impact, 'not stated'), COUNT(DISTINCT record_id) FROM edge_side_effect "
                             "WHERE platform = 'reddit' AND side_effect_id = 'side_effect:pt:10028813' GROUP BY 1"))
        self.assertEqual({r['impact']: r['records'] for r in out['rows']}, want)
        out = self.g.query_graph(**args(platforms=['webmd'], targets=[pt(NAUSEA)]), group_by=['action_taken'], top_n=10)
        want = {}
        for rec, acts in self.sql("SELECT record_id, action_taken FROM edge_side_effect WHERE platform = 'webmd' "
                                  "AND side_effect_id = 'side_effect:pt:10028813'"):
            for a in json.loads(acts) or ['none stated']:
                want.setdefault(a, set()).add(rec)
        self.assertEqual({r['action_taken']: r['records'] for r in out['rows']}, {k: len(v) for k, v in want.items()})

    def test_named_drugs_only_drops_community_ingredients(self):
        out = self.g.query_graph(**args(platforms=['reddit'], ingredients=['semaglutide'], named_drugs_only=True,
                                        targets=[pt(NAUSEA)]), group_by=[], top_n=5)
        n = self.sql("SELECT COUNT(DISTINCT e.record_id) FROM edge_side_effect e JOIN drug d ON d.id = e.drug_id "
                     "WHERE e.platform = 'reddit' AND d.ingredient = 'semaglutide' AND e.ingredient_basis IN ('name', 'profile') "
                     "AND e.side_effect_id = 'side_effect:pt:10028813'")[0][0]
        self.assertEqual(out['rows'][0]['records'], n)
        self.assertEqual(out['summary'][0]['universe_records_with_community_ingredient'], 0)

    def test_invalid_values_and_unknown_codes(self):
        with self.assertRaises(QueryError):
            self.g.query_graph(**args(platforms=[]), group_by=[], top_n=5)
        with self.assertRaises(QueryError):
            self.g.query_graph(**args(target='purpose'), group_by=['impact'], top_n=5)
        out = self.g.query_graph(**args(targets=[pt('99999999')]), group_by=[], top_n=5)
        self.assertEqual(out['codes_not_in_graph'], {'targets': ['pt:99999999']})

    def test_link_terms(self):
        out = self.g.link_terms('nausea', 'side_effect')
        self.assertEqual(out['rows'][0]['name'], 'Nausea')
        out = self.g.link_terms('throwing up', 'side_effect')
        self.assertIn('Vomiting', [r['name'] for r in out['rows'][:3]])
        out = self.g.link_terms('stomach problems', 'side_effect')
        self.assertIn(('soc', 'Gastrointestinal disorders'), [(r['level'], r['name']) for r in out['groups']])
        self.assertEqual(self.g.link_terms('pcos', 'purpose')['rows'][0]['name'], 'Polycystic ovarian syndrome')
        self.assertEqual(self.g.link_terms('anxiety', 'side_effect')['rows'][0]['name'], 'Anxiety')
        self.assertEqual(self.g.link_terms('gallstones', 'side_effect')['rows'][0]['name'], 'Cholelithiasis')
        self.assertIn(('soc', 'Skin and subcutaneous tissue disorders'),
                      [(r['level'], r['name']) for r in self.g.link_terms('skin', 'side_effect')['groups']])
        for generic in ('side effects', 'Saxenda complaints side effects', 'Rybelsus'):
            with self.assertRaises(QueryError):
                self.g.link_terms(generic, 'side_effect')
        self.assertTrue(self.g.link_terms('injection site', 'side_effect')['rows'])

    def test_evidence_is_deterministic_and_filtered(self):
        a = args(brands=['Zepbound'], sexes=['female'], targets=[pt(NAUSEA)])
        one = self.g.get_evidence(**a, wording=None, limit=4)
        two = self.g.get_evidence(**a, wording=None, limit=4)
        self.assertEqual([r['edge_id'] for r in one['rows']], [r['edge_id'] for r in two['rows']])
        self.assertTrue(one['rows'])
        for r in one['rows']:
            self.assertEqual((r['brand'], r['writer_sex'], r['pt']), ('Zepbound', 'female', 'Nausea'))
            self.assertIn(r['edge_id'], one['_full_evidence'])
        self.assertEqual(len(one['_record_ids']), len(one['rows']))
        self.assertNotIn('record_id', one['rows'][0])

    def test_action_filter_sets_the_universe(self):
        out = self.g.query_graph(**args(brands=['Mounjaro'], actions=['stopped'], targets=[pt(NAUSEA)]), group_by=[], top_n=5)
        for r in out['rows']:
            recs = {}
            for rec, sid, acts in self.sql("SELECT e.record_id, e.side_effect_id, e.action_taken FROM edge_side_effect e "
                                           "JOIN drug d ON d.id = e.drug_id WHERE e.platform = ? AND d.brand = 'Mounjaro'",
                                           (r['platform'],)):
                if 'stopped' in json.loads(acts):
                    recs.setdefault(rec, set()).add(sid)
            self.assertEqual(r['denominator'], len(recs))
            self.assertEqual(r['records'], sum('side_effect:pt:10028813' in v for v in recs.values()))

    def test_evidence_wording_filter(self):
        out = self.g.get_evidence(**args(brands=['Wegovy']), wording='sulfur burps', limit=5)
        self.assertTrue(out['rows'])
        for r in out['rows']:
            self.assertIn('sulfur', r['wording'].lower())
            self.assertIn('burp', r['wording'].lower())

    def frozen_rq2(self):
        rows = {}
        with gzip.open(FROZEN / 'rq2_faers_vs_patient_primary_clinical.csv.gz', 'rt') as f:
            for r in csv.DictReader(f):
                rows[(r['level'], r['ingredient'], r['brand'], r['pt_code'])] = r
        return rows

    def test_compare_faers_matches_frozen_rq2_table(self):
        frozen = self.frozen_rq2()
        scopes = {'all': ('all', 'all', 'all'), 'brand:Wegovy': ('brand', 'semaglutide', 'wegovy'),
                  'brand:Mounjaro': ('brand', 'tirzepatide', 'mounjaro'), 'ingredient:liraglutide': ('ingredient', 'liraglutide', ''),
                  'unspecified_brand:tirzepatide': ('unspecified', 'tirzepatide', 'unspecified')}
        out = self.g.compare_faers(targets=[pt(NAUSEA), pt(VOMITING), pt(PANCREATITIS)], scopes=list(scopes), sexes=None,
                                   age_groups=None, rank_by='faers', top_n=5, include_outcomes=False)
        self.assertEqual(len(out['rows']), 15)
        for r in out['rows']:
            f = frozen[scopes[r['scope']] + (r['concept'],)]
            self.assertEqual([r['faers_reports'], r['faers_denominator'], r['reddit_records'], r['reddit_denominator'],
                              r['webmd_records'], r['webmd_denominator']],
                             [int(f[k]) for k in ('faers_n_reports', 'faers_denominator', 'reddit_n_records', 'reddit_denominator',
                                                  'webmd_n_records', 'webmd_denominator')], r['scope'] + ' ' + r['concept'])

    def test_compare_faers_outcomes_match_frozen_table(self):
        with open(FROZEN / 'faers_rq2_hospitalization_action.csv', newline='') as f:
            frozen = {(r['level'], r['ingredient'], r['brand']): r for r in csv.DictReader(f) if r['config'] == 'primary'}
        scopes = {'all': ('all', 'all', 'all'), 'brand:Trulicity': ('brand', 'dulaglutide', 'trulicity'),
                  'ingredient:semaglutide': ('ingredient', 'semaglutide', '')}
        out = self.g.compare_faers(targets=[pt(NAUSEA)], scopes=list(scopes), sexes=None, age_groups=None, rank_by='faers',
                                   top_n=5, include_outcomes=True)
        for o in out['outcomes']:
            f = frozen[scopes[o['scope']]]
            self.assertEqual([o['faers_reports'], o['faers_hospitalization'], o['faers_action_known_reports'],
                              o['faers_drug_withdrawn'], o['faers_dose_reduced']],
                             [int(f[k]) for k in ('n_reports', 'hospitalization_yes', 'action_known_reports',
                                                  'action_any_drug_withdrawn', 'action_any_dose_reduced')], o['scope'])

    def test_compare_faers_population_filter_matches_frozen_rq3_table(self):
        with gzip.open(FROZEN / 'faers_rq3_pt_by_age_sex.csv.gz', 'rt') as f:
            frozen = {(r['level'], r['brand'], r['age_group'], r['sex'], r['pt_code']): int(r['n_reports']) for r in csv.DictReader(f)}
        out = self.g.compare_faers(targets=[pt(NAUSEA)], scopes=['all', 'brand:Ozempic'], sexes=['female'], age_groups=['45-54'],
                                   rank_by='faers', top_n=5, include_outcomes=False)
        got = {r['scope']: r['faers_reports'] for r in out['rows']}
        self.assertEqual(got['all'], frozen[('all', 'all', '45-54', 'female', NAUSEA)])
        self.assertEqual(got['brand:Ozempic'], frozen[('brand', 'ozempic', '45-54', 'female', NAUSEA)])
        self.assertIsNotNone(out['summary'][0]['faers_excluded_missing_sex'])

    def test_compare_faers_excludes_nonclinical_concepts(self):
        code = self.g.run("MATCH (n:SideEffect {pt: 'Off label use'}) RETURN n.pt_code AS c")[0]['c']
        out = self.g.compare_faers(targets=[pt(code)], scopes=['all'], sexes=None, age_groups=None, rank_by='faers', top_n=5,
                                   include_outcomes=False)
        self.assertEqual(out['rows'], [])
        self.assertEqual(out['excluded_nonclinical_concepts'], [code])


# ---------------------------------------------------------------------------------------------------- verifier
def outputs_fixture():
    return {'t0': {'call': 't0', 'tool': 'overview', 'rows': [{'row': 1, 'platform': 'reddit', 'records_processed': 500}]},
            't2': {'call': 't2', 'tool': 'query_graph', 'summary': [{'row': 1, 'platform': 'reddit', 'universe_records': 30}],
                   'rows': [{'row': 2, 'platform': 'reddit', 'pt': 'Nausea', 'records': 8, 'denominator': 30, 'percent': 26.7}]},
            't3': {'call': 't3', 'tool': 'get_evidence', 'rows': [{'row': 1, 'edge_id': 'reddit:x:d0:s0'}],
                   '_full_evidence': {'reddit:x:d0:s0': ['The nausea hit me hard on day two, so I stayed in bed.']}}}


def answer(text, cites, quotes=(), answer_type='answered', limitations=()):
    return agent.FinalAnswer(answer=text, answer_type=answer_type,
                             citations=[agent.Citation(id=i, call=c, row=r, field=f, value=v) for i, c, r, f, v in cites],
                             quotes=[agent.Quote(edge_id=e, text=t) for e, t in quotes], limitations=list(limitations))


class VerifierTest(unittest.TestCase):
    def test_valid_answer_with_quote(self):
        a = answer('Nausea: 8 [c1] of 30 [c2] Reddit records (26.7% [c3]); one writer said "the nausea hit me hard on day two".',
                   [('c1', 't2', 2, 'records', '8'), ('c2', 't2', 2, 'denominator', '30'), ('c3', 't2', 2, 'percent', '26.7')],
                   [('reddit:x:d0:s0', 'The nausea hit me hard on day two')])
        self.assertEqual(agent.verify(a, outputs_fixture())[0], [])

    def test_rounded_percent_is_accepted(self):
        a = answer('About 27% [c1].', [('c1', 't2', 2, 'percent', '27')])
        self.assertEqual(agent.verify(a, outputs_fixture())[0], [])

    def test_wrong_value_and_missing_citation_fail(self):
        a = answer('9 [c1] records; 30 [c2] in all.', [('c1', 't2', 2, 'records', '9')])
        problems = agent.verify(a, outputs_fixture())[0]
        self.assertTrue(any('c1' in p and 'cited value' in p for p in problems))
        self.assertTrue(any('[c2] has no citation' in p for p in problems))

    def test_number_in_text_must_match_citation(self):
        a = answer('Nausea was reported in 12 records [c1].', [('c1', 't2', 2, 'records', '8')])
        self.assertTrue(any('not the number written just before' in p for p in agent.verify(a, outputs_fixture())[0]))

    def test_each_tag_follows_its_own_number(self):
        a = answer('FAERS: 8 reports out of 30 [c1] [c2].', [('c1', 't2', 2, 'records', '8'), ('c2', 't2', 2, 'denominator', '30')])
        problems = agent.verify(a, outputs_fixture())[0]
        self.assertEqual(len(problems), 2)
        self.assertIn('[c1]', problems[0])
        self.assertIn('the number 8', problems[1])

    def test_tag_on_a_name_is_only_a_warning(self):
        a = answer('Top PT: Nausea [c1].', [('c1', 't2', 2, 'pt', 'Nausea')])
        problems, warnings = agent.verify(a, outputs_fixture())
        self.assertEqual(problems, [])
        self.assertTrue(any('non-numeric' in w for w in warnings))

    def test_malformed_citation_id_is_a_problem_not_a_crash(self):
        a = answer('8 [c1].', [('c1', 't2', 2, 'records', '8'), ('d c4', 't2', 2, 'records', '8')])
        problems = agent.verify(a, outputs_fixture())[0]
        self.assertTrue(any("'d c4'" in p for p in problems))

    def test_bad_call_row_field(self):
        a = answer('8 [c1] 8 [c2] 8 [c3]', [('c1', 't9', 2, 'records', '8'), ('c2', 't2', 7, 'records', '8'), ('c3', 't2', 2, 'n', '8')])
        self.assertEqual(len(agent.verify(a, outputs_fixture())[0]), 3)

    def test_quotes_must_be_exact_and_returned(self):
        bad = answer('A writer said "the nausea was unbearable for weeks".', [],
                     [('reddit:x:d0:s0', 'the nausea was unbearable for weeks'), ('reddit:z:d0:s0', 'anything')])
        problems = agent.verify(bad, outputs_fixture())[0]
        self.assertTrue(any('not an exact passage' in p for p in problems))
        self.assertTrue(any('reddit:z:d0:s0' in p for p in problems))
        unlisted = answer('A writer said "so I stayed in bed".', [])
        self.assertTrue(any('quotes lists no exact source passage' in p for p in agent.verify(unlisted, outputs_fixture())[0]))

    def test_quotes_across_sentences_and_wrong_edge_hint(self):
        outputs = outputs_fixture()
        outputs['t3']['_full_evidence']['reddit:x:d0:s0'].append('Then it passed.')
        outputs['t3']['_full_evidence']['reddit:y:d0:s0'] = ['I waited ten days before my next shot.']
        across = answer('One writer: "so I stayed in bed. Then it passed".', [],
                        [('reddit:x:d0:s0', 'so I stayed in bed. Then it passed')])
        self.assertEqual(agent.verify(across, outputs)[0], [])
        wrong = answer('A writer said "I waited ten days before my next shot".', [],
                       [('reddit:x:d0:s0', 'I waited ten days before my next shot')])
        self.assertTrue(any('cite that edge_id' in p and 'reddit:y:d0:s0' in p for p in agent.verify(wrong, outputs)[0]))

    def test_quoted_names_and_question_words_are_not_quotes(self):
        a = answer('The PT "Nausea Vomiting Group" matched; you asked whether it happens "about as often as FAERS".', [])
        outputs = outputs_fixture()
        outputs['t2']['rows'][0]['pt'] = 'Nausea vomiting group'
        self.assertEqual(agent.verify(a, outputs, 'Is it reported about as often as FAERS says?')[0], [])

    def test_declined_answer_without_data_is_valid(self):
        a = answer('I cannot give medical advice; please ask your doctor or pharmacist.', [], answer_type='declined')
        self.assertEqual(agent.verify(a, outputs_fixture()), ([], []))

    def test_uncited_numbers_fail_and_limitation_numbers_warn(self):
        a = answer('Women aged 45-54 wrote 8 records.', [], limitations=['Only 16 records state an age.'])
        problems, warnings = agent.verify(a, outputs_fixture())
        self.assertEqual(len(problems), 1)
        self.assertIn('the number 8', problems[0])
        self.assertEqual(len(warnings), 1)
        mixed = answer('Nausea: 8 [c1] records, about 27% of Reddit records.', [('c1', 't2', 2, 'records', '8')])
        self.assertTrue(any('the number 27' in p for p in agent.verify(mixed, outputs_fixture())[0]))

    def test_exempt_numbers(self):
        a = answer('On 1 mg, writers aged 52 and over, the 2nd dose, top 5 PTs: 8 [c1] records (see MedDRA 29.1, 2026-06-30).',
                   [('c1', 't2', 2, 'records', '8')])
        self.assertEqual(agent.verify(a, outputs_fixture(), 'I am 52 and on 1 mg')[0], [])


class ConceptCheckTest(unittest.TestCase):
    def outputs(self, counted):
        link = {'call': 't1', 'tool': 'link_terms', 'text': 'weight loss', '_kind': 'purpose', 'groups': [],
                'rows': [{'row': 1, 'level': 'pt', 'code': '10047895', 'name': 'Weight decreased', 'match': 'name+verbatim',
                          'matching_wording_records_reddit': 4292, 'matching_wording_records_webmd': 89},
                         {'row': 2, 'level': 'pt', 'code': '10050300', 'name': 'Weight control', 'match': 'verbatim',
                          'matching_wording_records_reddit': 159, 'matching_wording_records_webmd': 868},
                         {'row': 3, 'level': 'pt', 'code': '10049084', 'name': 'Weight loss poor', 'match': 'verbatim',
                          'matching_wording_records_reddit': 60, 'matching_wording_records_webmd': 0}]}
        query = {'call': 't2', 'tool': 'query_graph', 'filters': {'target': 'purpose'}, 'rows': [],
                 '_counted': {'purpose': counted, 'side_effect': []}}
        return {'t1': link, 't2': query}

    def test_large_matching_row_left_out_fails_unless_named(self):
        problems = agent.concept_gaps('Weight control was reported in 25 of 293 posts.', self.outputs(['10050300']))
        self.assertEqual(len(problems), 1)
        self.assertIn('Weight decreased', problems[0])
        self.assertEqual(agent.concept_gaps('Weight decreased and Weight control together: ...', self.outputs(['10050300'])), [])
        self.assertEqual(agent.concept_gaps('weight loss: ...', self.outputs(['10047895', '10050300'])), [])

    def test_unused_link_output_is_not_checked(self):
        self.assertEqual(agent.concept_gaps('nothing counted', self.outputs([])), [])

    def test_record_ids_in_the_answer_fail(self):
        outs = {'t3': {'call': 't3', 'tool': 'get_evidence', 'rows': [], '_record_ids': ['t3_zqrdpc']}}
        self.assertTrue(agent.leaked_ids('Most everything tastes weird (t3_zqrdpc).', outs))
        self.assertTrue(agent.leaked_ids('see reddit:t1_jglk7yw:d0:s0', {}))
        self.assertEqual(agent.leaked_ids('Most everything tastes weird.', outs), [])


class QueryCheckTest(unittest.TestCase):
    def call(self, tool, **a):
        return {'tool': tool, 'arguments': json.dumps(a), 'error': None}

    def test_requires_and_forbids(self):
        calls = [self.call('query_graph', brands=['Wegovy', 'Ozempic'], actions=['stopped'], named_drugs_only=False,
                           targets=[{'concept': 'gi', 'level': 'soc', 'code': GI_SOC}], require_purposes=[pt(T2DM)])]
        ok = {'requires': [{'tool': 'query_graph', 'args': {'brands': ['Wegovy'], 'actions': ['stopped'],
                                                             'target_levels': ['soc'], 'require_purposes': [f'pt:{T2DM}']}}],
              'forbids': [{'tool': 'query_graph', 'args': {'named_drugs_only': True}}]}
        self.assertEqual(check_calls(ok, calls), [])
        bad = {'requires': [{'tool': 'get_evidence', 'args': {}}, {'tool': 'query_graph', 'args': {'brands': ['Zepbound']}}],
               'forbids': [{'tool': 'query_graph', 'args': {'actions': ['stopped']}}]}
        self.assertEqual(len(check_calls(bad, calls)), 3)
        failed = [{'tool': 'query_graph', 'arguments': '{}', 'error': 'bad'}]
        self.assertEqual(len(check_calls({'requires': [{'tool': 'query_graph', 'args': {}}]}, failed)), 1)


# ---------------------------------------------------------------------------------------------------- loop
class StubGraph:
    manifest = {'kg_version': 'stub', 'loaded_utc': None}

    def overview(self):
        return {'tool': 'overview', 'rows': [{'row': 1, 'platform': 'reddit', 'records_processed': 500}]}

    def query_graph(self, **kw):
        return {'tool': 'query_graph', 'summary': [{'row': 1, 'platform': 'reddit', 'universe_records': 30}],
                'rows': [{'row': 2, 'platform': 'reddit', 'pt': 'Nausea', 'records': 8, 'denominator': 30}]}

    def close(self):
        pass


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def model_dump(self, **_):
        return json.loads(json.dumps(self.data))


class FakeClient:
    def __init__(self, scripted):
        self.scripted, self.bodies = list(scripted), []
        self.responses = self

    def create(self, **body):
        self.bodies.append(json.loads(json.dumps(body)))
        return FakeResponse(self.scripted.pop(0))


def response(output):
    return {'status': 'completed', 'output': output, 'usage': {'input_tokens': 1000, 'output_tokens': 100,
                                                                'input_tokens_details': {'cached_tokens': 0}}}


def final(obj):
    return response([{'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'enc'},
                     {'type': 'message', 'role': 'assistant', 'phase': 'final_answer',
                      'content': [{'type': 'output_text', 'text': json.dumps(obj)}]}])


def call(name, arguments, call_id='fc_1'):
    return response([{'type': 'reasoning', 'id': 'rs_0', 'summary': [], 'encrypted_content': 'enc'},
                     {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': json.dumps(arguments)}])


QUERY = call('query_graph', {**args(brands=['Mounjaro'], targets=[pt(NAUSEA)]), 'group_by': [], 'top_n': 5})
EMPTY = {'answer': 'Insufficient data.', 'answer_type': 'limited', 'citations': [], 'quotes': [], 'limitations': []}


class LoopTest(unittest.TestCase):
    def test_tool_round_then_verified_answer(self):
        good = {'answer': 'Reddit: 8 [c1] of 30 [c2] records.', 'answer_type': 'answered', 'quotes': [],
                'limitations': ['records, not patients'],
                'citations': [{'id': 'c1', 'call': 't1', 'row': 2, 'field': 'records', 'value': '8'},
                              {'id': 'c2', 'call': 't1', 'row': 2, 'field': 'denominator', 'value': '30'}]}
        client = FakeClient([QUERY, final(good)])
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'trace.jsonl'
            res = agent.run('How often is nausea reported with Mounjaro?', StubGraph(), client=client, trace_path=trace)
            self.assertTrue(res['verified'], res['problems'])
            self.assertFalse(res['repair_used'])
            line = json.loads(trace.read_text())
            self.assertEqual(line['result']['run_id'], res['run_id'])
            self.assertEqual([c['call'] for c in line['tool_calls']], ['t0', 't1'])
        body = client.bodies[0]
        self.assertIs(body['store'], False)
        self.assertEqual(body['include'], ['reasoning.encrypted_content'])
        self.assertTrue(all(t['strict'] for t in body['tools']))
        self.assertTrue(body['text']['format']['strict'])
        second = client.bodies[1]['input']
        self.assertEqual(second[1]['type'], 'reasoning')
        self.assertEqual(json.loads(second[-1]['output'])['call'], 't1')

    def test_invalid_tool_arguments_return_an_error_output(self):
        client = FakeClient([call('query_graph', {'target': 'side_effect'}), final(EMPTY)])
        res = agent.run('q', StubGraph(), client=client, trace_path=None)
        self.assertTrue(res['tool_calls'][1]['error'])

    def test_failed_verification_gets_one_repair_round(self):
        bad = {'answer': 'Reddit: 999 [c1] records.', 'answer_type': 'answered', 'quotes': [], 'limitations': [],
               'citations': [{'id': 'c1', 'call': 't0', 'row': 1, 'field': 'records_processed', 'value': '999'}]}
        client = FakeClient([final(bad), final(bad)])
        res = agent.run('q', StubGraph(), client=client, trace_path=None)
        self.assertFalse(res['verified'])
        self.assertTrue(res['repair_used'])
        self.assertIn('Verification of your final answer failed', client.bodies[1]['input'][-1]['content'])

    def test_refusal_is_an_anomaly_not_an_answer(self):
        refusal = response([{'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'no'}]}])
        res = agent.run('q', StubGraph(), client=FakeClient([refusal]), trace_path=None)
        self.assertFalse(res['verified'])
        self.assertIsNone(res['answer'])
        self.assertTrue(res['problems'][0].startswith('answer not accepted (refusal)'))

    def test_response_cap_stops_a_model_that_keeps_calling_tools(self):
        res = agent.run('q', StubGraph(), client=FakeClient([QUERY for _ in range(20)]), trace_path=None, max_tool_rounds=2)
        self.assertFalse(res['verified'])
        self.assertTrue(res['stop_reason'].startswith('response cap'))

    def test_round_limit_forces_answer(self):
        client = FakeClient([QUERY, QUERY, final(EMPTY)])
        res = agent.run('q', StubGraph(), client=client, trace_path=None, max_tool_rounds=2)
        self.assertTrue(res['verified'])
        self.assertEqual([b['tool_choice'] for b in client.bodies], ['auto', 'auto', 'none'])


if __name__ == '__main__':
    unittest.main()
