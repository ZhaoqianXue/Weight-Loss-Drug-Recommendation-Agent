"""Website backend over the knowledge graph and the agent: view shaping, caching, HTTP contract and the streamed agent
events. A fake graph returns rows in the agent tools' formats, so no Neo4j server or API call is needed."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from weightloss.web.app import create_app
from weightloss.web.ask import AskService
from weightloss.web.kg import GraphUnavailable, KGService

NOTES = ['Counts are records (Reddit posts or comments, WebMD reviews), not patients; one person can write several records.']


def row(n, **fields):
    return {'row': n, **fields}


class FakeGraph:
    """Answers the four tools and the overview with fixed rows; records every call."""

    manifest = {'kg_version': 'kg-test', 'loaded_utc': '2026-10-18T00:00:00+00:00',
                'kg_build': {'built_utc': '2026-10-17T00:00:00+00:00',
                             'extraction': {'model': 'model-x', 'prompt_version': 'p-1'}}}

    def __init__(self):
        self.calls = []

    def overview(self):
        self.calls.append(('overview', {}))
        return {'tool': 'overview', 'graph': 'kg-test', 'meddra_version': '29.1', 'loaded_utc': '2026-10-18T00:00:00+00:00',
                'rows': [row(1, platform='reddit', records_processed=100, records_with_side_effect=60, side_effect_statements=90,
                             records_with_purpose=20, purpose_statements=25),
                         row(2, platform='webmd', records_processed=10, records_with_side_effect=6, side_effect_statements=8,
                             records_with_purpose=5, purpose_statements=5)],
                'faers': [row(3, source='FAERS', reports_primary_configuration=500, status='FAERS frozen')], 'notes': NOTES}

    def query_graph(self, **a):
        self.calls.append(('query_graph', a))
        dims, target = a['group_by'], a['target']
        rows = []
        if dims == ['soc']:
            rows = [row(2, platform='reddit', soc='Gastrointestinal disorders', soc_code='100', records=40, denominator=60, percent=66.7, sparse=False, small=False),
                    row(3, platform='reddit', soc='[uncoded]', soc_code='uncoded', records=5, denominator=60, percent=8.3, sparse=False, small=False)]
        elif dims[1:] == ['pt', 'soc']:
            level = dims[0]
            rows = [row(2, platform=a['platforms'][0], **{level: 'Ozempic'}, pt='Nausea', pt_code='1', soc='GI', soc_code='100', records=30,
                        denominator=50, percent=60.0, sparse=False, small=False),
                    row(3, platform=a['platforms'][0], **{level: 'Ozempic'}, pt='[uncoded]', pt_code='uncoded', soc=None, soc_code=None,
                        records=3, denominator=50, percent=6.0, sparse=False, small=False),
                    row(4, platform=a['platforms'][0], **{level: 'unspecified'}, pt='Nausea', pt_code='1', soc='GI', soc_code='100',
                        records=9, denominator=20, percent=45.0, sparse=False, small=False)]
        elif dims == ['pt'] and a.get('targets'):
            rows = [row(2, platform=a['platforms'][0], pt='Nausea', pt_code='1', records=35, denominator=60, percent=58.3, sparse=False, small=False)]
        elif dims == ['pt']:
            rows = [row(2, platform=p, pt='Nausea', pt_code='1', records=30, denominator=60, percent=50.0, sparse=False, small=False) for p in a['platforms']]
        elif dims == ['impact']:
            rows = [row(2, platform='reddit', impact='not stated', records=50, denominator=60, percent=83.3, sparse=False, small=False),
                    row(3, platform='reddit', impact='hospitalization', records=2, denominator=60, percent=3.3, sparse=True, small=True)]
        elif dims == ['age_group', 'sex']:
            rows = [row(2, platform='reddit', age_group='25-34', sex='female', records=7, denominator=60, percent=11.7, sparse=True, small=True)]
        else:  # one population dimension: a distribution
            rows = [row(2, platform='reddit', **{dims[0]: 'semaglutide'}, records=40, denominator=60, percent=66.7, sparse=False, small=False)]
        return {'tool': 'query_graph', 'summary': [row(1, platform=p, universe_records=60, records_before_age_sex_filter=60,
                                                       excluded_missing_age=None, excluded_missing_sex=None,
                                                       universe_records_with_uncoded_mention=4) for p in a['platforms']],
                'denominator_definition': 'records with a statement', 'rows_not_shown': {'reddit': 7}, 'rows': rows, 'notes': NOTES}

    def compare_faers(self, **a):
        self.calls.append(('compare_faers', a))
        return {'tool': 'compare_faers', 'status': 'FAERS frozen',
                'summary': [row(1, scope=a['scopes'][0], faers_denominator=400, reddit_denominator=60, webmd_denominator=6,
                                record_window='2017-12-05 to 2026-06-30')],
                'rows': [row(2, scope=a['scopes'][0], concept='Nausea', pt_code='1', faers_reports=80, faers_denominator=400,
                             faers_percent=20.0, reddit_records=30, reddit_denominator=60, reddit_percent=50.0, webmd_records=3,
                             webmd_denominator=6, webmd_percent=50.0, reddit_small=False, webmd_small=True)],
                'notes': ['never compute ratios']}

    def get_evidence(self, **a):
        self.calls.append(('get_evidence', a))
        return {'tool': 'get_evidence', 'summary': [row(1, platform=a['platforms'][0], matching_statements=12, matching_records=11)],
                'rows': [row(2, edge_id='reddit:t1_a:d0:s0', platform=a['platforms'][0], date='2025-01-01',
                             ingredient='semaglutide', brand='Ozempic', ingredient_basis='name', pt='Nausea', pt_code='1', soc='GI',
                             wording='nauseous', writer_age_group='not stated', writer_sex='not stated', dose=None, duration=None,
                             intensity='mild', impact=None, action_taken=[], evidence=['I felt nauseous all week.'])],
                'notes': NOTES, '_full_evidence': {'reddit:t1_a:d0:s0': ['I felt nauseous all week.']}}

    def link_terms(self, text, kind):
        self.calls.append(('link_terms', {'text': text, 'kind': kind}))
        return {'tool': 'link_terms', 'rows': [row(1, level='pt', code='1', name='Nausea')]}


class FakeResponse:
    def __init__(self, output):
        self.output = output


class FakeItem:
    def __init__(self, kind):
        self.type = kind


class FakeClient:
    class responses:  # noqa: N801 - mirrors the OpenAI client attribute
        @staticmethod
        def create(**body):
            return FakeResponse([FakeItem('function_call')])


def fake_run(question, graph, client, trace_path, label):
    """Stands in for weightloss.agent.agent.run: one model response, one tool call, a trace line, a result."""
    graph.overview()
    client.responses.create(model='m')
    output = graph.query_graph(target='side_effect', platforms=['reddit'], group_by=['pt'], top_n=5, targets=None)
    evidence = graph.get_evidence(target='side_effect', platforms=['reddit'], wording=None, limit=1)
    public = {k: v for k, v in evidence.items() if not k.startswith('_')}
    answer = {'answer': 'Nausea: 30 [c1] of 60 [c2] records; one writer said "felt nauseous all week".', 'answer_type': 'answered',
              'citations': [{'id': 'c1', 'call': 't1', 'row': 2, 'field': 'records', 'value': '30'},
                            {'id': 'c2', 'call': 't1', 'row': 2, 'field': 'denominator', 'value': '60'}],
              'quotes': [{'edge_id': 'reddit:t1_a:d0:s0', 'text': 'felt nauseous all week'}], 'limitations': ['Records, not patients.']}
    result = {'run_id': 'run-1', 'question': question, 'verified': True, 'answer': answer, 'problems': [], 'repair_used': False,
              'stop_reason': None, 'cost_usd': 0.002}
    trace = {'model': 'gpt-test', 'reasoning_effort': 'medium', 'agent_version': 'agent-test',
             'tool_calls': [{'call': 't0', 'tool': 'overview', 'arguments': {}, 'output': {}},
                            {'call': 't1', 'tool': 'query_graph', 'arguments': json.dumps({'target': 'side_effect', 'top_n': 5}),
                             'output': output, 'seconds': 0.1},
                            {'call': 't2', 'tool': 'get_evidence', 'arguments': {'limit': 1}, 'output': public, 'seconds': 0.1}],
             'result': result}
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    Path(trace_path).write_text(json.dumps(trace) + '\n')
    return result


def events(response):
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith('data: ')]


class KGServiceTest(unittest.TestCase):
    def test_graph_view_drops_unnamed_brands_and_uncoded_terms(self):
        graph = FakeGraph()
        view = KGService(graph=graph).graph_view('reddit', 'brand', 'side_effect', drugs=None, top_n=3)
        self.assertEqual([n['id'] for n in view['nodes']], ['drug:Ozempic', 'side_effect:1'])
        self.assertEqual(view['links'], [{'source': 'drug:Ozempic', 'target': 'side_effect:1', 'kind': 'side_effect', 'records': 30,
                                          'denominator': 50, 'percent': 60.0, 'sparse': False, 'small': False}])
        pt = view['nodes'][1]
        self.assertEqual((pt['records'], pt['denominator'], pt['soc']), ({'side_effect': 30}, 60, 'GI'))  # from the totals query
        totals = [args for name, args in graph.calls if name == 'query_graph' and args['group_by'] == ['pt']]
        # Totals count the named brands only, never records that name no brand.
        self.assertEqual(totals[0]['brands'], ['Ozempic', 'Wegovy', 'Rybelsus', 'Mounjaro', 'Zepbound', 'Victoza', 'Saxenda', 'Trulicity'])
        self.assertEqual(view['nodes'][0]['records'], {'side_effect': 50})
        self.assertEqual(view['rows_not_shown'], {'side_effect': 7})
        first = next(args for name, args in graph.calls if name == 'query_graph' and args['group_by'][1:] == ['pt', 'soc'])
        # Computed once at the largest top_n (plus one row for the uncoded bucket), then cut to the requested 3.
        self.assertEqual((first['platforms'], first['group_by'], first['top_n'], first['brands']), (['reddit'], ['brand', 'pt', 'soc'], 11, None))

    def test_filters_keep_all_and_none_apart(self):
        graph = FakeGraph()
        kg = KGService(graph=graph)
        kg.graph_view('webmd', 'ingredient', 'purpose', drugs=['semaglutide'], age_groups=['25-34'], sexes=None, soc='100')
        args = graph.calls[0][1]
        self.assertEqual((args['ingredients'], args['age_groups'], args['sexes'], args['targets']),
                         (['semaglutide'], ['25-34'], None, [{'level': 'soc', 'code': '100'}]))
        kg.graph_view('webmd', 'ingredient', 'purpose', drugs=[])
        views = [args for name, args in graph.calls if name == 'query_graph' and args['group_by'][1:] == ['pt', 'soc']]
        self.assertEqual([v['ingredients'] for v in views], [['semaglutide'], []])

    def test_patterns_skip_uncoded_and_keep_platforms_apart(self):
        patterns = KGService(graph=FakeGraph()).patterns()
        sections = {s['key']: s for s in patterns['sections']}
        self.assertEqual([r['label'] for r in sections['socs']['rows']], ['Gastrointestinal disorders'])
        self.assertEqual({r['platform'] for r in sections['side_effects']['rows']}, {'reddit', 'webmd'})
        self.assertEqual([r['label'] for r in sections['impacts']['rows']], ['not stated', 'hospitalization'])
        self.assertEqual(patterns['population'][0]['records'], 7)

    def test_results_are_cached_in_memory_and_on_disk(self):
        with tempfile.TemporaryDirectory() as cache:
            graph = FakeGraph()
            first = KGService(graph=graph, cache_dir=Path(cache)).faers('brand:Ozempic')
            calls = len(graph.calls)
            again = KGService(graph=graph, cache_dir=Path(cache))
            self.assertEqual(again.faers('brand:Ozempic'), first)
            self.assertEqual(len(graph.calls), calls)  # served from disk
            changed = FakeGraph()
            changed.manifest = {**FakeGraph.manifest, 'loaded_utc': 'later'}
            KGService(graph=changed, cache_dir=Path(cache)).faers('brand:Ozempic')
            self.assertEqual(len(changed.calls), 1)  # a reloaded graph starts a new cache

    def test_evidence_passes_drug_and_term(self):
        graph = FakeGraph()
        view = KGService(graph=graph).evidence('side_effect', 'reddit', 'brand', 'Ozempic', '1', 3)
        args = graph.calls[0][1]
        self.assertEqual((args['brands'], args['targets'], args['limit']), (['Ozempic'], [{'level': 'pt', 'code': '1'}], 3))
        self.assertEqual(view['matching_records'], 11)
        self.assertNotIn('_full_evidence', view)


class WebApiTest(unittest.TestCase):
    def client(self, graph=None, **ask):
        kg = KGService(graph=graph or FakeGraph())
        service = AskService(kg, run=fake_run, client_factory=lambda: FakeClient(), key_check=lambda: True,
                             trace_dir=Path(self.trace_dir.name), **ask)
        return TestClient(create_app(frontend_dir=Path(self.trace_dir.name), kg_service=kg, ask_service=service, warm=False))

    def setUp(self):
        self.trace_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.trace_dir.cleanup()

    def test_views(self):
        with self.client() as client:
            overview = client.get('/api/overview').json()
            self.assertEqual((overview['faers_reports'], overview['versions']['extraction_prompt']), (500, 'p-1'))
            self.assertEqual([s['code'] for s in overview['socs']], ['100'])
            self.assertEqual(client.post('/api/graph', json={'platform': 'reddit', 'top_n': 2}).status_code, 200)
            self.assertEqual(client.post('/api/graph', json={'top_n': 11}).status_code, 422)
            self.assertEqual(client.post('/api/graph', json={'unknown': 1}).status_code, 422)
            self.assertEqual(client.get('/api/patterns').status_code, 200)
            faers = client.post('/api/faers', json={'scope': 'ingredient:semaglutide'}).json()
            self.assertEqual((faers['rows'][0]['faers_reports'], faers['rows'][0]['code'], faers['rows'][0]['webmd_small']), (80, '1', True))
            self.assertEqual(faers['window'], '2017-12-05 to 2026-06-30')
            self.assertEqual(client.post('/api/faers', json={'scope': 'drop table'}).status_code, 422)
            evidence = client.post('/api/evidence', json={'target': 'side_effect', 'platform': 'webmd', 'code': '1'}).json()
            self.assertEqual(evidence['statements'][0]['evidence'], ['I felt nauseous all week.'])
            self.assertEqual(evidence['statements'][0]['record_id'], 't1_a')  # from the statement id

    def test_graph_down_answers_503(self):
        def down():
            raise GraphUnavailable('not running')
        kg = KGService(connect=down)
        with TestClient(create_app(frontend_dir=Path(self.trace_dir.name), kg_service=kg, warm=False)) as client:
            response = client.get('/api/overview')
            self.assertEqual((response.status_code, response.json()), (503, {'detail': 'not running'}))

    def test_ask_streams_progress_then_a_resolved_result(self):
        with self.client() as client, client.stream('POST', '/api/ask', json={'question': 'Nausea on Ozempic?'}) as response:
            self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))
            stream = events(response)
        kinds = [(e['type'], e.get('status')) for e in stream]
        self.assertEqual(kinds, [('start', None), ('model', 'running'), ('model', 'done'), ('tool', 'running'), ('tool', 'done'),
                                 ('tool', 'running'), ('tool', 'done'), ('result', None)])
        self.assertEqual(stream[4]['rows'], 2)  # one summary row and one data row
        result = stream[-1]['result']
        self.assertEqual(result['citations'][0]['source']['records'], 30)
        self.assertEqual(result['citations'][0]['arguments'], {'target': 'side_effect', 'top_n': 5})
        self.assertEqual(result['quotes'][0]['statement']['record_id'], 't1_a')
        self.assertEqual([c['tool'] for c in result['tool_calls']], ['query_graph', 'get_evidence'])
        self.assertEqual((result['model'], result['verified'], result['cost_usd']), ('gpt-test', True, 0.002))
        self.assertEqual(len(list(Path(self.trace_dir.name).rglob('*.jsonl'))), 1)

    def test_ask_refuses_without_key_or_over_budget(self):
        kg = KGService(graph=FakeGraph())
        no_key = AskService(kg, run=fake_run, key_check=lambda: False, trace_dir=Path(self.trace_dir.name))
        self.assertFalse(no_key.status()['available'])
        with self.client(budget_usd=0.001) as client:
            with client.stream('POST', '/api/ask', json={'question': 'q'}) as response:
                self.assertEqual(events(response)[-1]['type'], 'result')
            self.assertFalse(client.get('/api/ask/status').json()['available'])
            with client.stream('POST', '/api/ask', json={'question': 'q'}) as response:
                self.assertEqual([e['type'] for e in events(response)], ['error'])
        with self.client() as client:
            self.assertEqual(client.post('/api/ask', json={'question': ''}).status_code, 422)

    def test_agent_failure_ends_the_stream_with_an_error(self):
        def broken(*args, **kwargs):
            raise RuntimeError('boom')
        kg = KGService(graph=FakeGraph())
        service = AskService(kg, run=broken, client_factory=lambda: FakeClient(), key_check=lambda: True, trace_dir=Path(self.trace_dir.name))
        with TestClient(create_app(frontend_dir=Path(self.trace_dir.name), kg_service=kg, ask_service=service, warm=False)) as client:
            with client.stream('POST', '/api/ask', json={'question': 'q'}) as response:
                self.assertEqual([e['type'] for e in events(response)], ['start', 'error'])


if __name__ == '__main__':
    unittest.main()
