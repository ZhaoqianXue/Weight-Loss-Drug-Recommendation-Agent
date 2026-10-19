"""Website views over the real knowledge graph (skipped when the local Neo4j instance is not running)."""
import collections
import unittest

from weightloss.web.kg import graph_running

RUNNING = graph_running()


@unittest.skipUnless(RUNNING, 'local Neo4j is not running (python -m weightloss.graph.neo4j_db start)')
class WebKnowledgeGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from weightloss.web.kg import KGService
        cls.kg = KGService()

    @classmethod
    def tearDownClass(cls):
        cls.kg.close()

    def test_graph_view_follows_the_counting_rules(self):
        view = self.kg.graph_view('webmd', 'brand', 'both', top_n=3)
        nodes = {n['id']: n for n in view['nodes']}
        self.assertTrue(view['links'])
        self.assertNotIn('drug:unspecified', nodes)
        per_drug = collections.Counter()
        for link in view['links']:
            self.assertLessEqual(link['records'], link['denominator'])
            self.assertEqual(nodes[link['source']]['records'][link['kind']], link['denominator'])
            # A PT's total counts each record once across the shown drugs, so it is at least any one drug's count.
            self.assertGreaterEqual(nodes[link['target']]['records'][link['kind']], link['records'])
            per_drug[(link['source'], link['kind'])] += 1
        self.assertLessEqual(max(per_drug.values()), 3)

    def test_age_filter_reports_what_it_left_out(self):
        view = self.kg.graph_view('webmd', 'ingredient', 'side_effect', age_groups=['45-54'], top_n=2)
        summary = view['summary'][0]
        self.assertGreater(summary['records_before_age_sex_filter'], summary['universe_records'])
        self.assertGreater(summary['excluded_missing_age'], 0)

    def test_faers_rows_keep_channels_apart(self):
        view = self.kg.faers('ingredient:liraglutide', top_n=5)
        self.assertGreaterEqual(len(view['rows']), 5)  # rows tied at the cut are all kept
        self.assertTrue(view['window'])
        for row in view['rows']:
            self.assertTrue(row['code'])
            self.assertLessEqual(row['faers_reports'], row['faers_denominator'])
            self.assertLessEqual(row['reddit_records'], row['reddit_denominator'])

    def test_example_statements_carry_their_record(self):
        # Side-effect and purpose statements both come with their record; purposes have no action list of their own.
        for target, code in (('side_effect', '10028813'), ('purpose', '10047895')):  # Nausea; Weight decreased
            for platform in ('reddit', 'webmd'):
                view = self.kg.evidence(target, platform, level='brand', code=code, limit=3)
                self.assertTrue(view['statements'])
                for s in view['statements']:
                    self.assertTrue(s['record_id'])
                    self.assertIsInstance(s['action_taken'], list)
                    self.assertTrue(s['edge_id'].startswith(f"{platform}:{s['record_id']}:d"))


if __name__ == '__main__':
    unittest.main()
