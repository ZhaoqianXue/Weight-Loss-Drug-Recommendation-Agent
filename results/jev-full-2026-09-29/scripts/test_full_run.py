"""Offline tests for full_run.py (httpx.MockTransport; temp dirs; tiny fake manifest; no network, no key file)."""
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
KEY = 'ts-test-secret-abcdef'
REAL = httpx.Client


def ok(p=0.7, tok=100):
    return httpx.Response(200, headers={'x-typesafe-request-id': 'rq'}, json={'model': 'jev-1.13.0', 'answers': {'strict_keep': {
        'choice': 'keep' if p >= 0.5 else 'drop', 'confidence': 0.9, 'probabilities': {'keep': p, 'drop': round(1 - p, 2)}}}, 'usage': {'input_tokens': tok}})


class FullRunTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        base = Path(self.t.name)
        (base / 'res').mkdir(); (base / 'cache').mkdir()
        os.environ.update({'JEV_FULL_RES': str(base / 'res'), 'JEV_FULL_CACHE': str(base / 'cache'), 'TYPESAFE_API_KEY': KEY})
        import full_run
        self.m = importlib.reload(full_run)
        self.m.time.sleep = lambda s: None
        self.m.TARGET_RPM = 10 ** 9
        self.shards = {'s1': [f'reddit:t1_a{i}' for i in range(5)], 's2': [f'reddit:t1_b{i}' for i in range(4)]}
        man = {'records': 1_030_309, 'unique_ids': 1_030_309, 'shards': [
            {'shard': s, 'records': len(v), 'ids_sha256': hashlib.sha256('\n'.join(sorted(v)).encode()).hexdigest()} for s, v in self.shards.items()]}
        (base / 'res' / 'input-manifest.json').write_text(json.dumps(man))

        def fake_extract(shard, dst):
            Path(dst).write_text(''.join(json.dumps({'record_id': r, 'text': f'text {r}'}) + '\n' for r in self.shards[shard]))
        self.m.extract = fake_extract
        self.m.load_prompt = lambda: {'instructions': 'q', 'criteria': {'keep': 'a', 'drop': 'b'}}
        self.base = base

    def tearDown(self):
        httpx.Client = REAL
        self.t.cleanup()

    def go(self, handler):
        seen = []

        def h(req):
            seen.append(json.loads(req.content))
            assert req.headers['authorization'] == f'Bearer {KEY}'
            return handler(len(seen))
        httpx.Client = lambda **kw: REAL(transport=httpx.MockTransport(h))
        self.m.STOP.clear(); self.m.PAUSE_REASON.clear()
        self.m.run()
        return seen, json.loads((self.base / 'res' / 'run-status.json').read_text())

    def results(self):
        return [json.loads(x) for p in (self.base / 'cache' / 'results').glob('*.jsonl') for x in p.read_text().splitlines()]

    def test_persist_decide_and_resume_without_repaying(self):
        seen, st = self.go(lambda n: ok(0.5 if n == 1 else 0.3))
        self.assertEqual(len(seen), 9)
        self.assertEqual(set(seen[0]), {'model', 'state', 'questions'})
        res = self.results()
        self.assertEqual(len(res), 9)
        self.assertEqual(sum(r['decision'] == 'keep' for r in res), 1)  # p = 0.5 counts as keep
        seen2, st2 = self.go(lambda n: ok())
        self.assertEqual(len(seen2), 0)
        self.assertEqual(st2['records_done'], 9)

    def test_429_retry_and_slowdown(self):
        seen, st = self.go(lambda n: httpx.Response(429, headers={'retry-after': '1'}, text='slow') if n == 1 else ok())
        self.assertEqual(len(self.results()), 9)
        self.assertEqual(st['http_429'], 1)
        self.assertEqual(st['retries'], 1)

    def test_persistent_failure_is_error_not_drop(self):
        seen, st = self.go(lambda n: httpx.Response(500, text='boom') if n <= 3 else ok())
        res = self.results()
        self.assertEqual(len(res), 8)
        errs = [json.loads(x) for p in (self.base / 'cache' / 'errors').glob('*.jsonl') for x in p.read_text().splitlines()]
        self.assertEqual(len(errs), 1)
        self.assertNotIn(errs[0]['record_id'], {r['record_id'] for r in res})
        self.assertTrue(st['phase'].startswith('incomplete'))
        seen2, st2 = self.go(lambda n: ok())  # restart retries the failed record only
        self.assertEqual(len(seen2), 1)
        self.assertEqual(st2['phase'], 'completed')

    def test_budget_cap_pauses(self):
        self.m.MAX_USD = 1e-9
        seen, st = self.go(lambda n: ok())
        self.assertEqual(len(seen), 0)
        self.assertEqual(st['phase'], 'paused')
        self.assertIn('budget', st['pause_reason'])

    def test_auth_error_stops_and_key_not_written(self):
        seen, st = self.go(lambda n: httpx.Response(401, text=f'bad {KEY}'))
        self.assertEqual(st['phase'], 'paused')
        self.assertLessEqual(len(seen), 8)
        for p in self.base.rglob('*'):
            if p.is_file():
                self.assertNotIn(KEY, p.read_text())

    def test_error_storm_pauses_without_deadlock(self):
        self.m.STORM_CONSECUTIVE = 3
        import threading
        done = threading.Event()
        box = {}
        def target():
            box['r'] = self.go(lambda n: httpx.Response(500, text='boom'))
            done.set()
        th = threading.Thread(target=target, daemon=True); th.start()
        self.assertTrue(done.wait(20), 'deadlock: worker did not finish')
        seen, st = box['r']
        self.assertEqual(st['phase'], 'paused')
        self.assertIn('error storm', st['pause_reason'])
        self.assertEqual(len(self.results()), 0)

    def test_rebuild_result_from_logged_attempt(self):
        (self.base / 'cache' / 'attempts').mkdir(parents=True, exist_ok=True)
        (self.base / 'cache' / 'attempts' / 's1.jsonl').write_text(json.dumps({'record_id': 'reddit:t1_a0', 'attempt': 1, 'usd': 1e-6, 'unknown_usd': 0.0,
            'choice': 'drop', 'p_keep': 0.2, 'confidence': 0.8, 'input_tokens': 20, 'request_id': 'x', 'at': 't', 'error': None}) + '\n')
        seen, st = self.go(lambda n: ok())
        self.assertEqual(len(seen), 8)  # a0 is not paid again
        r = [x for x in self.results() if x['record_id'] == 'reddit:t1_a0']
        self.assertEqual(len(r), 1)
        self.assertTrue(r[0]['rebuilt_from_attempt'])

    def test_invalid_probability_is_error(self):
        bad = httpx.Response(200, json={'model': 'jev-1.13.0', 'answers': {'strict_keep': {'choice': 'keep', 'probabilities': {'keep': 1.5}}}, 'usage': {'input_tokens': 5}})
        seen, st = self.go(lambda n: bad)
        self.assertEqual(len(self.results()), 0)

    def test_stop_file(self):
        (self.base / 'cache' / 'STOP').write_text('x')
        seen, st = self.go(lambda n: ok())
        self.assertEqual(len(seen), 0)
        self.assertEqual(st['phase'], 'paused')


if __name__ == '__main__':
    unittest.main()
