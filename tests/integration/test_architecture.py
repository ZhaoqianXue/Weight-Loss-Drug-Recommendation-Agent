"""Regression tests for data immutability, installation, provenance and web packaging."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from weightloss.settings import get_settings, configure
from weightloss.provenance import dataset_digest
from weightloss.runs import atomic_json, operation, new_run, freeze

ROOT=Path(__file__).resolve().parents[2]

@contextlib.contextmanager
def isolated_run():
    original=get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        config=dict(original.config, frozen=False, run_id='test')
        for key in ('raw_dir','annotation_seed_dir','catalog','terminology','terminology_embeddings','web_source'):
            config[key]=str(original.path(key))
        config.update(extracted='data/interim/test/extracted_reviews_all.csv',standardized='data/processed/test/standardized_reviews_all.csv',web_build='artifacts/web',results='results/test',indexes='artifacts/indexes/test')
        for key in ('extracted','standardized'):
            dest=root/config[key];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original.path(key),dest)
        shutil.copyfile(original.dataset_manifest,(root/config['standardized']).parent/'dataset_manifest.json')
        conf=root/'configs/pipeline.json';atomic_json(conf,config)
        with patch.dict(os.environ,{'WEIGHTLOSS_ROOT':str(root),'WEIGHTLOSS_CONFIG':str(conf)}):
            get_settings.cache_clear()
            try:yield get_settings()
            finally:get_settings.cache_clear()
    get_settings.cache_clear()

class ArchitectureTests(unittest.TestCase):
    def test_core_imports_do_not_read_data_or_create_clients(self):
        code='''
from pathlib import Path
from unittest.mock import patch
with patch.object(Path, 'read_text', side_effect=AssertionError('read_text at import')), patch.object(Path, 'read_bytes', side_effect=AssertionError('read_bytes at import')):
 import weightloss.settings, weightloss.catalog, weightloss.provenance
 import weightloss.cli, weightloss.runs
 import weightloss.pipeline.refresh_data, weightloss.pipeline.validate_data
 import weightloss.pipeline.build_web, weightloss.pipeline.fixture
 import weightloss.extraction.extract_all, weightloss.standardization.standardize_all
'''
        subprocess.run([sys.executable,'-c',code],cwd=tempfile.gettempdir(),check=True)

    def test_cli_runs_outside_checkout(self):
        result=subprocess.run([sys.executable,'-m','weightloss','--root',str(ROOT),'validate'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['reviews'],2727)

    def test_frozen_snapshot_rejects_writes_without_changes(self):
        paths=get_settings();before=dataset_digest()
        result=subprocess.run([sys.executable,'-m','weightloss','refresh'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('frozen',result.stderr)
        self.assertEqual(dataset_digest(),before)

    def test_refresh_from_migrated_seeds_matches_frozen_bytes(self):
        from weightloss.pipeline.refresh_data import main
        with isolated_run() as paths:
            before=[dataset_digest(paths.path(k)) for k in ('extracted','standardized')]
            with contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual([dataset_digest(paths.path(k)) for k in ('extracted','standardized')],before)

    def test_new_run_is_separate_and_cannot_overwrite(self):
        with isolated_run() as paths:
            before=dataset_digest()
            conf=new_run('next-run')
            payload=json.loads(conf.read_text())
            self.assertFalse(payload['frozen'])
            self.assertEqual(dataset_digest(paths.root/payload['standardized']),before)
            self.assertNotEqual(paths.root/payload['standardized'],paths.path('standardized'))
            with self.assertRaises(ValueError):new_run('next-run')
            with self.assertRaises(ValueError):new_run('../escape')

    def test_failed_operation_records_failure_and_releases_lock(self):
        with isolated_run() as paths:
            with self.assertRaises(RuntimeError):
                with operation('intentional-test-failure'):
                    raise RuntimeError('test secret should not be serialized')
            report=json.loads(next((paths.path('results')/'operations').glob('*/manifest.json')).read_text())
            self.assertEqual(report['status'],'failed')
            self.assertNotIn('test secret',json.dumps(report))
            self.assertFalse((paths.path('results')/'.operation.lock').exists())
            self.assertTrue(report['input_sha256'])

    def test_freeze_validates_and_prevents_future_writes(self):
        with isolated_run() as paths:
            freeze()
            self.assertTrue(get_settings().config['frozen'])
            with self.assertRaises(ValueError):get_settings().require_writable()

    def test_corrupt_snapshot_rejected_before_web_build(self):
        from weightloss.pipeline.build_web import build_web
        with isolated_run() as paths:
            with paths.path('standardized').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'digest'):build_web()
            self.assertFalse(paths.path('web_build').exists())

    def test_synthetic_reproduction_is_repeatable_and_frozen(self):
        from weightloss.pipeline.fixture import reproduce
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'first',Path(tmp)/'second'
            with contextlib.redirect_stdout(io.StringIO()):
                reproduce(a);reproduce(b)
            for relative in ('data/processed/fixture/standardized_reviews_all.csv','results/fixture/metrics.json','artifacts/web/static/standardized_reviews_all.csv'):
                self.assertEqual((a/relative).read_bytes(),(b/relative).read_bytes())
            self.assertTrue(json.loads((a/'configs/pipeline.json').read_text())['frozen'])
            with self.assertRaises(ValueError):reproduce(a)

    def test_http_preview_resources(self):
        from weightloss.pipeline.build_web import build_web
        with isolated_run() as paths:
            build_web()
            class QuietHandler(SimpleHTTPRequestHandler):
                def log_message(self,*args):pass
            server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(paths.path('web_build'))))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for name in ('index.html','templates/knowledge_graph.html','static/review-assistant.js','static/standardized_reviews_all.csv','static/drugs.json','static/dataset_manifest.json'):
                    with urlopen(f'http://127.0.0.1:{server.server_port}/{name}') as response:
                        self.assertEqual(response.status,200)
                        self.assertTrue(response.read())
            finally:server.shutdown();server.server_close();thread.join()

    def test_flask_static_and_unconfigured_backend(self):
        import runpy
        with isolated_run() as paths:
            from weightloss.pipeline.build_web import build_web
            build_web()
            with patch.dict(os.environ,{'OPENAI_API_KEY':''}):
                app=runpy.run_path(str(ROOT/'apps/web/flask_app.py'))['app']
                client=app.test_client()
                for url in ('/','/static/standardized_reviews_all.csv','/static/review-assistant.js'):
                    response=client.get(url)
                    self.assertEqual(response.status_code,200);response.close()
                self.assertEqual(client.post('/chat',json={}).status_code,400)
                self.assertEqual(client.post('/chat',json={'message':'hello'}).status_code,503)

    def test_collector_failure_creates_no_partial_snapshot(self):
        from weightloss.ingestion import batch_scraper
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'new-snapshot'
            with patch.object(batch_scraper,'collect',side_effect=ValueError('failed page')):
                with self.assertRaises(ValueError):batch_scraper.main(['--output-dir',str(destination)])
            self.assertFalse(destination.exists())

if __name__=='__main__':unittest.main()
