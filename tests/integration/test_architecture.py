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
        config.update(extracted='data/interim/test/extracted_reviews_all.csv',standardized='data/processed/test/standardized_reviews_all.csv',web_build='results/test/demo',results='results/test',indexes='data/indexes/test')
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
            self.assertEqual(payload['web_build'],'results/next-run/demo')
            self.assertEqual(payload['indexes'],'data/indexes/next-run')
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
            for relative in ('data/processed/fixture/standardized_reviews_all.csv','results/fixture/metrics.json','results/fixture/demo/static/standardized_reviews_all.csv'):
                self.assertEqual((a/relative).read_bytes(),(b/relative).read_bytes())
            self.assertTrue(json.loads((a/'configs/pipeline.json').read_text())['frozen'])
            build=json.loads((a/'results/fixture/demo/build_manifest.json').read_text())
            self.assertTrue(build['config']['frozen'])
            self.assertEqual(build['run_provenance_sha256'][str((a/'results/fixture/frozen.json').resolve())],dataset_digest(a/'results/fixture/frozen.json'))
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

    def test_stages_share_environment_and_new_environment_gets_new_file(self):
        from weightloss.runs import environment_reference
        from types import SimpleNamespace
        with isolated_run() as paths:
            with operation('first') as first:
                reference = dict(first['environment'])
            environment = paths.path('results')/reference['path']
            before = environment.stat().st_mtime_ns
            with operation('second') as second:
                self.assertEqual(second['environment'], reference)
            self.assertEqual(environment.stat().st_mtime_ns, before)
            self.assertEqual(dataset_digest(environment),reference['sha256'])
            with patch('weightloss.runs.distributions',return_value=[SimpleNamespace(metadata={'Name':'synthetic'},version='1')]):
                changed = environment_reference(paths)
            self.assertNotEqual(changed['fingerprint'],reference['fingerprint'])
            self.assertEqual(len(list(environment.parent.glob('*.json'))),2)

    def test_corrupt_environment_is_rejected_and_lock_released(self):
        with isolated_run() as paths:
            with operation('first') as report:
                environment=paths.path('results')/report['environment']['path']
            environment.write_text('{}')
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                with operation('second'):pass
            self.assertFalse((paths.path('results')/'.operation.lock').exists())
            self.assertEqual(environment.read_text(),'{}')

    def test_web_build_has_only_current_manifest_and_run_reference(self):
        with isolated_run() as paths:
            from weightloss.cli import main
            atomic_json(paths.path('results')/'manifest.json',{'run_id':'test'})
            with contextlib.redirect_stdout(io.StringIO()):
                main(['build-web']);main(['build-web'])
            report=json.loads((paths.path('web_build')/'build_manifest.json').read_text())
            self.assertEqual(report['run_id'],'test')
            self.assertEqual(report['dataset_sha256'],dataset_digest())
            self.assertEqual(report['config'],paths.config)
            self.assertIn(str(paths.path('results')/'manifest.json'),report['run_provenance_sha256'])
            self.assertFalse((paths.path('results')/'operations').exists())
            self.assertFalse((paths.path('results')/'environments').exists())

    def test_web_and_operation_share_cross_process_write_lock(self):
        from weightloss.pipeline.build_web import build_web
        with isolated_run() as paths:
            with operation('outer'):
                # A nested build on the owning thread is valid, but a second writer is not.
                build_web()
                before=dataset_digest(paths.path('web_build')/'build_manifest.json')
                result=subprocess.run([sys.executable,'-m','weightloss','build-web'],capture_output=True,text=True)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertTrue((paths.path('results')/'.operation.lock').exists())
                self.assertEqual(dataset_digest(paths.path('web_build')/'build_manifest.json'),before)
            self.assertFalse((paths.path('results')/'.operation.lock').exists())

    def test_extraction_resumes_after_interruption_without_repeating_completed_row(self):
        from types import SimpleNamespace
        from weightloss.extraction.extract_all import main
        from weightloss.pipeline.refresh_data import read
        from unittest.mock import Mock
        with isolated_run() as paths:
            pending=[r for r in read(paths.path('extracted')) if r['Extraction Status']=='pending' and r['Textual Review'].strip()]
            response={'structured_info':{'side_effects':[]},'relations':[]}
            extractor=Mock(side_effect=[response,RuntimeError('synthetic interruption')])
            stub=SimpleNamespace(safe_extract_structured=extractor)
            with patch.dict(sys.modules,{'weightloss.extraction.schema_extraction':stub}), patch.dict(os.environ,{'OPENAI_API_KEY':'synthetic'}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    with operation('extract'):main(['--limit','2'])
                rows={r['Review ID']:r for r in read(paths.path('extracted'))}
                self.assertEqual(rows[pending[0]['Review ID']]['Extraction Status'],'completed')
                self.assertEqual(rows[pending[1]['Review ID']]['Extraction Status'],'pending')
                extractor.reset_mock(side_effect=True);extractor.return_value=response
                with operation('extract'):main(['--limit','1'])
                self.assertEqual(extractor.call_count,1)
                self.assertEqual(extractor.call_args.kwargs['text'],pending[1]['Textual Review'])
                self.assertFalse((paths.path('results')/'.operation.lock').exists())

if __name__=='__main__':unittest.main()
