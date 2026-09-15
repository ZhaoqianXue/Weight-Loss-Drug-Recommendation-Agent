"""Local, inspectable provenance for every explicit workflow operation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import uuid
import threading
from .settings import get_settings
from .provenance import dataset_digest

_ACTIVE_EVENTS = None
_LOCKS = threading.local()

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def git_state(root):
    def git(*args):
        result = subprocess.run(['git','-C',str(root),*args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain'))}

def hashes(paths):
    return {str(p):dataset_digest(p) for p in paths if Path(p).is_file()}

def record_event(kind, **fields):
    if _ACTIVE_EVENTS is not None:
        with _ACTIVE_EVENTS.open('a') as stream:
            stream.write(json.dumps({'at':now(),'kind':kind,**fields},ensure_ascii=False)+'\n')

@contextmanager
def run_lock():
    """One writer per run, including nested builds in a workflow on this thread."""
    lock = get_settings().path('results')/'.operation.lock'
    lock = lock.resolve()
    held = getattr(_LOCKS, 'held', set())
    if lock in held:
        yield
        return
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    _LOCKS.held = held | {lock}
    try:
        yield
    finally:
        _LOCKS.held = held
        lock.unlink()


def environment_reference(settings):
    """Store the installed environment once for each content fingerprint in a run."""
    payload = {
        'schema_version': 1,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'packages': {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']},
        'lock_sha256': hashes((settings.root/'requirements').glob('*.lock')),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = settings.path('results')/'environments'/f'{fingerprint}.json'
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Stored environment does not match its fingerprint')
    else:
        atomic_json(path, payload)
    return {'path': str(path.relative_to(settings.path('results'))), 'fingerprint': fingerprint, 'sha256': dataset_digest(path)}


@contextmanager
def operation(name, parameters=None):
    with run_lock():
        with _operation(name, parameters) as report:
            yield report


@contextmanager
def _operation(name, parameters=None):
    global _ACTIVE_EVENTS
    settings = get_settings()
    folder = settings.path('results')/'operations'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    previous = _ACTIVE_EVENTS
    _ACTIVE_EVENTS = folder/'events.jsonl'
    report = {'schema_version':2,'run_id':settings.config['run_id'],'operation':name,'started_at':now(),'status':'running'}
    try:
        inputs = [settings.raw,settings.collection_manifest,settings.path('extracted'),settings.path('standardized'),settings.path('catalog'),settings.path('terminology'),settings.path('terminology_embeddings')]
        inputs.extend(settings.path('annotation_seed_dir').glob('*'))
        sources = list((settings.root/'src/weightloss').rglob('*.py'))
        report = {'schema_version':2,'run_id':settings.config['run_id'],'operation':name,'started_at':now(),'status':'running','code':git_state(settings.root),'source_sha256':hashes(sources),'environment':environment_reference(settings),'config':settings.config,'config_sha256':dataset_digest(settings.config_path),'parameters':parameters or {},'input_sha256':hashes(inputs),'models':{'extraction':'gpt-4.1-nano','standardization':'gpt-4.1-nano','embedding':'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb','revision':'unknown; record provider response metadata when available','temperature':0.2,'seed':None,'retry_policy':'Provider client defaults; individual retry count is unavailable'},'limitations':['Model calls are not guaranteed to be byte-identical on repeat. Historical annotations have unknown original model revisions.']}
        if name not in ('extract','standardize','build-index','import-graph','chatbot'):
            report['models'] = {'mode':'no model called by this operation'}
        atomic_json(folder/'manifest.json',report)
        yield report
        report['status'] = 'completed'
    except BaseException as error:
        report['status'] = 'failed'
        # Do not serialize exception text that might contain an API payload/token.
        report['error_type'] = type(error).__name__
        raise
    finally:
        report['finished_at'] = now()
        try:
            report['output_sha256'] = hashes([settings.path('extracted'),settings.path('standardized'),settings.dataset_manifest])
        except OSError:
            report['output_sha256'] = {}
        try:
            from .pipeline.refresh_data import read
            from collections import Counter
            rows = read(settings.path('standardized'))
            report['coverage'] = {'rows':len(rows),'extraction_status':dict(Counter(r['Extraction Status'] for r in rows)),'standardization_status':dict(Counter(r.get('Standardization Status','unknown') for r in rows))}
        except (OSError, KeyError, ValueError):
            report['coverage'] = None
        try:
            atomic_json(folder/'manifest.json',report)
        finally:
            _ACTIVE_EVENTS = previous

def new_run(run_id, raw_dir=None):
    settings = get_settings()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Run ID must contain only letters, digits, hyphens or underscores (max 80).')
    root = settings.root
    config_path = root/'configs/runs'/f'{run_id}.json'
    targets = [root/'data/interim'/run_id, root/'data/processed'/run_id, root/'results'/run_id]
    if config_path.exists() or any(p.exists() for p in targets):
        raise ValueError('Run already exists; select a new ID.')
    config = dict(settings.config)
    config.update(run_id=run_id, frozen=False, extracted=f'data/interim/{run_id}/extracted_reviews_all.csv', standardized=f'data/processed/{run_id}/standardized_reviews_all.csv', results=f'results/{run_id}', indexes=f'data/indexes/{run_id}', web_build=f'results/{run_id}/demo')
    if raw_dir:
        raw = Path(raw_dir).expanduser().resolve()
        collection = json.loads((raw/'collection_manifest.json').read_text())
        if dataset_digest(raw/'webmd_all_reviews.csv') != collection['csv_sha256']:
            raise ValueError('New raw snapshot does not match its manifest')
        config.update(raw_dir=str(raw), snapshot_id=raw.name)
    for p in targets: p.mkdir(parents=True)
    shutil.copyfile(settings.path('extracted'),targets[0]/'extracted_reviews_all.csv')
    shutil.copyfile(settings.path('standardized'),targets[1]/'standardized_reviews_all.csv')
    shutil.copyfile(settings.dataset_manifest,targets[1]/'dataset_manifest.json')
    atomic_json(config_path,config)
    atomic_json(targets[2]/'manifest.json',{'schema_version':1,'run_id':run_id,'status':'working','created_at':now(),'parent_run':settings.config['run_id'],'parent_inputs_sha256':hashes([settings.path('extracted'),settings.path('standardized'),settings.dataset_manifest]),'code':git_state(root),'config_path':str(config_path.relative_to(root)),'note':'Run refresh before validation if selecting a different raw snapshot.'})
    return config_path

def freeze():
    from .pipeline.validate_data import validate
    settings = get_settings()
    settings.require_writable()
    validation = validate()
    payload = dict(settings.config, frozen=True)
    atomic_json(settings.path('results')/'frozen.json',{'frozen_at':now(),'validation':validation,'input_sha256':hashes([settings.raw,settings.path('extracted'),settings.path('standardized'),settings.dataset_manifest]),'config':payload,'code':git_state(settings.root)})
    atomic_json(settings.config_path,payload)
    get_settings.cache_clear()
