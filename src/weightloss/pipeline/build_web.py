"""Build a self-contained static preview from one validated processed snapshot."""
import json
from pathlib import Path
import shutil
import tempfile
from weightloss.settings import get_settings
from weightloss.provenance import dataset_digest
from weightloss.runs import run_lock, hashes
from .validate_data import validate

def build_web():
    with run_lock():
        return _build_web()


def _build_web():
    paths = get_settings()
    validate()
    source, destination = paths.path('web_source'), paths.path('web_build')
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError('Web output must be separate from application source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.web-', dir=destination.parent) as tmp:
        staging = Path(tmp)/'site'
        staging.mkdir()
        for folder in ('templates', 'static'):
            shutil.copytree(source/folder, staging/folder)
        for original, name in [(paths.path('standardized'),'standardized_reviews_all.csv'),(paths.path('catalog'),'drugs.json'),(paths.dataset_manifest,'dataset_manifest.json')]:
            shutil.copyfile(original, staging/'static'/name)
        # A root landing page redirects to the unchanged template, preserving relative URLs.
        (staging/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=templates/knowledge_graph.html"><a href="templates/knowledge_graph.html">Medication Experience Assistant</a>\n')
        manifest = {
            'schema_version': 2,
            'run_id': paths.config['run_id'],
            'run_results': str(paths.path('results')),
            'run_provenance_sha256': hashes([
                paths.path('results')/'manifest.json', paths.path('results')/'frozen.json',
            ]),
            'config': paths.config,
            'config_sha256': dataset_digest(paths.config_path),
            'dataset_path': str(paths.path('standardized')),
            'dataset_sha256': dataset_digest(),
            'source_sha256': {
                str(p.relative_to(source)): dataset_digest(p)
                for p in sorted(source.rglob('*'))
                if p.is_file() and p.suffix in ('.html', '.js')
            },
        }
        (staging/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        old = Path(tmp)/'previous'
        if destination.exists():
            destination.rename(old)
        try:
            staging.rename(destination)
        except BaseException:
            if old.exists(): old.rename(destination)
            raise
    validate(include_web=True)
    return destination
