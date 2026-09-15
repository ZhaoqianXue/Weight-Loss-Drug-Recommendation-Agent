"""Build a self-contained static preview from one validated processed snapshot."""
import json
from pathlib import Path
import shutil
import tempfile
from weightloss.settings import get_settings
from weightloss.provenance import dataset_digest
from .validate_data import validate

def build_web():
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
        (staging/'build_manifest.json').write_text(json.dumps({'run_id':paths.config['run_id'],'dataset_sha256':dataset_digest(),'source_sha256':{str(p.relative_to(source)):dataset_digest(p) for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.html','.js')}},indent=2)+'\n')
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
