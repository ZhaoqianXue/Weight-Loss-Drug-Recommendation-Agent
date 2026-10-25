"""Package one validated processed snapshot as the website backend's data (the frontend is built by apps/web)."""
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
        (staging/'static').mkdir(parents=True)
        for original, name in [(paths.path('standardized'),'standardized_reviews_all.csv'),(paths.path('catalog'),'drugs.json'),(paths.dataset_manifest,'dataset_manifest.json')]:
            shutil.copyfile(original, staging/'static'/name)
        manifest = {
            'schema_version': 3,
            'run_id': paths.config['run_id'],
            'run_results': str(paths.path('results')),
            'run_provenance_sha256': hashes([
                paths.path('results')/'manifest.json', paths.path('results')/'frozen.json',
            ]),
            'config': paths.config,
            'config_sha256': dataset_digest(paths.config_path),
            'dataset_path': str(paths.path('standardized')),
            'dataset_sha256': dataset_digest(),
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
