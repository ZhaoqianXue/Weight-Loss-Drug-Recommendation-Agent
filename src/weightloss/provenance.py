"""Dataset fingerprints shared by index builders and consumers."""
import hashlib
import json
from pathlib import Path
from .settings import get_settings

def dataset_digest(path=None):
    return hashlib.sha256(Path(path or get_settings().path('standardized')).read_bytes()).hexdigest()

def index_metadata(model, path=None):
    return {'dataset_sha256': dataset_digest(path), 'embedding_model': model}

def verify_index(directory, model, path=None):
    manifest = Path(directory)/'dataset_manifest.json'
    if not manifest.exists() or json.loads(manifest.read_text()) != index_metadata(model, path):
        raise ValueError('Missing or stale index provenance. Rebuild with weightloss build-index.')
