"""Shared dataset digest used to reject stale retrieval indexes."""
import hashlib
import json
from pathlib import Path
from drug_catalog import ROOT
DATA_PATH = ROOT / 'data_standardized/standardized_reviews_all.csv'

def dataset_digest(path=DATA_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def index_metadata(model, path=DATA_PATH):
    return {'dataset_sha256':dataset_digest(path),'embedding_model':model}

def verify_index(directory, model, path=DATA_PATH):
    manifest=Path(directory)/'dataset_manifest.json'
    if not manifest.exists() or json.loads(manifest.read_text()) != index_metadata(model,path):
        raise ValueError('Missing or stale index provenance. Rebuild with python3 code_chatbot/table_loader.py')
