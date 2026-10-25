"""Validate a selected snapshot before publication or retrieval."""
from collections import Counter
import json
from weightloss.settings import get_settings
from weightloss.catalog import load_catalog
from weightloss.provenance import dataset_digest
from .refresh_data import read, date

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate(include_web=False):
    paths = get_settings()
    raw = read(paths.raw)
    collection = json.loads(paths.collection_manifest.read_text())
    drugs = load_catalog()
    expected = {d['brand']:d for d in drugs}
    ids = [r['Review ID'] for r in raw]
    require(len(raw) == collection['total_reviews'], 'Collection count mismatch')
    require(len(set(ids)) == len(ids), 'Duplicate review IDs')
    require(set(r['Brand Name'] for r in raw) == set(expected), 'Catalog coverage mismatch')
    require(all(r['Drug Name'] == expected[r['Brand Name']]['generic'] for r in raw), 'Generic/brand mismatch')
    require(dataset_digest(paths.raw) == collection['csv_sha256'], 'Raw digest mismatch')
    require(all(1 <= float(r[f]) <= 5 for r in raw for f in ('Overall Rating','Effectiveness','Ease of Use','Satisfaction')), 'Invalid rating')
    require(all('\ufffd' not in r['Textual Review'] for r in raw), 'Corrupted Unicode')
    files = {}
    for key in ('extracted', 'standardized'):
        path = paths.path(key)
        rows = read(path)
        require([r['Review ID'] for r in rows] == ids, f'{key}: IDs/order mismatch')
        for source, row in zip(raw, rows):
            for field, value in source.items():
                expected_value = date(value) if field == 'Date' and 'standardized_info' in row else value
                require(row[field] == expected_value, f'{key}: source field {field} changed')
            for field in ('structured_info','relations','standardized_info','standardized_relations'):
                if field in row:
                    json.loads(row[field])
        files[key] = {'rows':len(rows),'sha256':dataset_digest(path)}
    standardized = read(paths.path('standardized'))
    dataset = json.loads(paths.dataset_manifest.read_text())
    require(dataset['total_reviews'] == len(raw), 'Dataset count mismatch')
    require(dataset['generics'] == len({d['generic'] for d in drugs}), 'Generic count mismatch')
    require({b['brand']:b['reviews'] for b in dataset['brands']} == dict(Counter(r['Brand Name'] for r in raw)), 'Brand counts mismatch')
    require(dataset['raw_csv_sha256'] == collection['csv_sha256'], 'Dataset source mismatch')
    require(dataset['standardized_csv_sha256'] == files['standardized']['sha256'], 'Standardized digest mismatch')
    for row in standardized:
        if row['Extraction Status'] == 'pending':
            require(json.loads(row['standardized_info'])['side_effects'] == [], 'Pending row has invented effects')
    if include_web:
        for source, name in [(paths.path('standardized'),'standardized_reviews_all.csv'),(paths.dataset_manifest,'dataset_manifest.json'),(paths.path('catalog'),'drugs.json')]:
            require(source.read_bytes() == (paths.path('web_build')/'static'/name).read_bytes(), f'Stale web resource: {name}')
    return {'status':'passed','reviews':len(raw),'generics':len({d['generic'] for d in drugs}),'brands':len(drugs),'text_reviews':sum(bool(r['Textual Review']) for r in raw),'extraction_status':dict(Counter(r['Extraction Status'] for r in standardized)),'files':files}

def main():
    print(json.dumps(validate(), indent=2))
