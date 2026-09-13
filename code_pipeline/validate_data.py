"""Validate the published snapshot and emit a concise machine-readable audit."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drug_catalog import ROOT, DRUGS
from code_pipeline.refresh_data import read, date

def main():
    raw=read(ROOT/'data_webmd/webmd_all_reviews.csv')
    collection=json.loads((ROOT/'data_webmd/collection_manifest.json').read_text())
    expected={d['brand']:d for d in DRUGS}
    assert len(raw)==collection['total_reviews']
    assert len({r['Review ID'] for r in raw})==len(raw)
    assert set(r['Brand Name'] for r in raw)==set(expected)
    assert all(r['Drug Name']==expected[r['Brand Name']]['generic'] for r in raw)
    assert hashlib.sha256((ROOT/'data_webmd/webmd_all_reviews.csv').read_bytes()).hexdigest()==collection['csv_sha256']
    assert all(1 <= float(row[field]) <= 5 for row in raw for field in ('Overall Rating','Effectiveness','Ease of Use','Satisfaction'))
    assert all('\ufffd' not in row['Textual Review'] for row in raw)
    ids=[r['Review ID'] for r in raw]
    files={}
    for name in ['data_extracted/extracted_reviews_all.csv','data_standardized/standardized_reviews_all.csv','code_website/data/standardized_reviews_all.csv','code_website/static/standardized_reviews_all.csv']:
        path=ROOT/name;rows=read(path)
        assert [r['Review ID'] for r in rows]==ids
        for source, row in zip(raw, rows):
            for field, value in source.items():
                expected_value = date(value) if field == 'Date' and 'standardized_info' in row else value
                assert row[field] == expected_value, (name, row['Review ID'], field)
        for row in rows:
            for field in ('structured_info','relations','standardized_info','standardized_relations'):
                if field in row: json.loads(row[field])
        files[name]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(set(files[p]['sha256'] for p in files if not p.startswith('data_extracted')))==1
    standardized=read(ROOT/'data_standardized/standardized_reviews_all.csv')
    assert (ROOT/'config/drugs.json').read_bytes()==(ROOT/'code_website/static/drugs.json').read_bytes()
    dataset_path=ROOT/'data_standardized/dataset_manifest.json'
    assert dataset_path.read_bytes()==(ROOT/'code_website/static/dataset_manifest.json').read_bytes()
    dataset=json.loads(dataset_path.read_text())
    assert dataset['total_reviews']==len(raw)
    assert dataset['generics']==len({d['generic'] for d in DRUGS})
    assert {b['brand']:b['reviews'] for b in dataset['brands']}==dict(Counter(r['Brand Name'] for r in raw))
    assert dataset['raw_csv_sha256']==collection['csv_sha256']
    assert dataset['standardized_csv_sha256']==files['data_standardized/standardized_reviews_all.csv']['sha256']
    status=Counter(r['Extraction Status'] for r in standardized)
    print(json.dumps({'status':'passed','reviews':len(raw),'generics':len({d['generic'] for d in DRUGS}),'brands':len(DRUGS),'pages':sum(b['pages'] for b in collection['brands']),'text_reviews':sum(bool(r['Textual Review']) for r in raw),'source_visibility':dict(Counter(r['Source Visibility'] for r in raw)),'extraction_status':dict(status),'pending_with_text':sum(r['Extraction Status']=='pending' and bool(r['Textual Review']) for r in standardized),'files':files},indent=2))

if __name__=='__main__': main()
