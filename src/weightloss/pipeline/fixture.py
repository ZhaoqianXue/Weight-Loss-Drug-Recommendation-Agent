"""Offline pipeline rehearsal using synthetic reviews and frozen annotation responses."""
import json
import os
from pathlib import Path
import shutil
from weightloss.settings import get_settings, configure
from weightloss.provenance import dataset_digest
from weightloss.runs import atomic_json, operation, freeze
from .refresh_data import write, main as refresh
from .validate_data import validate
from .build_web import build_web
from weightloss.evaluation.standardization_stat import summarize

def reproduce(destination):
    origin = get_settings()
    if destination.exists():
        raise ValueError('Fixture output already exists; choose a fresh directory.')
    payload = json.loads((origin.root/'tests/fixtures/reviews.json').read_text())
    destination.mkdir(parents=True)
    raw_dir = destination/'data/raw/synthetic'
    seed = destination/'data/external/annotation_seeds'
    extracted = destination/'data/interim/fixture/extracted_reviews_all.csv'
    standardized = destination/'data/processed/fixture/standardized_reviews_all.csv'
    write(raw_dir/'webmd_all_reviews.csv',payload['raw'])
    for brand in sorted({r['Brand Name'] for r in payload['raw']}):
        write(raw_dir/f'webmd_{brand.lower()}_reviews.csv',[r for r in payload['raw'] if r['Brand Name']==brand])
    atomic_json(raw_dir/'collection_manifest.json',{'completed_at':'2000-01-01T00:00:00+00:00','total_reviews':len(payload['raw']),'csv_sha256':dataset_digest(raw_dir/'webmd_all_reviews.csv'),'brands':[],'synthetic':True})
    write(seed/'extracted_reviews_all.csv',payload['annotations_extracted'])
    write(seed/'standardized_reviews_all.csv',payload['annotations_standardized'])
    write(extracted,payload['annotations_extracted'])
    write(standardized,payload['annotations_standardized'])
    catalog = destination/'configs/drugs.json'
    catalog.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(origin.path('catalog'),catalog)
    config = dict(origin.config,project_root=str(destination),run_id='fixture',snapshot_id='synthetic',frozen=False,raw_dir='data/raw/synthetic',annotation_seed_dir='data/external/annotation_seeds',extracted='data/interim/fixture/extracted_reviews_all.csv',standardized='data/processed/fixture/standardized_reviews_all.csv',catalog='configs/drugs.json',web_source=str(origin.path('web_source')),web_build='results/fixture/demo',results='results/fixture',indexes='data/indexes/fixture',terminology='data/external/unused.csv',terminology_embeddings='data/external/unused-embeddings.csv')
    config_path=destination/'configs/pipeline.json'
    atomic_json(config_path,config)
    old = {key:os.environ.get(key) for key in ('WEIGHTLOSS_ROOT','WEIGHTLOSS_CONFIG')}
    try:
        configure(destination,config_path)
        with operation('reproduce-fixture',{'mode':'synthetic frozen-response replay','fixture_sha256':dataset_digest(origin.root/'tests/fixtures/reviews.json')}) as report:
            # Include the source checkout because fixture data intentionally live elsewhere.
            report['source_checkout'] = str(origin.root)
            report['source_sha256'] = {str(p.relative_to(origin.root)):dataset_digest(p) for p in sorted((origin.root/'src/weightloss').rglob('*.py'))}
            refresh()
            check=validate()
            from .refresh_data import read
            rows=read(standardized)
            metrics=summarize(rows)
            if metrics != payload['expected_metrics']:
                raise ValueError('Synthetic fixture differs from hand-specified expected metrics')
            atomic_json(destination/'results/fixture/metrics.json',metrics)
            report['validation']=check
            report['models']={'mode':'offline replay; no model called'}
            freeze()
            build_web()
            validate(include_web=True)
        return destination/'results/fixture'
    finally:
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        get_settings.cache_clear()
