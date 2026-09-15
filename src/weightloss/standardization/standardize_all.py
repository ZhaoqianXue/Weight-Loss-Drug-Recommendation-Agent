"""Standardize newly extracted records; preserve historical and pending rows."""
import argparse
import json
import os
import sys
from pathlib import Path
from weightloss.pipeline.refresh_data import read, write, date, write_dataset_manifest
from weightloss.settings import get_settings
from weightloss.runs import record_event

def main(argv=None):
    paths = get_settings()
    paths.require_writable()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args(argv)
    if args.limit is not None and args.limit < 0: parser.error("--limit must be nonnegative")
    ex=read(paths.path('extracted'))
    path=paths.path('standardized')
    rows=read(path); by_id={r['Review ID']:r for r in rows}
    pending=[r for r in ex if r.get('Extraction Status')=='completed' and by_id[r['Review ID']].get('Standardization Status') not in ('completed','historical_reused')]
    print('%d extracted reviews await standardization' % len(pending))
    if not pending: return
    if not os.environ.get('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY required; no data changed.')
    import pandas as pd
    from weightloss.standardization import baseline as module
    texts, embeddings=module.prepare_ae_data(pd.read_csv(paths.path('terminology_embeddings')))
    for row in pending[:args.limit]:
        result=module.process_row(pd.Series(row),module.get_model(),texts,embeddings)
        json.loads(result['standardized_info']); json.loads(result['standardized_relations'])
        target=by_id[row['Review ID']]
        target.update({k:v for k,v in row.items() if k not in ('structured_info','relations')})
        target.update(result);target['Date']=date(target['Date']);target['Standardization Status']='completed'
        write(path,rows)
        record_event('standardization', review_id=row['Review ID'], status='completed')
    write_dataset_manifest(rows)

if __name__=='__main__': main()
