"""Standardize newly extracted records; preserve historical and pending rows."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code_pipeline.refresh_data import ROOT, read, write, date, publish_website

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=None)
    args=parser.parse_args()
    ex=read(ROOT/'data_extracted/extracted_reviews_all.csv')
    path=ROOT/'data_standardized/standardized_reviews_all.csv'
    rows=read(path); by_id={r['Review ID']:r for r in rows}
    pending=[r for r in ex if r.get('Extraction Status')=='completed' and by_id[r['Review ID']].get('Standardization Status') not in ('completed','historical_reused')]
    print('%d extracted reviews await standardization' % len(pending))
    if not pending: return
    if not os.environ.get('OPENAI_API_KEY'): raise SystemExit('OPENAI_API_KEY required; no data changed.')
    import pandas as pd
    from code_standardization import standardization as module
    texts, embeddings=module.prepare_ae_data(module.merged_df_embedded)
    for row in pending[:args.limit]:
        result=module.process_row(pd.Series(row),module.model,texts,embeddings)
        json.loads(result['standardized_info']); json.loads(result['standardized_relations'])
        target=by_id[row['Review ID']]
        target.update({k:v for k,v in row.items() if k not in ('structured_info','relations')})
        target.update(result);target['Date']=date(target['Date']);target['Standardization Status']='completed'
        write(path,rows)
    publish_website(rows)

if __name__=='__main__': main()
