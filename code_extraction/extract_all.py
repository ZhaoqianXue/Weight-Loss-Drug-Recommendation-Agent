"""Incrementally extract pending current reviews, checkpointing every result."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code_pipeline.refresh_data import ROOT, read, write

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=None,help='Maximum pending reviews this run')
    args=parser.parse_args()
    path=ROOT/'data_extracted/extracted_reviews_all.csv'
    rows=read(path)
    pending=[r for r in rows if r.get('Extraction Status') in ('pending','failed') and r['Textual Review'].strip()]
    no_text=sum(r.get('Extraction Status') in ('pending','failed') and not r['Textual Review'].strip() for r in rows)
    print('%d unannotated rating-only records have no narrative and remain unknown' % no_text)
    print('%d pending reviews' % len(pending))
    if not pending: return
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is required. No data changed. Run code_pipeline/refresh_data.py first to prepare current rows.')
    from code_extraction.schema_extraction import safe_extract_structured
    for row in pending[:args.limit]:
        result=safe_extract_structured(text=row['Textual Review'],drug_name=row['Brand Name'],condition_from_csv=row['Condition'],medication_duration=row['Medication Duration'])
        if result:
            row['structured_info']=json.dumps(result['structured_info'],ensure_ascii=False)
            row['relations']=json.dumps(result['relations'],ensure_ascii=False)
            row['Extraction Status']='completed'
            row['Annotation Method']='gpt-4.1-nano structured extraction'
        else: row['Extraction Status']='failed'
        write(path,rows)
    remaining=sum(r.get('Extraction Status') in ('pending','failed') for r in rows)
    print('%d reviews remain pending or failed. Run standardize_all.py to publish completed annotations.' % remaining)

if __name__=='__main__': main()
