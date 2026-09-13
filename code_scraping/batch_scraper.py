"""Collect every catalog brand; publish only after all brands validate."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from drug_catalog import DRUGS, ROOT
from code_scraping.scraper import collect, write_csv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT / '.cache/webmd' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--output-dir',type=Path,default=ROOT / 'data_webmd')
    args = parser.parse_args()
    batches, summaries = [], []
    # Three independent brand streams; pagination remains sequential within a brand.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(collect, drug, args.run_dir) for drug in DRUGS]
        for future in futures:
            rows, summary = future.result()
            batches.append(rows); summaries.append(summary)
    all_rows = [r for batch in batches for r in batch]
    if len({r['Review ID'] for r in all_rows}) != len(all_rows):
        raise ValueError('Cross-brand duplicate IDs: investigate shared review pages before publishing')
    for drug, rows in zip(DRUGS,batches):
        write_csv(args.output_dir / ('webmd_' + drug['brand'].lower() + '_reviews.csv'),rows)
    target = args.output_dir / 'webmd_all_reviews.csv'
    write_csv(target,all_rows)
    manifest = {'completed_at':datetime.now(timezone.utc).isoformat(),'total_reviews':len(all_rows),'brands':summaries,'csv_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'rendered_records':sum(r['Source Visibility']=='rendered' for r in all_rows),'embedded_only_records':sum(r['Source Visibility']=='embedded_only' for r in all_rows),'text_records':sum(bool(r['Textual Review']) for r in all_rows),'count_unit':'Unique WebMD review ID, including rating-only records; aliases are not added separately.'}
    (args.output_dir / 'collection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published %d validated reviews' % len(all_rows),flush=True)

if __name__ == '__main__': main()
