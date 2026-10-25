"""Unified v1/v2/v3 CLI; immutable rules and existing checkpoints are preserved."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from weightloss.cleaning import reddit_baseline_columnar as baseline
from weightloss.cleaning import reddit_fast as evidence
from weightloss.cleaning.reddit import FastTextLanguage
from weightloss.cleaning.reddit_refinement import TextSegments
from weightloss.cleaning.reddit_runtime import worker_pool, progress
from weightloss.ingestion.reddit_production import write_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['v1','v2','v3'])
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report-file',type=Path,required=True)
    p.add_argument('--language-model',type=Path,default=Path('.cache/models/lid.176.bin'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--max-records',type=int)
    p.add_argument('--policy',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/reddit.json'))
    p.add_argument('--subreddits',nargs='+')
    a=p.parse_args(argv)
    if a.subreddits and a.stage!='v1':p.error('--subreddits applies only to v1')
    policies={'v1':'reddit-cleaning.json','v2':'reddit-refinement.json','v3':'reddit-evidence-final.json'}
    policy=json.loads((a.policy or Path('configs')/policies[a.stage]).read_text())
    progress(a.stage,'stage_started',workers=a.workers)
    if a.stage=='v1':
        with worker_pool(a.workers,baseline.initialize_worker,(str(a.language_model.resolve()),)) as pool:
            result=baseline.clean_production(a.source_dir,a.output_dir,a.report_file,policy,
                json.loads(a.config.read_text()),FastTextLanguage(a.language_model),
                a.subreddits,a.max_records,pool=pool)
    else:
        with worker_pool(a.workers,evidence.initialize_worker,
                (a.stage,str(a.source_dir.resolve()/'cleaned.sqlite'),str(a.language_model.resolve()),policy)) as pool:
            evidence._POOL=pool
            if a.stage=='v2':result=evidence.clean_strict(a.source_dir,a.output_dir,a.report_file,policy,FastTextLanguage(a.language_model),TextSegments(),a.max_records)
            else:result=evidence.run(a.source_dir,a.output_dir,a.report_file,policy,a.max_records)
        if a.stage=='v3' and a.max_records is None:result=evidence.release(a.output_dir,a.report_file)
    # Execution provenance is separate from immutable scientific rule fingerprints.
    names=['reddit_unified.py','reddit_runtime.py','reddit_columnar.py','reddit_baseline_columnar.py','reddit_fast.py']
    execution={'schema_version':1,'stage':a.stage,'workers':a.workers,
        'code_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
        'packages':{n:version(n) for n in ['duckdb','polars-lts-cpu','pyarrow','fasttext-wheel','markdown-it-py']}}
    write_json(a.report_file.with_suffix('.execution.json'),execution)
    progress(a.stage,'stage_completed',status=result['status'],rows=result.get('completed_records',result.get('unique_records')))
    return result

if __name__=='__main__':main()
