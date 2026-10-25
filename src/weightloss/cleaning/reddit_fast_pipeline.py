"""Resume frozen cleaning stages sequentially, with no heartbeat handoff delay."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json


def inputs_complete(directory):
    path=directory/'cleaned.sqlite'
    if not path.exists():return False
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        total,done=db.execute('SELECT count(*),coalesce(sum(done),0) FROM inputs').fetchone()
    return total==32 and done==32


def cleaners_idle(base):
    with ExitStack() as stack:
        for name in ('v1','v2-strict','v3-final'):
            f=stack.enter_context((base/name/'.cleaning.lock').open('a'))
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return False
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Expected1–4workers')
    root=Path.cwd();base=root/'.cache/reddit-cleaning';report=root/'results/reddit-cleaning-fast'
    report.mkdir(parents=True,exist_ok=True)
    state=dict(started_at=datetime.now(timezone.utc).isoformat(),workers=a.workers,status='starting')
    pin={name:sha256_file(Path(__file__).with_name(name)) for name in ['reddit_unified.py','reddit_runtime.py','reddit_baseline_columnar.py','reddit_columnar.py','reddit_fast.py','reddit_fast_pipeline.py','reddit_strict.py','reddit_evidence_final.py','reddit_quality_release.py','reddit_refinement.py','reddit.py']}
    with (base/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state['execution_code_hashes']=pin
        def save(status,**extra):
            state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**extra)
            write_json(report/'pipeline-state.json',state)
        save('waiting_for_cleaner')
        stages=[('v1','v1'),('v2','v2-strict'),('v3','v3-final')]
        for stage,folder in stages:
            while not cleaners_idle(base):
                if (base/'.pipeline-stop').exists():save('stopped');return
                time.sleep(10)
            if (base/'.pipeline-stop').exists():save('stopped');return
            if any(sha256_file(Path(__file__).with_name(n))!=h for n,h in pin.items()):
                save('code_changed');return
            complete=inputs_complete(base/folder)
            if stage=='v3' and complete:
                m=json.loads((root/'results/reddit-cleaning-v3/manifest.json').read_text())
                complete=m.get('full16_sources_cleaned') and m.get('quality_ready') and m.get('quality_screen',{}).get('status')=='complete_for_current_completed_inputs'
            if complete:continue
            free=shutil.disk_usage(base).free/2**30
            if free<16:save('disk_reserve',free_gib=free,stage=stage);return
            source=root/'.cache/reddit-production/full-01' if stage=='v1' else base/('v1' if stage=='v2' else 'v2-strict')
            cmd=[sys.executable,'-m','weightloss.cleaning.reddit_unified',stage,
                 '--workers',str(a.workers),'--source-dir',str(source),
                 '--output-dir',str(base/folder),'--report-file',str(root/f'results/reddit-cleaning-{stage}/manifest.json'),
                 '--language-model',str(root/'.cache/models/lid.176.bin')]
            save('running',stage=stage,free_gib=free)
            with (root/f'.cache/reddit-cleaning-fast-{stage}.log').open('a') as log:
                result=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:save('stage_failed',stage=stage,returncode=result.returncode);return
            if not inputs_complete(base/folder):save('stage_incomplete',stage=stage);return
        final=json.loads((root/'results/reddit-cleaning-v3/manifest.json').read_text())
        if not (final.get('full16_sources_cleaned') and final.get('quality_ready') and final.get('quality_screen',{}).get('status')=='complete_for_current_completed_inputs'):
            save('publication_incomplete');return
        save('complete')

if __name__=='__main__':main()
