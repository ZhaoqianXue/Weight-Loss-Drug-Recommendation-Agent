"""Bounded, read-only production equivalence check using four actual workers."""
import hashlib
import json
from pathlib import Path
import sqlite3
from weightloss.cleaning import reddit_baseline_columnar as baseline, reddit_fast as fast
from weightloss.cleaning.reddit_runtime import worker_pool, ordered_compute
from weightloss.cleaning.reddit_quality_release import veto


def stamp():
    result={}
    for stage in ['v1','v2-strict','v3-final']:
        folder=Path('.cache/reddit-cleaning')/stage
        for name in ['cleaned.sqlite','run-spec.json']:
            p=folder/name;st=p.stat();result[str(p)]=[st.st_size,st.st_mtime_ns]
    for p in Path('results').glob('reddit-cleaning-v[123]/manifest.json'):
        result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def main():
    before=stamp();root=Path.cwd();model=str(root/'.cache/models/lid.176.bin')
    dbs=[sqlite3.connect(f'file:.cache/reddit-cleaning/{n}/cleaned.sqlite?mode=ro',uri=True) for n in ['v1','v2-strict','v3-final']]
    policies=[json.loads(Path('configs',n).read_text()) for n in ['reddit-cleaning.json','reddit-refinement.json','reddit-evidence-final.json']]
    tasks1=[];tasks2=[];ids=[]
    for key,path,metadata in dbs[0].execute('SELECT key,path,metadata FROM inputs ORDER BY key'):
        entry=json.loads(metadata);absent={(r['kind'],r['id']) for r in entry['verification'].get('documented_source_absences',[])}
        with open(path,'rb') as stream:
            for number in range(1,41):
                line=stream.readline()
                if not line:break
                tasks1.append((line,key,number,entry['kind'],entry['spec'],policies[0],absent));tasks2.append((key,number,line))
                ids.append(('t3_' if key.endswith('/posts') else 't1_')+json.loads(line)['id'])
    with worker_pool(4,baseline.initialize_worker,(model,)) as pool:
        out1=ordered_compute(pool,baseline.compute_batch,tasks1)
    for r in out1:assert r==dbs[0].execute('SELECT * FROM records WHERE fullname=?',(r[0],)).fetchone()
    with worker_pool(4,fast.initialize_worker,('v2',str(root/'.cache/reddit-cleaning/v1/cleaned.sqlite'),model,policies[1])) as pool:
        fast._POOL=pool;out2=fast.parallel_compute(tasks2)
    for r,_,_ in out2:assert r==dbs[1].execute('SELECT * FROM records WHERE fullname=?',(r[0],)).fetchone()
    tasks3=[dbs[1].execute('''SELECT r.rowid,r.fullname,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,r.quality_decision,r.quality_reason,r.features_json,c.fullname
        FROM records r LEFT JOIN canonical_text c ON r.text_hash=c.text_hash WHERE r.fullname=?''',(ident,)).fetchone() for ident in ids]
    with worker_pool(4,fast.initialize_worker,('v3',str(root/'.cache/reddit-cleaning/v2-strict/cleaned.sqlite'),model,policies[2])) as pool:
        fast._POOL=pool;out3=fast.parallel_compute(tasks3)
    historical_annotations=0
    for task,(r,_,comparison) in zip(tasks3,out3):
        def lookup(ident):
            row=dbs[1].execute("SELECT features_json FROM records WHERE fullname=?",(ident,)).fetchone()
            return json.loads(row[0]) if row else None
        feature=fast.evidence.decide(json.loads(task[9]),lookup,policies[2])
        base=dict(fullname=task[1],provenance=json.loads(task[9])["provenance"],text_sha256=task[3],created_utc=task[4],subreddit=task[5],kind=task[6])
        assert r==fast.prepared_record(base,feature,task[2]),"New executor differs from frozen rule"
        actual_record=dbs[2].execute('SELECT * FROM records WHERE fullname=?',(r[0],)).fetchone()
        if r!=actual_record:
            # Legacy migration predates the explanatory information.basis key.
            # Frozen original decide() reproduces the same added annotation;
            # no persisted record is modified to "repair" this historical detail.
            assert r[:9]==actual_record[:9]
            expected_feature=json.loads(r[9]);actual_feature=json.loads(actual_record[9])
            assert 'basis' not in actual_feature['information']
            expected_feature['information'].pop('basis')
            assert expected_feature==actual_feature
            historical_annotations+=1
        assert comparison==dbs[2].execute('SELECT * FROM comparisons WHERE fullname=?',(r[0],)).fetchone()
        if r[7]=='keep':
            reason=veto(json.loads(r[9]));actual=dbs[2].execute('SELECT reason FROM quality_exclusions WHERE fullname=?',(r[0],)).fetchone()
            assert actual==((reason,) if reason else None)
    for db in dbs:db.close()
    assert stamp()==before,'Production metadata changed'
    result=dict(records_per_stage=len(ids),communities=16,inputs=32,workers=4,new_vs_frozen_rule_fields_equal=True,stored_decisions_equal=True,historical_missing_basis_annotations=historical_annotations,quality_sample_equal=True,production_metadata_unchanged=True,scope='Up to first forty rows from every input; no complete reprocessing or production writes',before=before)
    Path('results/reddit-cleaning-fast/unification-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='before'}))

if __name__=='__main__':main()
