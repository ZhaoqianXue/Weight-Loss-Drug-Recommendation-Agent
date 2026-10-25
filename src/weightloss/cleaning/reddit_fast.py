"""Ordered parallel execution of frozen Reddit cleaning rules.

Original modules and run fingerprints remain authoritative. This execution
adapter changes scheduling and reporting only; workers never write databases.
"""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from weightloss.cleaning import reddit_strict as strict
from weightloss.cleaning import reddit_evidence_final as evidence
from weightloss.cleaning import reddit_quality_release as quality
from weightloss.cleaning.reddit import FastTextLanguage, URL
from weightloss.cleaning.reddit_refinement import TextSegments, canonical, refine
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json

from weightloss.cleaning.reddit_runtime import worker_pool, ordered_compute, guard_disk, checkpoint, progress, lock_file

SCHEMA = strict.SCHEMA
def columnar_connect(path, reserve_gib=0):
    from weightloss.cleaning.reddit_columnar import connect
    return connect(path, reserve_gib=reserve_gib)


finalize_input = strict.finalize_input
source_digest = evidence.source_digest
validate_policy = evidence.validate_policy
_POOL = None
_WORK = {}


def initialize_worker(stage, source_path, model, policy):
    db = sqlite3.connect(f'file:{source_path}?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    column = 'record_json' if stage == 'v2' else 'features_json'
    db.execute('PRAGMA cache_size=-32768')
    cache=OrderedDict()
    ready = {row[0] for row in db.execute("SELECT key FROM inputs WHERE done=1")}
    def prefetch(ids):
        missing=sorted({i for i in ids if isinstance(i,str) and i not in cache})
        for start in range(0,len(missing),250):
            chunk=missing[start:start+250]
            found={ident:payload for ident,payload,key in db.execute(f'SELECT fullname,{column},input_key FROM records WHERE fullname IN ('+','.join('?' for _ in chunk)+')',chunk) if key in ready}
            for ident in chunk:cache[ident]=json.loads(found[ident]) if ident in found else None
        while len(cache)>16384:cache.popitem(last=False)
    def lookup(ident):
        if not isinstance(ident,str):return None
        if ident not in cache:prefetch([ident])
        cache.move_to_end(ident)
        return cache[ident]
    _WORK['prefetch']=prefetch

    _WORK.update(stage=stage, db=db, lookup=lookup, policy=policy,
                 detector=FastTextLanguage(model) if stage == 'v2' else None,
                 parser=TextSegments() if stage == 'v2' else None)


def prepared_record(base, features, raw_hash):
    # Match frozen insert_record serialization and removal of parent bodies.
    features = json.loads(canonical(features))
    context = features['context']
    for item in context['ancestors'] + ([context['thread']] if context['thread'] else []):
        item.pop('text', None)
    decision, reason = features.pop('decision'), features.pop('reason')
    return (base['fullname'], base['provenance']['input_key'], raw_hash,
            base['text_sha256'], base['created_utc'], base['subreddit'], base['kind'],
            decision, reason, canonical(features))


def compute_batch(tasks):
    results=[]; lookup=_WORK['lookup']; policy=_WORK['policy']
    if _WORK['stage']=='v2':
        ids=[('t3_' if t[0].endswith('/posts') else 't1_')+json.loads(t[2])['id'] for t in tasks]
        _WORK['prefetch'](ids)
        current=[lookup(i) for i in ids]
        for _ in range(policy['context_depth']+1):
            ids={r.get(k) for r in current if r for k in ('parent_id','link_id') if isinstance(r.get(k),str)}
            _WORK['prefetch'](ids);current=[lookup(i) for i in ids]
    else:
        ids=set()
        for task in tasks:
            context=json.loads(task[9])['context']
            ids.update(n['fullname'] for n in context['ancestors'])
            if context['thread']:ids.add(context['thread']['fullname'])
        _WORK['prefetch'](ids)
    for task in tasks:
        if _WORK['stage']=='v2':
            key, line_number, line = task
            raw=json.loads(line); kind=key.split('/')[1]
            ident=('t3_' if kind=='posts' else 't1_')+raw['id']; base=lookup(ident)
            raw_hash=hashlib.sha256(line).hexdigest()
            if not base or base['provenance']['raw_line_sha256']!=raw_hash or base['provenance']['input_key']!=key:
                raise ValueError('Raw record differs from baseline provenance')
            features=refine(raw,base,_WORK['detector'],_WORK['parser'],lookup,policy)
            base=dict(base,provenance=dict(base['provenance'],line=line_number))
            results.append((prepared_record(base,features,raw_hash),line_number,None))
        else:
            rid,ident,raw_hash,text_hash,stamp,sub,kind,old_dec,old_reason,payload,canonical_id=task
            feature=json.loads(payload); result=evidence.decide(feature,lookup,policy)
            base=dict(fullname=ident,provenance=feature['provenance'],text_sha256=text_hash,created_utc=stamp,subreddit=sub,kind=kind)
            if old_dec=='keep' and canonical_id!=ident:old_dec,old_reason='exclude','duplicate_text'
            comparison=(ident,old_dec,old_reason,feature['language_evidence']['words']<20 or len(URL.sub(' ',feature['prose_text']).strip())<80,
                        not feature['target_mentions'],old_reason in ['uncertain_language_segment','uncertain_language'])
            results.append((prepared_record(base,result,raw_hash),feature['provenance']['line'],comparison))
    return results


def parallel_compute(tasks):
    return ordered_compute(_POOL, compute_batch, tasks)


def insert_batch(db, results):
    ids=list(dict.fromkeys(r[0][0] for r in results)); seen={}
    for i in range(0,len(ids),250):
        chunk=ids[i:i+250]
        seen.update(db.execute('SELECT fullname,raw_hash FROM records WHERE fullname IN ('+','.join('?' for _ in chunk)+')',chunk))
    records=[]; duplicates=[]; comparisons=[]
    for record,line,comparison in results:
        ident,key,raw_hash=record[:3]
        if ident in seen:
            if seen[ident]!=raw_hash:raise ValueError('Conflicting raw versions of a fullname')
            duplicates.append((key,line,ident))
        else:
            seen[ident]=raw_hash;records.append(record)
        if comparison is not None:comparisons.append(comparison)
    db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',records)
    db.executemany('INSERT INTO duplicates VALUES (?,?,?)',duplicates)
    if comparisons:db.executemany('INSERT INTO comparisons VALUES (?,?,?,?,?,?)',comparisons)


def output_connect(path, reserve_gib=0):
    from weightloss.cleaning.reddit_columnar import connect
    return connect(path, stage='v3', reserve_gib=reserve_gib)


def summarize(db, source, scope_count, processed):
    return strict.summarize(db,source,scope_count,processed)

def clean_strict(baseline_dir, directory, report_path, policy, detector, parser, max_records=None):
    baseline_dir,directory,report_path = (Path(p).resolve() for p in (baseline_dir,directory,report_path))
    if max_records is not None and max_records < 1:
        raise ValueError('max_records must be positive')
    required_positive = ['minimum_prose_words','minimum_prose_chars','language_min_words',
        'language_segment_chars','context_depth','context_excerpt_chars','commit_every']
    if policy.get('schema_version') != 2 or any(policy[k] < 1 for k in required_positive) or not .5 <= policy['language_confidence'] <= 1 or policy['reserve_gib'] < 0:
        raise ValueError('Invalid strict cleaning policy')
    spec = json.loads((baseline_dir/'run-spec.json').read_text())
    production = Path(spec['production_directory']).resolve()
    for protected in (baseline_dir,production):
        if directory == protected or directory.is_relative_to(protected) or protected.is_relative_to(directory) or report_path.is_relative_to(protected):
            raise ValueError('Output must be separate from raw and baseline data')
    if report_path.suffix != '.json' or report_path == directory/'run-spec.json':
        raise ValueError('Invalid report path')
    if sha256_file(Path(strict.__file__).with_name('reddit.py')) != spec['cleaner_sha256']:
        raise ValueError('Baseline implementation changed')
    if detector.fingerprint != spec['language']:
        raise ValueError('Expected the pinned baseline language model')
    current_scope = json.loads((production/'production-spec.json').read_text())
    if current_scope != spec['production_scope']:
        raise ValueError('Production scope changed')
    fingerprint = dict(schema_version=2, baseline_directory=str(baseline_dir),
        baseline_spec_sha256=sha256_file(baseline_dir/'run-spec.json'),
        strict_sha256=sha256_file(Path(strict.__file__)),
        refinement_sha256=sha256_file(Path(strict.__file__).with_name('reddit_refinement.py')),
        policy=policy, language=detector.fingerprint, parser=parser.fingerprint)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.cleaning.lock').open('a') as output_lock, (baseline_dir/'.cleaning.lock').open('a') as baseline_lock:
        try:
            lock_file(output_lock)
            lock_file(baseline_lock,fcntl.LOCK_SH)
        except BlockingIOError:
            raise ValueError('A cleaner is active; retry after it finishes')
        path=directory/'run-spec.json'
        if path.exists() and json.loads(path.read_text()) != fingerprint:
            raise ValueError('Strict rules/model/scope changed; use a new derived run')
        if not path.exists(): write_json(path,fingerprint)
        db=columnar_connect(directory/'cleaned.sqlite', policy['reserve_gib'])
        source=sqlite3.connect(f'file:{baseline_dir / "cleaned.sqlite"}?mode=ro',uri=True)
        source.execute('PRAGMA query_only=ON')
        try:
            db.executescript(SCHEMA)
            inputs=source.execute('SELECT key,path,sha256,expected_rows FROM inputs WHERE done=1 ORDER BY key').fetchall()
            if not {r[0] for r in db.execute('SELECT key FROM inputs')} <= {r[0] for r in inputs}:
                raise ValueError('A previously completed baseline input is no longer available')
            with db:
                for key,path,sha,rows in inputs:
                    path=Path(path).resolve()
                    if not path.is_relative_to(production):
                        raise ValueError('Unexpected source path')
                    old=db.execute('SELECT path,sha256,expected_rows FROM inputs WHERE key=?',(key,)).fetchone()
                    if old and old!=(str(path),sha,rows):
                        raise ValueError('Frozen source input changed')
                    db.execute('INSERT OR IGNORE INTO inputs(key,path,sha256,expected_rows) VALUES (?,?,?,?)',(key,str(path),sha,rows))

            @lru_cache(maxsize=8192)
            def lookup(fullname):
                r=source.execute('SELECT record_json FROM records WHERE fullname=? AND input_key IN (SELECT key FROM inputs WHERE done=1)',(fullname,)).fetchone()
                return json.loads(r[0]) if r else None

            processed=0
            for key,path,sha,expected in inputs:
                if sha256_file(Path(path)) != sha:
                    raise ValueError('Source export hash mismatch')
                offset,lines,done=db.execute('SELECT offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
                if done: continue
                kind=key.split('/')[1]
                with open(path,'rb') as stream:
                    stream.seek(offset);eof=False
                    while not eof and (max_records is None or processed < max_records):
                        guard_disk(directory, policy['reserve_gib'])
                        tasks=[]
                        for _ in range(policy['commit_every']):
                            if max_records is not None and processed+len(tasks)>=max_records:break
                            line=stream.readline()
                            if not line:eof=True;break
                            tasks.append((key,lines+len(tasks)+1,line))
                        results=parallel_compute(tasks)
                        with db:
                            insert_batch(db,results)
                            lines+=len(tasks);processed+=len(tasks)
                            checkpoint(db,key,stream.tell(),lines)
                        progress('v2','records_committed',input=key,rows=lines) if lines%10000==0 else None
                        if eof:
                            if lines!=expected or sha256_file(Path(path))!=sha:
                                raise ValueError('Source count/hash mismatch at EOF')
                            with db: finalize_input(db,key)
                    if max_records is not None and processed>=max_records: break
            result=summarize(db,source,len(spec['production_scope']['targets']),processed)
            report_path.parent.mkdir(parents=True,exist_ok=True);write_json(report_path,result)
            return result
        finally:
            db.close();source.close()


def run(source_dir, directory, report_path, policy, max_records=None):
    source_dir,directory,report_path=(Path(p).resolve() for p in (source_dir,directory,report_path))
    validate_policy(policy)
    if max_records is not None and max_records<1: raise ValueError('max_records must be positive')
    spec=json.loads((source_dir/'run-spec.json').read_text())
    baseline=Path(spec['baseline_directory'])
    base_spec=json.loads((baseline/'run-spec.json').read_text())
    production=Path(base_spec['production_directory']).resolve()
    for protected in [source_dir,baseline,production]:
        if directory==protected or directory.is_relative_to(protected) or protected.is_relative_to(directory) or report_path.is_relative_to(protected):
            raise ValueError('Output must be outside all source directories')
    if report_path.suffix!='.json' or report_path.is_relative_to(directory): raise ValueError('Invalid aggregate report path')
    module_dir=Path(evidence.__file__).parent
    if sha256_file(baseline/'run-spec.json')!=spec['baseline_spec_sha256']:
        raise ValueError('Baseline provenance changed')
    for name,key in [('reddit_strict.py','strict_sha256'),('reddit_refinement.py','refinement_sha256')]:
        if sha256_file(module_dir/name)!=spec[key]: raise ValueError('Pinned v2 implementation changed')
    if json.loads((production/'production-spec.json').read_text())!=base_spec['production_scope']:
        raise ValueError('Raw source scope changed')
    fingerprint=dict(schema_version=3,source_directory=str(source_dir),source_spec_sha256=sha256_file(source_dir/'run-spec.json'),
                     code_sha256=sha256_file(Path(evidence.__file__)),policy=policy)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for folder,mode in [(directory,fcntl.LOCK_EX),(source_dir,fcntl.LOCK_SH)]:
            lock=stack.enter_context((folder/'.cleaning.lock').open('a'))
            try: lock_file(lock,mode)
            except BlockingIOError: raise ValueError('Another cleaner is active')
        fp=directory/'run-spec.json'
        if fp.exists() and json.loads(fp.read_text())!=fingerprint: raise ValueError('Rules/source/code changed; create a new run')
        if not fp.exists(): write_json(fp,fingerprint)
        source=sqlite3.connect(f'file:{source_dir / "cleaned.sqlite"}?mode=ro',uri=True)
        source.execute('PRAGMA query_only=ON');stack.callback(source.close)
        db=output_connect(directory/'cleaned.sqlite', policy['reserve_gib']);stack.callback(db.close)
        db.executescript(SCHEMA+'''
            CREATE TABLE IF NOT EXISTS comparisons(fullname TEXT PRIMARY KEY,v2_decision TEXT,v2_reason TEXT,
                old_short INTEGER,old_no_name INTEGER,old_language_veto INTEGER);
        ''')
        inputs=source.execute('SELECT key,path,sha256,expected_rows FROM inputs WHERE done=1 ORDER BY key').fetchall()
        if not {r[0] for r in db.execute('SELECT key FROM inputs')} <= {r[0] for r in inputs}:
            raise ValueError('A completed source input disappeared')
        @lru_cache(maxsize=8192)
        def lookup(fullname):
            row=source.execute('SELECT features_json FROM records WHERE fullname=? AND input_key IN (SELECT key FROM inputs WHERE done=1)',(fullname,)).fetchone()
            return json.loads(row[0]) if row else None
        processed=0
        for key,path,raw_sha,raw_count in inputs:
            evidence_sha,expected=source_digest(source,key)
            previous=db.execute('SELECT sha256,expected_rows,offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
            if previous and previous[:2]!=(evidence_sha,expected): raise ValueError('Frozen source evidence changed')
            with db:
                db.execute('INSERT OR IGNORE INTO inputs(key,path,sha256,expected_rows) VALUES (?,?,?,?)',(key,str(source_dir/'cleaned.sqlite'),evidence_sha,expected))
            offset,lines,done=db.execute('SELECT offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
            if done: continue
            while max_records is None or processed<max_records:
                guard_disk(directory, policy['reserve_gib'])
                limit=policy['commit_every'] if max_records is None else min(policy['commit_every'],max_records-processed)
                rows=source.execute('''SELECT r.rowid,r.fullname,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,
                    r.quality_decision,r.quality_reason,r.features_json,c.fullname
                    FROM records r LEFT JOIN canonical_text c ON r.text_hash=c.text_hash
                    WHERE r.input_key=? AND r.rowid>? ORDER BY r.rowid LIMIT ?''',(key,offset,limit)).fetchall()
                if not rows:
                    if lines!=expected: raise ValueError('Evidence count mismatch')
                    with db: finalize_input(db,key)
                    break
                results=parallel_compute(rows)
                with db:
                    insert_batch(db,results)
                    offset=rows[-1][0];lines+=len(rows);processed+=len(rows)
                    checkpoint(db,key,offset,lines)
                if lines%10000<limit: progress('v3','records_committed',input=key,rows=lines,expected=expected)
            if max_records is not None and processed>=max_records: break
        counts=dict(db.execute('SELECT decision,count(*) FROM cleaned_records GROUP BY decision'))
        reasons=dict(db.execute('SELECT reason,count(*) FROM cleaned_records GROUP BY reason'))
        source_inputs=[dict(zip(['key','processed','expected','done'],r)) for r in db.execute('SELECT key,lines,expected_rows,done FROM inputs ORDER BY key')]
        complete=len(source_inputs)==len(inputs) and all(r['done'] for r in source_inputs)
        transitions=[dict(zip(['v2','v3','records'],r)) for r in db.execute('SELECT c.v2_decision,r.decision,count(*) FROM comparisons c JOIN cleaned_records r USING(fullname) GROUP BY c.v2_decision,r.decision')]
        restored_gates=dict(zip(['old_short','old_no_name','old_language_veto'],db.execute("SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM comparisons c JOIN cleaned_records r USING(fullname) WHERE c.v2_decision='exclude' AND r.decision='keep'").fetchone()))
        communities=[dict(zip(['subreddit','decision','records'],r)) for r in db.execute('SELECT subreddit,decision,count(*) FROM cleaned_records GROUP BY subreddit,decision')]
        report=dict(schema_version=3,status='all_current_inputs_cleaned' if complete else 'partial',
                    generated_at=datetime.now(timezone.utc).isoformat(),source_input_files=len(inputs),inputs=source_inputs,
                    full16_sources_cleaned=complete and len(inputs)==32,completed_records=sum(counts.values()),
                    processed_records=db.execute('SELECT count(*) FROM records').fetchone()[0],
                    processed_this_invocation=processed,dispositions=counts,reasons=reasons,transitions=transitions,
                    newly_retained_old_gate_hits_overlapping=restored_gates,by_community=communities,
                    independent_precision=None,clinical_eligibility='not_assessed',policy=fingerprint)
        report_path.parent.mkdir(parents=True,exist_ok=True);write_json(report_path,report)
        return report


def release(directory, report_path):
    directory,report_path=Path(directory).resolve(),Path(report_path).resolve()
    if not (directory/'run-spec.json').exists() or report_path.is_relative_to(directory) or report_path.suffix!='.json':
        raise ValueError('Expected an existing derived run and separate aggregate report')
    spec=json.loads((directory/'run-spec.json').read_text())
    for protected in [Path(spec['source_directory']), Path(json.loads((Path(spec['source_directory'])/'run-spec.json').read_text())['baseline_directory'])]:
        if report_path.is_relative_to(protected):raise ValueError('Protected report path')
    fingerprint=dict(schema_version=1,code_sha256=sha256_file(Path(quality.__file__)),base_spec_sha256=sha256_file(directory/'run-spec.json'))
    with (directory/'.cleaning.lock').open('a') as lock:
        try:lock_file(lock)
        except BlockingIOError:raise ValueError('Evidence cleaner is still active')
        fp=directory/'quality-screen-spec.json'
        if fp.exists() and json.loads(fp.read_text())!=fingerprint:raise ValueError('Publication rules changed')
        if not fp.exists():write_json(fp,fingerprint)
        db=output_connect(directory/'cleaned.sqlite', spec['policy']['reserve_gib'])
        try:
            db.executescript('CREATE TABLE IF NOT EXISTS quality_exclusions(fullname TEXT PRIMARY KEY,reason TEXT);')
            # Idempotent: only newly arrived keep candidates need screening.
            progress('v3','publication_check_started')
            old=db.execute('SELECT count(*) FROM quality_exclusions').fetchone()[0]
            new_rows=[];checked=0
            for ident,payload in db.execute("SELECT r.fullname,r.features_json FROM records r JOIN inputs i ON r.input_key=i.key AND i.done=1 WHERE r.quality_decision='keep' AND r.fullname NOT IN (SELECT fullname FROM quality_exclusions)"):
                checked+=1
                if checked%100000==0:progress("v3","publication_check_progress",rows=checked)
                reason=quality.veto(json.loads(payload))
                if reason:new_rows.append((ident,reason))
            with db:
                db.executemany('INSERT INTO quality_exclusions VALUES (?,?)',new_rows)
                # One transaction publishes the override-aware partition.
                db.execute('DROP VIEW IF EXISTS kept_records');db.execute('DROP VIEW IF EXISTS excluded_records');db.execute('DROP VIEW IF EXISTS cleaned_records')
                db.execute('''CREATE VIEW cleaned_records AS
                    SELECT r.fullname,r.input_key,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,
                    CASE WHEN q.fullname IS NOT NULL THEN 'exclude' WHEN r.quality_decision='keep' AND c.fullname=r.fullname THEN 'keep' ELSE 'exclude' END AS decision,
                    CASE WHEN q.fullname IS NOT NULL THEN q.reason WHEN r.quality_decision='keep' AND c.fullname!=r.fullname THEN 'duplicate_text' ELSE r.quality_reason END AS reason,
                    c.fullname AS canonical_fullname,r.features_json
                    FROM records r JOIN inputs i ON r.input_key=i.key AND i.done=1
                    LEFT JOIN canonical_text c ON r.text_hash=c.text_hash LEFT JOIN quality_exclusions q ON r.fullname=q.fullname''')
                db.execute("CREATE VIEW kept_records AS SELECT * FROM cleaned_records WHERE decision='keep'")
                db.execute("CREATE VIEW excluded_records AS SELECT * FROM cleaned_records WHERE decision='exclude'")
            progress('v3','publication_check_committed',added_exclusions=len(new_rows))
            report=json.loads(report_path.read_text())
            report['dispositions']=dict(db.execute('SELECT decision,count(*) FROM cleaned_records GROUP BY decision'))
            report['reasons']=dict(db.execute('SELECT reason,count(*) FROM cleaned_records GROUP BY reason'))
            report['completed_records']=sum(report['dispositions'].values())
            report['transitions']=[dict(zip(['v2','v3','records'],r)) for r in db.execute('SELECT c.v2_decision,r.decision,count(*) FROM comparisons c JOIN cleaned_records r USING(fullname) GROUP BY c.v2_decision,r.decision')]
            report['by_community']=[dict(zip(['subreddit','decision','records'],r)) for r in db.execute('SELECT subreddit,decision,count(*) FROM cleaned_records GROUP BY subreddit,decision')]
            report['newly_retained_old_gate_hits_overlapping']=dict(zip(['old_short','old_no_name','old_language_veto'],db.execute("SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM comparisons c JOIN cleaned_records r USING(fullname) WHERE c.v2_decision='exclude' AND r.decision='keep'").fetchone()))
            report['quality_screen']=dict(status='complete_for_current_completed_inputs',previous_exclusions=old,
                added_exclusions=len(new_rows),total_exclusions=old+len(new_rows),fingerprint=fingerprint)
            report['generated_at']=datetime.now(timezone.utc).isoformat();report['quality_ready']=True
            write_json(report_path,report)
            return report
        finally:db.close()


def main():
    from weightloss.cleaning.reddit_unified import main as unified_main
    return unified_main()

if __name__=='__main__':main()
