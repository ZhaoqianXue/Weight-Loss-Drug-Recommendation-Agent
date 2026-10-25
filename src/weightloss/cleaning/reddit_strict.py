"""Resumable strict binary cleaning over verified immutable Reddit exports.

v1 is an internal provenance/index stage. Only kept_records from this stage is
the analysis text-quality view. Exclusions remain auditable, never deleted.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from weightloss.cleaning.reddit import FastTextLanguage
from weightloss.cleaning.reddit_refinement import TextSegments, canonical, refine
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json


SCHEMA = '''
CREATE TABLE IF NOT EXISTS inputs (
    key TEXT PRIMARY KEY, path TEXT, sha256 TEXT, expected_rows INTEGER,
    offset INTEGER DEFAULT 0, lines INTEGER DEFAULT 0, done INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS records (
    fullname TEXT PRIMARY KEY, input_key TEXT, raw_hash TEXT, text_hash TEXT,
    created_utc INTEGER, subreddit TEXT, kind TEXT,
    quality_decision TEXT CHECK(quality_decision IN ('keep','exclude')),
    quality_reason TEXT, features_json TEXT);
CREATE INDEX IF NOT EXISTS records_input ON records(input_key);
CREATE INDEX IF NOT EXISTS records_summary ON records(quality_decision,quality_reason);
CREATE TABLE IF NOT EXISTS canonical_text (
    text_hash TEXT PRIMARY KEY, fullname TEXT, created_utc INTEGER);
CREATE TABLE IF NOT EXISTS duplicates (
    input_key TEXT, line INTEGER, fullname TEXT, PRIMARY KEY(input_key,line));
CREATE VIEW IF NOT EXISTS cleaned_records AS
SELECT r.fullname,r.input_key,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,
    CASE WHEN r.quality_decision='keep' AND c.fullname=r.fullname THEN 'keep' ELSE 'exclude' END AS decision,
    CASE WHEN r.quality_decision='keep' AND c.fullname!=r.fullname THEN 'duplicate_text' ELSE r.quality_reason END AS reason,
    c.fullname AS canonical_fullname,r.features_json
FROM records r JOIN inputs i ON r.input_key=i.key AND i.done=1
LEFT JOIN canonical_text c ON r.text_hash=c.text_hash;
CREATE VIEW IF NOT EXISTS kept_records AS
SELECT r.* FROM cleaned_records r JOIN inputs i ON r.input_key=i.key WHERE r.decision='keep' AND i.done=1;
CREATE VIEW IF NOT EXISTS excluded_records AS
SELECT r.* FROM cleaned_records r JOIN inputs i ON r.input_key=i.key WHERE r.decision='exclude' AND i.done=1;
'''


def insert_record(db, base, features, raw_hash):
    key, line = base['provenance']['input_key'], base['provenance']['line']
    previous = db.execute('SELECT raw_hash FROM records WHERE fullname=?', (base['fullname'],)).fetchone()
    if previous:
        if previous[0] != raw_hash:
            raise ValueError('Conflicting raw versions of a fullname')
        db.execute('INSERT INTO duplicates VALUES (?,?,?)',(key,line,base['fullname']))
        return
    # Store bounded context references, not millions of repeated parent bodies.
    features = json.loads(canonical(features))
    context = features['context']
    for item in context['ancestors'] + ([context['thread']] if context['thread'] else []):
        item.pop('text', None)
    decision, reason = features.pop('decision'), features.pop('reason')
    db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
        (base['fullname'],key,raw_hash,base['text_sha256'],base['created_utc'],base['subreddit'],base['kind'],
         decision,reason,canonical(features)))


def finalize_input(db, key):
    # Publish an input and its deterministic duplicate representatives together.
    # Incomplete inputs cannot displace previously published keep records.
    db.execute('''INSERT INTO canonical_text
        SELECT text_hash,fullname,created_utc FROM records
        WHERE input_key=? AND quality_decision='keep'
        ON CONFLICT(text_hash) DO UPDATE SET fullname=excluded.fullname,created_utc=excluded.created_utc
        WHERE (excluded.created_utc,excluded.fullname) < (canonical_text.created_utc,canonical_text.fullname)''',(key,))
    db.execute('UPDATE inputs SET done=1 WHERE key=?',(key,))


def summarize(db, source, scope_count, processed):
    inputs = [dict(zip(['key','rows_read','expected_rows','done'], r))
              for r in db.execute('SELECT key,lines,expected_rows,done FROM inputs ORDER BY key')]
    ready = [r for r in inputs if r['done']]
    dispositions = dict(db.execute('SELECT decision,count(*) FROM cleaned_records GROUP BY decision'))
    reasons = dict(db.execute('SELECT reason,count(*) FROM cleaned_records GROUP BY reason'))
    view_counts = {name: db.execute('SELECT count(*) FROM '+name).fetchone()[0]
                   for name in ['kept_records','excluded_records']}
    baseline_ready = source.execute('SELECT count(*) FROM inputs WHERE done=1').fetchone()[0]
    complete = len(ready) == scope_count*2 and baseline_ready == scope_count*2
    return dict(schema_version=2, status='all_sources_cleaned' if complete else 'partial',
                generated_at=datetime.now(timezone.utc).isoformat(),
                scope_communities=scope_count, completed_input_files=len(ready),
                baseline_completed_input_files=baseline_ready, inputs=inputs,
                unique_records=db.execute('SELECT count(*) FROM records').fetchone()[0],
                completed_records=sum(dispositions.values()), dispositions=dispositions,
                reasons=reasons, analysis_views_completed_inputs_only=view_counts,
                records_processed_this_invocation=processed,
                classification='binary_precision_first',
                clinical_eligibility='not_yet_assessed',
                threshold_accuracy='not_estimated_against_independent_gold_labels')


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
    if sha256_file(Path(__file__).with_name('reddit.py')) != spec['cleaner_sha256']:
        raise ValueError('Baseline implementation changed')
    if detector.fingerprint != spec['language']:
        raise ValueError('Expected the pinned baseline language model')
    current_scope = json.loads((production/'production-spec.json').read_text())
    if current_scope != spec['production_scope']:
        raise ValueError('Production scope changed')
    fingerprint = dict(schema_version=2, baseline_directory=str(baseline_dir),
        baseline_spec_sha256=sha256_file(baseline_dir/'run-spec.json'),
        strict_sha256=sha256_file(Path(__file__)),
        refinement_sha256=sha256_file(Path(__file__).with_name('reddit_refinement.py')),
        policy=policy, language=detector.fingerprint, parser=parser.fingerprint)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.cleaning.lock').open('a') as output_lock, (baseline_dir/'.cleaning.lock').open('a') as baseline_lock:
        try:
            fcntl.flock(output_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(baseline_lock,fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('A cleaner is active; retry after it finishes')
        path=directory/'run-spec.json'
        if path.exists() and json.loads(path.read_text()) != fingerprint:
            raise ValueError('Strict rules/model/scope changed; use a new derived run')
        if not path.exists(): write_json(path,fingerprint)
        db=sqlite3.connect(directory/'cleaned.sqlite')
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
                        if shutil.disk_usage(directory).free < policy['reserve_gib']*2**30:
                            raise ValueError('Disk reserve reached; checkpoint preserved')
                        with db:
                            for _ in range(policy['commit_every']):
                                if max_records is not None and processed >= max_records: break
                                line=stream.readline()
                                if not line: eof=True;break
                                lines+=1; raw=json.loads(line)
                                fullname=('t3_' if kind=='posts' else 't1_')+raw['id']
                                base=lookup(fullname)
                                raw_hash=hashlib.sha256(line).hexdigest()
                                if not base or base['provenance']['raw_line_sha256'] != raw_hash or base['provenance']['input_key'] != key:
                                    raise ValueError('Raw record differs from baseline provenance')
                                features=refine(raw,base,detector,parser,lookup,policy)
                                # Duplicate source lines use their current line for the event.
                                base=dict(base,provenance=dict(base['provenance'],line=lines))
                                insert_record(db,base,features,raw_hash)
                                processed+=1
                            db.execute('UPDATE inputs SET offset=?,lines=? WHERE key=?',(stream.tell(),lines,key))
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report-file',type=Path,required=True)
    p.add_argument('--language-model',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=Path('configs/reddit-refinement.json'))
    p.add_argument('--max-records',type=int)
    a=p.parse_args()
    result=clean_strict(a.baseline_dir,a.output_dir,a.report_file,json.loads(a.policy.read_text()),
                        FastTextLanguage(a.language_model),TextSegments(),a.max_records)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
