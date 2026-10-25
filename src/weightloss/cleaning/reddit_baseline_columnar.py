"""Stage-one adapter for future runs; original record rules and checkpoints remain intact."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from weightloss.cleaning.reddit_runtime import worker_pool, chunks, guard_disk, checkpoint, progress, lock_file
from weightloss.cleaning import reddit as baseline
from weightloss.cleaning.reddit import clean_record, discover, summarize, FastTextLanguage
from weightloss.cleaning.reddit_columnar import connect as columnar_connect
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json


_DETECTOR = None


def initialize_worker(model):
    global _DETECTOR
    _DETECTOR = FastTextLanguage(model)


def compute_batch(tasks, detector=None):
    detector = detector or _DETECTOR
    results=[]
    for line,key,number,kind,spec,policy,absent in tasks:
        row=json.loads(line)
        record=clean_record(row,kind,spec,detector,policy,(kind,row.get('id')) in absent)
        raw_hash=hashlib.sha256(line).hexdigest()
        record['provenance']=dict(input_key=key,line=number,raw_line_sha256=raw_hash)
        results.append((record['fullname'],key,number,raw_hash,record['text_sha256'],len(record['text']),record['decision'],record['reason'],json.dumps(record,ensure_ascii=False)))
    return results

def clean_production(production_dir, output_dir, report_file, policy, config, detector,
                     selected=None, max_records=None, pool=None):
    production, directory, report_path = (Path(p).expanduser().resolve() for p in (production_dir, output_dir, report_file))
    if directory == production or directory.is_relative_to(production) or production.is_relative_to(directory):
        raise ValueError('Cleaning output must be separate from production')
    if report_path == directory or report_path.is_relative_to(production) or report_path == production:
        raise ValueError('Report must not replace raw data or output directory')
    if report_path.suffix != '.json':
        raise ValueError('Report requires a JSON path')
    if max_records is not None and max_records < 1:
        raise ValueError('max_records must be positive')
    if policy.get('schema_version') != 1 or not 0.5 <= policy['language_confidence'] <= 1 or policy['language_min_words'] < 1 or policy['language_max_chars'] < 1 or policy['commit_every'] < 1 or policy['reserve_gib'] < 0:
        raise ValueError('Invalid cleaning policy')
    scope, inputs, pending = discover(production, config, selected)
    fingerprint = dict(schema_version=1, production_directory=str(production), production_scope=scope,
        policy=policy, configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        cleaner_sha256=sha256_file(Path(baseline.__file__)), language=detector.fingerprint,
        selected=sorted(selected) if selected else None)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Refusing a nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.cleaning.lock').open('a') as lock:
        try: lock_file(lock)
        except BlockingIOError: raise ValueError('Another cleaner owns this output directory')
        spec_path = directory/'run-spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != fingerprint:
            raise ValueError('Cleaning rules/model/scope changed; use a new derived run')
        if not spec_path.exists(): write_json(spec_path, fingerprint)
        db = columnar_connect(directory/'cleaned.sqlite', stage='v1', reserve_gib=policy['reserve_gib'])
        try:
            db.executescript('''CREATE TABLE IF NOT EXISTS inputs (
                key TEXT PRIMARY KEY,subreddit TEXT,kind TEXT,path TEXT,sha256 TEXT,expected_rows INTEGER,
                offset INTEGER DEFAULT 0,lines INTEGER DEFAULT 0,done INTEGER DEFAULT 0,metadata TEXT);
                CREATE TABLE IF NOT EXISTS records (fullname TEXT PRIMARY KEY,input_key TEXT,line INTEGER,
                raw_hash TEXT,text_hash TEXT,text_length INTEGER,decision TEXT,reason TEXT,record_json TEXT);
                CREATE INDEX IF NOT EXISTS records_input ON records(input_key);
                CREATE INDEX IF NOT EXISTS records_text ON records(text_hash);
                CREATE TABLE IF NOT EXISTS duplicates (input_key TEXT,line INTEGER,fullname TEXT,raw_hash TEXT,
                PRIMARY KEY(input_key,line));
                CREATE VIEW IF NOT EXISTS kept_records AS SELECT * FROM records WHERE decision='keep';
                CREATE VIEW IF NOT EXISTS review_records AS SELECT * FROM records WHERE decision='review';
                CREATE VIEW IF NOT EXISTS excluded_records AS SELECT * FROM records WHERE decision='exclude';''')
            known = {r[0] for r in db.execute('SELECT key FROM inputs')}
            if not known <= {i['key'] for i in inputs}:
                raise ValueError('A previously selected source is no longer verified/available')
            for entry in inputs:
                old = db.execute('SELECT path,sha256,expected_rows FROM inputs WHERE key=?',(entry['key'],)).fetchone()
                if old and old != (entry['path'], entry['sha256'], entry['expected_rows']):
                    raise ValueError('Frozen source export changed; use a new derived run')
                with db:
                    db.execute('INSERT OR IGNORE INTO inputs(key,subreddit,kind,path,sha256,expected_rows,metadata) VALUES (?,?,?,?,?,?,?)',
                               (entry['key'],entry['subreddit'],entry['kind'],entry['path'],entry['sha256'],entry['expected_rows'],json.dumps(entry)))
            processed = 0
            for entry in inputs:
                key = entry['key'];offset,lines,done = db.execute('SELECT offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
                path = Path(entry['path'])
                if sha256_file(path) != entry['sha256']:
                    raise ValueError('Source export hash mismatch')
                if done: continue
                absent = {(r['kind'],r['id']) for r in entry['verification'].get('documented_source_absences',[])}
                with path.open('rb') as stream:
                    stream.seek(offset)
                    eof = False
                    while not eof and (max_records is None or processed < max_records):
                        guard_disk(directory, policy['reserve_gib'])
                        tasks=[]
                        for _ in range(policy['commit_every']):
                            if max_records is not None and processed+len(tasks)>=max_records: break
                            line=stream.readline()
                            if not line: eof=True;break
                            tasks.append((line,key,lines+len(tasks)+1,entry['kind'],entry['spec'],policy,absent))
                        batches=list(chunks(tasks))
                        results = pool.map(compute_batch,batches) if pool else (compute_batch(batch,detector) for batch in batches)
                        with db:
                            for batch in results:
                                for record in batch:
                                    ident,_,number,raw_hash=record[:4]
                                    old=db.execute('SELECT raw_hash FROM records WHERE fullname=?',(ident,)).fetchone()
                                    if old:
                                        if old[0]!=raw_hash:raise ValueError('Conflicting versions of the same Reddit fullname')
                                        db.execute('INSERT INTO duplicates VALUES (?,?,?,?)',(key,number,ident,raw_hash))
                                    else:db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)',record)
                            lines+=len(tasks);processed+=len(tasks)
                            checkpoint(db,key,stream.tell(),lines)
                        if lines%10000==0:progress('v1','records_committed',input=key,rows=lines)
                        if eof:
                            if lines != entry['expected_rows'] or sha256_file(path) != entry['sha256']:
                                raise ValueError('Source count/hash changed; input not accepted as complete')
                            with db: db.execute('UPDATE inputs SET done=1 WHERE key=?',(key,))
                    if max_records is not None and processed >= max_records: break
            report = summarize(db,directory,scope,pending,selected)
            report['records_processed_this_invocation'] = processed
            report_path.parent.mkdir(parents=True,exist_ok=True)
            write_json(report_path,report)
            return report
        finally: db.close()

def main():
    import sys
    from weightloss.cleaning.reddit_unified import main as unified_main
    argv=['--source-dir' if arg=='--production-dir' else arg for arg in sys.argv[1:]]
    return unified_main(['v1',*argv])

if __name__=='__main__':main()
