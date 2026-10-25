"""Checksum-verified, single-pass extraction of selected communities from dumps.

Each file commits atomically. Restart reuses completed files, restarts incomplete
files, and never infers historical completeness from a successful decompression.
Only projected research fields are saved; no author histories are collected.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
from .reddit import FIELDS, sha256_file, target_spec, utc


@contextmanager
def lines(path):
    if path.suffix != '.zst':
        with path.open('rb') as stream:
            yield stream
        return
    binary = shutil.which('zstd')
    if not binary:
        raise ValueError('Install the zstd CLI to import .zst files')
    # zstd verifies frame checksums and detects truncation. Never use shell=True.
    process = subprocess.Popen([binary, '-d', '-c', '--', str(path)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        yield process.stdout
        if process.wait() != 0:
            raise ValueError('Zstandard validation failed (corrupt or truncated archive)')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def import_archives(manifest_path, output_dir, config):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files', [])
    end = manifest.get('snapshot_end_exclusive')
    if not isinstance(end, int) or not files:
        raise ValueError('Manifest requires snapshot_end_exclusive epoch and a nonempty files list')
    specs = {name.casefold(): target_spec(config, name) for name in config['targets']}
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.writer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError('Another importer owns this output directory')
    db = sqlite3.connect(directory / 'archive.sqlite')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS files (sha256 TEXT,kind TEXT,status TEXT,rows INTEGER,selected INTEGER,source TEXT,
            PRIMARY KEY(sha256,kind));
        CREATE TABLE IF NOT EXISTS items (subreddit TEXT,kind TEXT,id TEXT,created_utc INTEGER,payload TEXT,
            PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS conflicts (kind TEXT,id TEXT,payload TEXT,source_sha256 TEXT,
            PRIMARY KEY(kind,id,source_sha256));
    ''')
    pinned = json.dumps({'manifest': manifest, 'specs': specs}, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()
    try:
        if old and old[0] != pinned:
            raise ValueError('Archive manifest or source scope changed; use a new directory')
        with db:
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('spec', pinned))
        for entry in files:
            path = (manifest_path.parent / entry['path']).resolve()
            kind, expected = entry['kind'], entry['sha256']
            if kind not in FIELDS or len(expected) != 64 or not entry.get('source_url'):
                raise ValueError('Each file requires kind, SHA-256 and source_url')
            lo, hi = entry['coverage_start_inclusive'], entry['coverage_end_exclusive']
            if not isinstance(lo, int) or not isinstance(hi, int) or lo >= hi:
                raise ValueError('File coverage must be a nonempty half-open epoch interval')
            if sha256_file(path) != expected:
                raise ValueError('Archive SHA-256 does not match manifest')
            prior = db.execute('SELECT status FROM files WHERE sha256=? AND kind=?', (expected, kind)).fetchone()
            if prior and prior[0] == 'complete':
                continue
            total = selected = 0
            try:
                with db:
                    with lines(path) as stream:
                        for raw in stream:
                            total += 1
                            row = json.loads(raw)
                            name = row.get('subreddit', '').casefold()
                            stamp = row.get('created_utc')
                            if not isinstance(stamp, int) or not lo <= stamp < hi:
                                raise ValueError('Archive record violates declared time coverage')
                            if name not in specs or not specs[name]['start'] <= stamp < end:
                                continue
                            ident = row.get('id')
                            if not isinstance(ident, str) or not ident:
                                raise ValueError('Selected archive record lacks an ID')
                            required = ('title', 'selftext') if kind == 'posts' else ('body','parent_id','link_id')
                            if any(key not in row for key in required):
                                raise ValueError('Selected archive record lacks required content fields')
                            payload = json.dumps({k: row[k] for k in FIELDS[kind].split(',') if k in row}, sort_keys=True)
                            existing = db.execute('SELECT payload FROM items WHERE kind=? AND id=?', (kind,ident)).fetchone()
                            if existing and existing[0] != payload:
                                db.execute('INSERT OR IGNORE INTO conflicts VALUES (?,?,?,?)', (kind,ident,payload,expected))
                            else:
                                db.execute('INSERT OR IGNORE INTO items VALUES (?,?,?,?,?)', (name,kind,ident,stamp,payload))
                            selected += 1
                    db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)',
                               (expected,kind,'complete',total,selected,entry['source_url']))
            except Exception:
                with db:
                    db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)',
                               (expected,kind,'failed',total,0,entry['source_url']))
                raise
        report = {'status':'files_validated_coverage_unverified', 'checked_at':utc(time.time()),
                  'files': [dict(zip(('sha256','kind','status','rows','selected','source'),r)) for r in db.execute('SELECT * FROM files')],
                  'counts':[dict(zip(('subreddit','kind','count'),r)) for r in db.execute('SELECT subreddit,kind,count(*) FROM items GROUP BY subreddit,kind')],
                  'declared_coverage_gaps':declared_coverage_gaps(files,specs,end),
                  'conflicting_versions':db.execute('SELECT count(*) FROM conflicts').fetchone()[0],
                  'note':'File integrity and declared bounds only. Missing source months and source-to-API ID agreement remain separate checks.'}
        temp=directory/'manifest.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(directory/'manifest.json')
        return report
    finally:
        db.close()
        lock.close()


def declared_coverage_gaps(files, specs, end):
    """Subtract explicitly declared file intervals; not evidence of source capture."""
    gaps=[]
    for name,spec in specs.items():
        for kind in FIELDS:
            intervals=[]
            for entry in files:
                names=entry.get('subreddits', [])
                covered=names=='all' or name in [n.casefold() for n in names]
                if entry['kind']==kind and covered:
                    intervals.append((entry['coverage_start_inclusive'],entry['coverage_end_exclusive']))
            cursor=spec['start']
            for lo,hi in sorted(intervals):
                if cursor>=end: break
                if hi<=cursor: continue
                if lo>cursor:
                    gaps.append({'subreddit':name,'kind':kind,'start':cursor,'end':min(lo,end)})
                cursor=max(cursor,hi)
            if cursor<end:
                gaps.append({'subreddit':name,'kind':kind,'start':cursor,'end':end})
    return gaps
