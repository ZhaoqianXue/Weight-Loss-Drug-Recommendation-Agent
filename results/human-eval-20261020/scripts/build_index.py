"""Keyword index (SQLite FTS5, BM25 ranking) over the 258,954 source records the knowledge graph was built from.

Used only by the text-retrieval baseline B2 (baselines.py). One row per record: the record text (its sentences joined)
and the source metadata that comes with the record, never anything derived by the pipeline. Reddit: platform, kind
(post or comment), community (subreddit) and date. WebMD: the reviewer profile (brand, ingredient, age group, gender,
condition chosen from the site's list, medication duration) and date. Both the text and the metadata are searchable.

Input: .cache/reextraction-20261016/full/inputs.jsonl (frozen, sha 63a79171...).
Output: .cache/human-eval-20261020/records.sqlite (records table + records_fts) and its manifest.
"""
import datetime as dt
import hashlib
import json
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUTS = os.path.join(ROOT, '.cache/reextraction-20261016/full/inputs.jsonl')
OUT = os.path.join(ROOT, '.cache/human-eval-20261020/records.sqlite')
VERSION = 'records-index-1.0.0'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def date_of(rec):
    if rec['platform'] == 'reddit':
        return dt.datetime.fromtimestamp(rec['created_utc'], dt.timezone.utc).date().isoformat()
    m, d, y = rec['date'].split('/')
    return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'


def meta_of(rec):
    if rec['platform'] == 'reddit':
        return f"r/{rec['community']}"
    p = rec['profile']
    parts = [p.get('brand'), p.get('ingredient'), f"age {p['age']}" if p.get('age') else None, p.get('gender'),
             p.get('condition'), f"taken {p['medication_duration']}" if p.get('medication_duration') else None]
    return '; '.join(x for x in parts if x)


def main():
    started = time.time()
    if os.path.exists(OUT):
        os.remove(OUT)
    db = sqlite3.connect(OUT)
    db.executescript("""
        CREATE TABLE records(rid INTEGER PRIMARY KEY, record_key TEXT UNIQUE, platform TEXT, kind TEXT, community TEXT,
                             date TEXT, meta TEXT, text TEXT);
        CREATE VIRTUAL TABLE records_fts USING fts5(text, meta, content='records', content_rowid='rid',
                                                    tokenize='unicode61 remove_diacritics 2', prefix='3 4 5');
    """)
    rows, n = [], 0
    with open(INPUTS) as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            text = ' '.join(s['text'] for s in rec['sentences'])
            rows.append((n, f"{rec['platform']}:{rec['record_id']}", rec['platform'], rec['kind'], rec.get('community'),
                         date_of(rec), meta_of(rec), text))
            if len(rows) == 20000:
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)', rows)
                rows = []
    db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)', rows)
    db.execute("INSERT INTO records_fts(records_fts) VALUES ('rebuild')")
    db.execute("INSERT INTO records_fts(records_fts) VALUES ('optimize')")
    db.commit()
    counts = dict(db.execute('SELECT platform, count(*) FROM records GROUP BY platform').fetchall())
    db.close()
    manifest = {'version': VERSION, 'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                'inputs': os.path.relpath(INPUTS, ROOT), 'inputs_sha256': sha256(INPUTS), 'records': n, 'by_platform': counts,
                'tokenizer': 'unicode61 remove_diacritics 2', 'seconds': round(time.time() - started, 1)}
    with open(OUT + '.manifest.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
