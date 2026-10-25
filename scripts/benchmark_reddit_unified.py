"""Read-only production sampling; benchmark exact report results in disposable DBs."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from weightloss.cleaning import reddit_columnar as col
from weightloss.cleaning.reddit_runtime import chunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=200000);a=p.parse_args()
    if not 1<=a.rows<=200000:raise ValueError('Bounded sample must be 1–200000 rows')
    reports=[]
    for stage,folder in [('v1','v1'),('v2','v2-strict'),('v3','v3-final')]:
        with tempfile.TemporaryDirectory(prefix='reddit-unified-benchmark-') as td:
            path=Path(td)/'sample.sqlite'
            source=sqlite3.connect(f'file:.cache/reddit-cleaning/{folder}/cleaned.sqlite?mode=ro',uri=True)
            target=sqlite3.connect(path)
            tables=['inputs','records']+([] if stage=='v1' else ['canonical_text'])+(['comparisons','quality_exclusions'] if stage=='v3' else [])
            for table in tables:
                target.execute(source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone()[0])
            inputs=source.execute('SELECT * FROM inputs').fetchall()
            target.executemany('INSERT INTO inputs VALUES ('+','.join('?' for _ in inputs[0])+')',inputs)
            cursor=source.execute('SELECT * FROM records ORDER BY rowid LIMIT ?',(a.rows,));total=0
            while rows:=cursor.fetchmany(2000):
                total+=len(rows);target.executemany('INSERT INTO records VALUES ('+','.join('?' for _ in rows[0])+')',rows)
                names=[d[0] for d in cursor.description]
                if stage!='v1':
                    for hashes in chunks(list({r[names.index('text_hash')] for r in rows})):
                        data=source.execute('SELECT * FROM canonical_text WHERE text_hash IN ('+','.join('?' for _ in hashes)+')',hashes).fetchall()
                        target.executemany('INSERT OR IGNORE INTO canonical_text VALUES (?,?,?)',data)
                if stage=='v3':
                    for ids in chunks([r[0] for r in rows]):
                        for table in ['comparisons','quality_exclusions']:
                            data=source.execute('SELECT * FROM '+table+' WHERE fullname IN ('+','.join('?' for _ in ids)+')',ids).fetchall()
                            if data:target.executemany('INSERT OR IGNORE INTO '+table+' VALUES ('+','.join('?' for _ in data[0])+')',data)
            for sql, in source.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='records' AND sql IS NOT NULL"):target.execute(sql)
            if stage!='v1':
                for sql, in source.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name IN ('cleaned_records','kept_records','excluded_records')"):target.execute(sql)
            target.commit();source.close()
            queries=list(col.V1_QUERIES) if stage=='v1' else [q for q in col.EVIDENCE_QUERIES if stage=='v3' or 'comparisons' not in q]
            start=time.perf_counter();expected=[target.execute(q).fetchall() for q in queries];old=time.perf_counter()-start;target.close()
            target=col.connect(path,stage=stage,backend="columnar");start=time.perf_counter();actual=[target.execute(q).fetchall() for q in queries];new=time.perf_counter()-start;target.close()
            assert actual==expected,stage
            reports.append(dict(stage=stage,rows=total,queries=len(queries),sqlite_seconds=old,unified_seconds=new,speedup=old/new,exact_equal=True))
    result=dict(scope='Report queries on bounded actual production rows; includes columnar transfers and group/join work; excludes text processing. SQLite runs first, so cache order can affect timing.',benchmarks=reports)
    Path('results/reddit-cleaning-fast/unification-benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
