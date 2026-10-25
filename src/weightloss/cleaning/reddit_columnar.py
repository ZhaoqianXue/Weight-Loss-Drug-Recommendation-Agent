"""Bounded Arrow transfer and DuckDB operations over SQLite checkpoints.

No text bodies are copied. SQLite remains the atomic durable publication store.
Only explicitly supported aggregate queries are delegated to DuckDB.
"""
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
import sqlite3
import tempfile
from pathlib import Path
from weightloss.cleaning.reddit_runtime import guard_disk
import duckdb
import pyarrow as pa
import polars as pl


def engine():
    temporary = tempfile.TemporaryDirectory(prefix='reddit-duckdb-')
    db = duckdb.connect(config={'threads': 4, 'memory_limit': '512MB',
                               'temp_directory': temporary.name,
                               'max_temp_directory_size': '2GB'})
    return db, temporary


def transfer(source, target, name, query, schema, parameters=()):
    """Copy narrow columns in bounded batches, including correct empty schemas."""
    cursor = sqlite3.Connection.execute(source, query, parameters)
    target.register('_batch', pa.Table.from_batches([], schema=schema))
    target.execute(f'CREATE TABLE {name} AS SELECT * FROM _batch')
    while rows := cursor.fetchmany(25000):
        guard_disk(getattr(source,"output_directory",Path(".")),getattr(source,"reserve_gib",0))
        arrays = [pa.array(col, type=field.type) for col, field in zip(zip(*rows), schema)]
        batch = pa.Table.from_arrays(arrays, schema=schema)
        target.register('_batch', batch)
        target.execute(f'INSERT INTO {name} SELECT * FROM _batch')
    target.unregister('_batch')


def schema(names, integers=()):
    return pa.schema([(n, pa.int64() if n in integers else pa.string()) for n in names.split()])


class Rows:
    def __init__(self, rows): self.rows = iter(rows)
    def __iter__(self): return self
    def __next__(self): return next(self.rows)
    def fetchone(self): return next(self.rows, None)
    def fetchall(self): return list(self.rows)


class ColumnarConnection(sqlite3.Connection):
    """Read-only analytics snapshot, lazily loaded only at final reporting."""
    analytics = None
    scratch = None
    stage = 'v2'
    analytics_enabled = True
    reserve_gib = 0
    output_directory = Path('.')

    def invalidate(self):
        if self.analytics is not None:
            self.analytics.close(); self.analytics = None
            self.scratch.cleanup(); self.scratch = None

    def snapshot(self):
        if self.analytics is not None: return self.analytics
        guard_disk(self.output_directory, self.reserve_gib)
        db, scratch = engine()
        try:
            if self.stage == 'v1':
                transfer(self, db, 'inputs', 'SELECT key,subreddit FROM inputs', schema('key subreddit'))
                db.execute('CREATE TABLE record_counts(input_key VARCHAR,decision VARCHAR,reason VARCHAR,n BIGINT)')
                db.execute('CREATE TABLE text_counts(text_hash VARCHAR,n BIGINT)')
                cursor = sqlite3.Connection.execute(self, 'SELECT input_key,text_hash,text_length,decision,reason FROM records')
                fields = schema('input_key text_hash text_length decision reason', ['text_length'])
                while rows := cursor.fetchmany(25000):
                    frame = pl.from_arrow(pa.Table.from_arrays([pa.array(c,type=f.type) for c,f in zip(zip(*rows),fields)],schema=fields))
                    counts = frame.group_by(['input_key','decision','reason']).len(name='n')
                    db.register('_counts',counts.to_arrow())
                    db.execute('INSERT INTO record_counts SELECT * FROM _counts')
                    texts = frame.filter(pl.col('text_length')>0).group_by('text_hash').len(name='n')
                    db.register('_counts',texts.to_arrow())
                    db.execute('INSERT INTO text_counts SELECT * FROM _counts')
                db.unregister('_counts')
            else:
                evidence_snapshot(self, db, self.stage)
            self.analytics, self.scratch = db, scratch
            return db
        except BaseException:
            db.close(); scratch.cleanup(); raise

    def execute(self, sql, parameters=(), /):
        queries = V1_QUERIES if self.stage == 'v1' else EVIDENCE_QUERIES
        if self.analytics_enabled and sql in queries:
            return Rows(self.snapshot().execute(queries[sql], parameters).fetchall())
        if not sql.lstrip().upper().startswith(('SELECT', 'PRAGMA', 'EXPLAIN')): self.invalidate()
        return super().execute(sql, parameters)

    def executemany(self, sql, parameters, /):
        self.invalidate()
        return super().executemany(sql, parameters)

    def executescript(self, sql, /):
        self.invalidate()
        return super().executescript(sql)

    def rollback(self):
        self.invalidate()
        return super().rollback()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            if exc_type is not None:self.invalidate()

    def close(self):
        self.invalidate()
        return super().close()


def connect(path, stage='v2', backend=None, reserve_gib=0):
    db = sqlite3.connect(path, factory=ColumnarConnection)
    db.stage = stage
    db.analytics_enabled = (backend or ("sqlite" if stage == "v1" else "columnar")) == "columnar"
    db.reserve_gib = reserve_gib
    db.output_directory = Path(path).resolve().parent
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    return db


V1_QUERIES = {
    'SELECT decision,count(*) FROM records GROUP BY decision': 'SELECT decision,sum(n) FROM record_counts GROUP BY decision ORDER BY 1 NULLS FIRST',
    'SELECT reason,count(*) FROM records GROUP BY reason': 'SELECT reason,sum(n) FROM record_counts GROUP BY reason ORDER BY 1 NULLS FIRST',
    'SELECT i.subreddit,r.decision,count(*) FROM records r JOIN inputs i ON i.key=r.input_key GROUP BY 1,2 ORDER BY 1,2': 'SELECT i.subreddit,r.decision,sum(n) FROM record_counts r JOIN inputs i ON i.key=r.input_key GROUP BY 1,2 ORDER BY 1,2',
    'SELECT count(*) FROM (SELECT text_hash FROM records WHERE text_length>0 GROUP BY text_hash HAVING count(*)>1)': 'SELECT count(*) FROM (SELECT text_hash FROM text_counts GROUP BY text_hash HAVING sum(n)>1)',
}
EVIDENCE_QUERIES = {
    'SELECT decision,count(*) FROM cleaned_records GROUP BY decision': 'SELECT decision,sum(n) FROM report_counts GROUP BY decision ORDER BY 1 NULLS FIRST',
    'SELECT reason,count(*) FROM cleaned_records GROUP BY reason': 'SELECT reason,sum(n) FROM report_counts GROUP BY reason ORDER BY 1 NULLS FIRST',
    'SELECT count(*) FROM kept_records': "SELECT coalesce(sum(n),0) FROM report_counts WHERE decision='keep'",
    'SELECT count(*) FROM excluded_records': "SELECT coalesce(sum(n),0) FROM report_counts WHERE decision='exclude'",
    'SELECT subreddit,decision,count(*) FROM cleaned_records GROUP BY subreddit,decision': 'SELECT subreddit,decision,sum(n) FROM report_counts GROUP BY subreddit,decision ORDER BY 1 NULLS FIRST,2 NULLS FIRST',
    'SELECT c.v2_decision,r.decision,count(*) FROM comparisons c JOIN cleaned_records r USING(fullname) GROUP BY c.v2_decision,r.decision': 'SELECT v2_decision,decision,sum(n) FROM transition_counts GROUP BY v2_decision,decision ORDER BY 1 NULLS FIRST,2 NULLS FIRST',
    "SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM comparisons c JOIN cleaned_records r USING(fullname) WHERE c.v2_decision='exclude' AND r.decision='keep'": "SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM transition_counts WHERE v2_decision='exclude' AND decision='keep'",
}


def evidence_snapshot(source, db, stage):
    """One narrow scan, bounded joins, shared v2/v3 and publication aggregates."""
    from weightloss.cleaning.reddit_runtime import chunks, progress
    progress(stage, 'statistics_started')
    transfer(source, db, 'inputs', 'SELECT key,done FROM inputs', schema('key done', ['done']))
    transfer(source, db, 'canonical_text', 'SELECT text_hash,fullname FROM canonical_text', schema('text_hash fullname'))
    view = sqlite3.Connection.execute(source, "SELECT sql FROM sqlite_master WHERE name='cleaned_records'").fetchone()
    overrides = bool(view and 'quality_exclusions' in view[0])
    if overrides:
        transfer(source, db, 'quality_exclusions', 'SELECT fullname,reason FROM quality_exclusions', schema('fullname reason'))
    else:
        db.execute('CREATE TABLE quality_exclusions(fullname VARCHAR,reason VARCHAR)')
    db.execute('CREATE TABLE report_counts(subreddit VARCHAR,decision VARCHAR,reason VARCHAR,n BIGINT)')
    db.execute('CREATE TABLE transition_counts(v2_decision VARCHAR,decision VARCHAR,n BIGINT,old_short BIGINT,old_no_name BIGINT,old_language_veto BIGINT)')
    cursor = sqlite3.Connection.execute(source, 'SELECT fullname,input_key,text_hash,subreddit,quality_decision,quality_reason FROM records')
    fields = schema('fullname input_key text_hash subreddit quality_decision quality_reason')
    comparison_schema = schema('fullname v2_decision old_short old_no_name old_language_veto', ['old_short','old_no_name','old_language_veto'])
    processed = 0
    while rows := cursor.fetchmany(25000):
        guard_disk(source.output_directory,source.reserve_gib)
        batch = pa.Table.from_arrays([pa.array(c,type=f.type) for c,f in zip(zip(*rows),fields)],schema=fields)
        db.register('_records',batch)
        joined = db.execute("""SELECT r.fullname,r.subreddit,
            CASE WHEN q.fullname IS NOT NULL THEN 'exclude' WHEN r.quality_decision='keep' AND c.fullname=r.fullname THEN 'keep' ELSE 'exclude' END AS decision,
            CASE WHEN q.fullname IS NOT NULL THEN q.reason WHEN r.quality_decision='keep' AND c.fullname!=r.fullname THEN 'duplicate_text' ELSE r.quality_reason END AS reason
            FROM _records r JOIN inputs i ON r.input_key=i.key AND i.done=1
            LEFT JOIN canonical_text c ON r.text_hash=c.text_hash
            LEFT JOIN quality_exclusions q ON r.fullname=q.fullname""").fetch_arrow_table()
        frame = pl.from_arrow(joined)
        counts = frame.group_by(['subreddit','decision','reason']).len(name='n')
        db.register('_counts',counts.to_arrow())
        db.execute('INSERT INTO report_counts SELECT * FROM _counts')
        if stage == 'v3':
            # Avoid a full comparisons hash table in every batch: fullname-only
            # lookups use the primary key (the same fix as context prefetch).
            comparisons=[]
            for ids in chunks([r[0] for r in rows]):
                comparisons.extend(sqlite3.Connection.execute(source,
                    'SELECT fullname,v2_decision,old_short,old_no_name,old_language_veto FROM comparisons WHERE fullname IN ('+','.join('?' for _ in ids)+')',ids))
            if comparisons:
                arrow=pa.Table.from_arrays([pa.array(c,type=f.type) for c,f in zip(zip(*comparisons),comparison_schema)],schema=comparison_schema)
                transitions=frame.join(pl.from_arrow(arrow),on='fullname',how='inner').group_by(['v2_decision','decision']).agg(
                    pl.len().alias('n'),pl.col('old_short').sum(),pl.col('old_no_name').sum(),pl.col('old_language_veto').sum())
                db.register('_counts',transitions.to_arrow())
                db.execute('INSERT INTO transition_counts SELECT * FROM _counts')
        processed += len(rows)
        if processed % 250000 == 0: progress(stage, 'statistics_progress', rows=processed)
    db.unregister('_records');db.unregister('_counts')
    for table in ['canonical_text','inputs','quality_exclusions']:db.execute('DROP TABLE '+table)
    progress(stage, 'statistics_completed', rows=processed)


from weightloss.cleaning.reddit_strict import finalize_input
