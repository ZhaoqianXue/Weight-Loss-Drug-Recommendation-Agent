"""Coverage-oriented basic denoising before Jev, layered on reddit_noise.

Adds reviewed moderation/bot families, social-phrase combinations, bot commands,
media placeholders, repeated-text flooding and an optional length floor. Full-message rules only: no drug lexicon, symptom dictionary, first-person
or relevance rule. Raw rows and frozen v1/v2/v3/basic runs are never modified;
every exclusion keeps its reason, matched evidence and source feature digest.

Corpus runs use `pipeline`: one scan of the source writes reusable Parquet shards,
then duplicates, the length floor and exports are computed from those shards.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import unicodedata

from . import reddit_basic as basic, reddit_noise as noise

URL = basic.URL
LEXICAL = re.compile(r"[^\W_]+(?:['’.][^\W_]+)*")
SKIN_AND_JOINERS = re.compile('[\U0001F3FB-\U0001F3FF‍︎️⃣]')
ELONGATED = re.compile(r'([a-z])\1{2,}')
_COMPILED = {}


def compact(text):
    """Anchor surface: case-folded words only; URLs and punctuation dropped."""
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'").replace('‘', "'")
    return ' '.join(re.sub(r"[^\w']+", ' ', URL.sub(' ', text)).split())


def lexical_words(text):
    return len(LEXICAL.findall(URL.sub(' ', text)))


def skeleton_digest(text):
    """Digit-preserving duplicate key: case, spacing, punctuation and emoji ignored."""
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'")
    text = re.sub(r"[^\w<>']+", ' ', URL.sub(' <url> ', text))
    return hashlib.blake2b(' '.join(text.split()).encode(), digest_size=16).digest()


def validate_policy(policy):
    basic.validate_policy(policy)
    d = policy.get('denoise')
    if not isinstance(d, dict) or d.get('schema_version') != 1:
        raise ValueError('Missing denoise policy section')
    dup = d['duplicates']
    for key in ('long_min_lexical_words', 'quick_min_lexical_words', 'quick_window_seconds'):
        if type(dup[key]) is not int or dup[key] < 1:
            raise ValueError('Invalid duplicate policy')
    if type(d['min_lexical_words']) is not int or d['min_lexical_words'] < 0:
        raise ValueError('Invalid length floor')
    ids = [a['id'] for a in d['automation_anchors']]
    if len(ids) != len(set(ids)) or any(a['position'] not in ('start', 'anywhere') for a in d['automation_anchors']):
        raise ValueError('Invalid automation anchors')
    compiled(policy)


def compiled(policy):
    """Compile once per denoise section object (policies are treated as immutable)."""
    d = policy['denoise']
    cached = _COMPILED.get(id(d))
    if cached is not None and cached[0] is d:
        return cached[1]
    key = hashlib.sha256(basic.canonical(d).encode()).hexdigest()
    if key not in _COMPILED:
        s = d['social']
        vocative = '(?: (?:' + '|'.join(s['vocatives']) + '))*'
        lead = '(?:(?:' + '|'.join(s['leading_fillers']) + ') )?'
        _COMPILED[key] = dict(
            anchors=[dict(a, regex=re.compile(('^' if a['position'] == 'start' else r'(?:^| )') + a['pattern'] + r'(?: |$)'))
                     for a in d['automation_anchors']],
            bot_disclosure=re.compile(d['bot_disclosure']),
            bot_command=re.compile(d['bot_command'], re.I),
            media=[re.compile(p, re.I) for p in d['media_placeholders']],
            strong=re.compile(lead + '(?:' + '|'.join(s['strong_units']) + ')' + vocative),
            weak=re.compile(lead + '(?:' + '|'.join(s['weak_units'] + s['vocatives']) + ')' + vocative),
            emoticon=re.compile(s['emoticons'], re.I),
            separator=re.compile(s['separators']),
            emoji=set(s['emoji']), max_words=s['max_words'], unit_max=s['unit_max_words'],
            command_words=d['bot_command_max_lexical_words'])
    _COMPILED[id(d)] = (d, _COMPILED[key])
    return _COMPILED[key]


def automation_family(prose, feature, c):
    """Reviewed whole-family anchors on unquoted prose; quotes never trigger."""
    surface = compact('\n\n'.join(s['text'] for s in prose))
    if not surface:
        return None
    if c['bot_disclosure'].search(surface):
        return 'self_declared_bot'
    head = surface[:400]  # start anchors are shorter than this prefix
    for a in c['anchors']:
        if feature.get('kind') not in a.get('kinds', ['comments']):
            continue
        if a.get('subreddits') and feature.get('subreddit', '').casefold() not in a['subreddits']:
            continue
        if a['regex'].search(head if a['position'] == 'start' else surface):
            return a['id']
    return None


def _social_surface(text, c, squeeze):
    text = unicodedata.normalize('NFKC', text).casefold().replace('’', "'").replace('‘', "'")
    text = SKIN_AND_JOINERS.sub('', URL.sub(' ', text))
    text = re.sub(r'(?<!\w)/?u/[\w-]+', ' <user> ', text)
    text = c['emoticon'].sub(' . ', text)
    text = ''.join(' . ' if ch in c['emoji'] else ch for ch in text)
    return ELONGATED.sub(r'\1' * squeeze, text)


def _segment_clause(words, c):
    """Cover a clause with consecutive units; returns (covered, uses_strong_unit)."""
    best = [None] * (len(words) + 1)
    best[0] = False
    for i in range(len(words)):
        if best[i] is None:
            continue
        for j in range(i + 1, min(len(words), i + c['unit_max']) + 1):
            span = ' '.join(words[i:j])
            strong = bool(c['strong'].fullmatch(span))
            if not strong and not c['weak'].fullmatch(span):
                continue
            value = best[i] or strong
            if best[j] is None or (value and not best[j]):
                best[j] = value
    return best[-1]


def social_combination(text, c):
    """Whole text = listed social units (+ listed emoji/emoticons), at least one strong.

    Any digit, unlisted word or unlisted emoji keeps the record. Agreement and
    answer words (same, me too, yes, no), bare adjectives and questions are not
    strong units, so 'Great', 'Same here' or 'Thanks, no nausea now' remain.
    """
    for squeeze in (2, 1):
        surface = _social_surface(text, c, squeeze)
        if any(ch.isdigit() for ch in surface) or len(surface.split()) > c['max_words']:
            return None
        clauses = [' '.join(part.split()) for part in c['separator'].split(surface)]
        clauses = [part for part in clauses if part]
        if not clauses:
            return None
        results = [_segment_clause(part.split(' '), c) for part in clauses]
        if None not in results and any(results):
            return clauses
    return None


def strip_media(text, c):
    for pattern in c['media']:
        text = pattern.sub(' ', text)
    return text


def own_segments(result):
    omit = set(result.get('noise_placeholder_segments', []))
    return [s for i, s in enumerate(result['segments']) if i not in omit]


def _record(feature, policy):
    """Everything the corpus pipeline persists, computed once per record.

    Returns (result, skeleton, text, quoted_text, title). Text is kept only for
    record-level candidates; excluded rows keep their original in the source.
    """
    c = compiled(policy)
    result = noise.prepare(feature, policy)
    result['denoise_policy_revision'] = policy['revision']
    segments = own_segments(result)
    unquoted = [s for s in segments if s['role'] not in ('quote', 'quoted_code')]
    words_raw = '\n\n'.join(s['text'] for s in unquoted)
    words_text = strip_media(words_raw, c)
    result['lexical_words'] = lexical_words(words_text)
    result['own_characters'] = len(words_text.strip())
    if result['decision'] != 'candidate':
        return result, None, None, None, None
    prose = [dict(s, text=strip_media(s['text'], c)) for s in unquoted if s['role'] == 'prose']
    text = '\n\n'.join(s['text'] for s in prose)
    detail, reason = None, None
    family = automation_family(prose, feature, c)
    if family:
        reason, detail = 'moderation_or_bot_template', family
    elif c['bot_command'].search('\n'.join(s['text'] for s in prose)) and result['lexical_words'] <= c['command_words']:
        reason, detail = 'bot_command_only', 'remindme'
    elif words_text != words_raw and not basic.readable(words_text):
        reason, detail = 'media_without_text', 'placeholder_only'  # quote-only rows stay candidates
    elif (feature.get('kind') == 'comments' and len(prose) == len(unquoted)
          and result['lexical_words'] <= c['max_words'] and (units := social_combination(text, c))):
        reason, detail = 'social_phrase_combination', units
    if reason:
        result.update(decision='exclude', reason=reason, denoise_detail=detail)
        return result, None, None, None, None
    quoted = '\n\n'.join(s['text'] for s in segments if s['role'] in ('quote', 'quoted_code')) or None
    title = '\n'.join(s['text'] for s in segments if s.get('field') == 'title') or None
    skeleton = skeleton_digest('\n\n'.join(s['text'] for s in segments))
    return result, skeleton, words_text.strip(), quoted, title


def prepare_record(feature, policy):
    """Record-level decision; duplicates need the corpus pass (see pipeline)."""
    result, skeleton, _, _, _ = _record(feature, policy)
    return result, skeleton


def prepare(feature, policy, duplicate=None):
    """Full decision. `duplicate` is this row's corpus-index entry, if any."""
    result, _ = prepare_record(feature, policy)
    if result['decision'] != 'candidate':
        return result
    if duplicate:
        result.update(decision='exclude', reason='duplicate_' + duplicate['rule'], denoise_detail=duplicate)
    elif result['lexical_words'] < policy['denoise']['min_lexical_words']:
        result.update(decision='exclude', reason='below_min_lexical_words',
                      denoise_detail=dict(min_lexical_words=policy['denoise']['min_lexical_words']))
    return result


# ---------------------------------------------------------------- corpus pipeline
#
# One scan of the frozen source: each worker parses every JSON row once, runs the
# record rules once and writes an Arrow/Parquet shard with the reusable results
# (reason, words, duplicate key, per-row source digest, candidate text). Duplicate
# detection, the length floor, counts and exports then read only those shards
# with Polars; the 33 GB source is never rescanned for later stages.

READY = 'ready_for_semantic_screen'
_WORKER = {}


def _shard_schema():
    import pyarrow as pa
    return pa.schema([('rowid', pa.int64()), ('fullname', pa.string()), ('kind', pa.string()),
                      ('subreddit', pa.string()), ('created_utc', pa.int64()), ('record_reason', pa.string()),
                      ('record_detail', pa.string()), ('lexical_words', pa.int32()), ('own_characters', pa.int32()),
                      ('context_parent', pa.string()), ('context_thread', pa.string()),
                      ('skeleton', pa.binary(16)), ('source_feature_sha256', pa.binary(32)),
                      ('text', pa.string()), ('quoted_text', pa.string()), ('title', pa.string())])


def _init_worker(source_dir, policy):
    db = sqlite3.connect(f'file:{Path(source_dir) / "cleaned.sqlite"}?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    _WORKER.update(db=db, policy=policy)


def _scan_chunk(task):
    """Worker: read one rowid range, write one shard atomically, return small stats."""
    import resource
    import time
    import pyarrow as pa
    import pyarrow.parquet as pq
    chunk, lo, hi, shard = task
    started = time.perf_counter()
    names = [f.name for f in _shard_schema()]
    cols = {name: [] for name in names}
    for rid, ident, stamp, payload in _WORKER['db'].execute(
            'SELECT rowid,fullname,created_utc,features_json FROM records WHERE rowid BETWEEN ? AND ?', (lo, hi)):
        feature = json.loads(payload)
        if feature['fullname'] != ident:
            raise ValueError('Source identity mismatch')
        result, skeleton, text, quoted, title = _record(feature, _WORKER['policy'])
        refs = [r['fullname'] for r in result.get('context_references', [])]
        detail = result.get('denoise_detail')
        values = (rid, ident, feature['kind'], feature['subreddit'], stamp, result['reason'],
                  None if detail is None else basic.canonical(detail), result['lexical_words'],
                  result['own_characters'], refs[0] if refs else None,
                  next((r for r in refs if r.startswith('t3_')), None), skeleton,
                  hashlib.sha256(payload.encode()).digest(), text, quoted, title)
        for name, value in zip(names, values):
            cols[name].append(value)
    table = pa.table(cols, schema=_shard_schema())
    temp = shard.with_suffix('.tmp')
    pq.write_table(table, temp, compression='zstd')
    temp.replace(shard)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return dict(chunk=chunk, rows=table.num_rows, seconds=time.perf_counter() - started,
                worker_max_rss_mib=peak / (2**20 if sys.platform == 'darwin' else 2**10))


class _TreeMemory:
    """Samples summed RSS of this process and its descendants (ps, every 0.5 s)."""
    def __init__(self):
        import threading
        self.peak_mib, self.peak_main_mib, self._stop = 0.0, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        import os
        import subprocess
        rows = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,rss='], capture_output=True, text=True).stdout.split('\n')
        children, rss = {}, {}
        for row in rows:
            parts = row.split()
            if len(parts) == 3:
                pid, ppid, kb = map(int, parts)
                children.setdefault(ppid, []).append(pid)
                rss[pid] = kb
        total, stack, main = 0, [os.getpid()], rss.get(os.getpid(), 0)
        while stack:
            pid = stack.pop()
            total += rss.get(pid, 0)
            stack.extend(children.get(pid, []))
        self.peak_mib = max(self.peak_mib, total / 1024)
        self.peak_main_mib = max(self.peak_main_mib, main / 1024)

    def _loop(self):
        while not self._stop.wait(0.5):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def _source_identity(path, cache):
    """Full SHA-256 once per output directory; reused while size/mtime/inode match."""
    st = path.stat()
    key = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino)
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached['key'] == key:
            return dict(cached, reused=True)
        raise ValueError('Source database changed since this run started; use a new output directory')
    identity = dict(key=key, sha256=basic.file_digest(path))
    basic.write_json(cache, identity)
    return dict(identity, reused=False)


def _guard(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    spec = json.loads((source_dir / 'run-spec.json').read_text())
    baseline = Path(spec['baseline_directory']).resolve()
    production = Path(json.loads((baseline / 'run-spec.json').read_text())['production_directory']).resolve()
    for protected in (source_dir, baseline, production):
        if output_dir == protected or output_dir.is_relative_to(protected) or protected.is_relative_to(output_dir):
            raise ValueError('Output must be separate from source and frozen runs')
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir / 'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    wal = source_dir / 'cleaned.sqlite-wal'
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Source has an uncheckpointed WAL; requires a frozen snapshot')
    output_dir.mkdir(parents=True, exist_ok=True)
    return source_dir, output_dir


def _locks(stack, source_dir, output_dir):
    for path, mode in ((source_dir / '.cleaning.lock', fcntl.LOCK_SH), (output_dir / '.cleaning.lock', fcntl.LOCK_EX)):
        handle = stack.enter_context(path.open('r' if mode == fcntl.LOCK_SH else 'a'))
        try:
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Cleaner is active; checkpoint preserved') from error


def _scan_policy(policy):
    """Shards depend on record-level rules only; duplicates and the floor come later."""
    reduced = json.loads(json.dumps(policy))
    reduced['denoise'].pop('min_lexical_words')
    reduced['denoise'].pop('duplicates')
    for key in ('commit_every', 'reserve_gib'):
        reduced.pop(key, None)
    return reduced


def _dedupe(records, policy, pl):
    """Earliest identical candidate stays; later copies point to it (Polars sort + windows)."""
    dup = policy['denoise']['duplicates']
    cand = (pl.scan_parquet(records).filter(pl.col('record_reason') == READY)
            .select('rowid', 'created_utc', 'lexical_words', 'skeleton').collect())
    cand = cand.sort(['skeleton', 'created_utc', 'rowid']).with_columns(
        pl.col('rowid').first().over('skeleton').alias('duplicate_of_rowid'),
        pl.len().over('skeleton').alias('copies'),
        (pl.col('created_utc') - pl.col('created_utc').shift(1).over('skeleton')).alias('seconds_after_previous'),
        pl.int_range(pl.len()).over('skeleton').alias('position'))
    found = cand.filter(pl.col('position') > 0).with_columns(
        pl.when(pl.col('lexical_words') >= dup['long_min_lexical_words']).then(pl.lit('long_text'))
        .when((pl.col('lexical_words') >= dup['quick_min_lexical_words'])
              & (pl.col('seconds_after_previous') <= dup['quick_window_seconds'])).then(pl.lit('quick_repeat'))
        .alias('rule')).filter(pl.col('rule').is_not_null())
    names = (pl.scan_parquet(records).select(pl.col('rowid').alias('duplicate_of_rowid'), pl.col('fullname').alias('duplicate_of'))
             .join(found.select('duplicate_of_rowid').unique().lazy(), on='duplicate_of_rowid').collect())
    return len(cand), found.join(names, on='duplicate_of_rowid').select(
        'rowid', 'rule', 'duplicate_of', 'duplicate_of_rowid', 'copies', 'seconds_after_previous')


def _finalize_shard(shard, duplicates, floor, out_decisions, out_candidates, pl):
    df = pl.read_parquet(shard).join(duplicates, on='rowid', how='left')
    ready = pl.col('record_reason') == READY
    dup = pl.col('rule').is_not_null()
    short = pl.col('lexical_words') < floor
    df = df.with_columns(
        pl.when(~ready).then(pl.col('record_reason'))
        .when(dup).then(pl.lit('duplicate_') + pl.col('rule'))
        .when(short).then(pl.lit('below_min_lexical_words'))
        .otherwise(pl.lit(READY)).alias('reason'),
        pl.when(~ready).then(pl.col('record_detail'))
        .when(dup).then(pl.struct('rule', 'duplicate_of', 'copies', 'seconds_after_previous').struct.json_encode())
        .when(short).then(pl.lit(json.dumps(dict(min_lexical_words=floor))))
        .alias('detail'))
    df = df.with_columns(pl.when(pl.col('reason') == READY).then(pl.lit('candidate')).otherwise(pl.lit('exclude')).alias('decision'))
    keep = ['rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'decision', 'reason', 'detail', 'lexical_words',
            'own_characters', 'context_parent', 'context_thread', 'source_feature_sha256']
    df.select(keep).write_parquet(out_decisions, compression='zstd')
    df.filter(pl.col('decision') == 'candidate').select(
        'rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'lexical_words', 'text', 'quoted_text', 'title',
        'context_parent', 'context_thread', 'source_feature_sha256').write_parquet(out_candidates, compression='zstd')


def pipeline(source_dir, output_dir, policy, workers=6, chunk_rows=100000, max_chunks=None):
    """Resumable scan → dedupe → finalize. Only the scan reads the source database."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    import polars as pl
    validate_policy(policy)
    started = time.perf_counter()
    source_dir, output_dir = _guard(source_dir, output_dir)
    metrics = {}
    with ExitStack() as stack:
        _locks(stack, source_dir, output_dir)
        source = sqlite3.connect(f'file:{source_dir / "cleaned.sqlite"}?mode=ro', uri=True)
        stack.callback(source.close)
        if source.execute('SELECT count(*) FROM inputs WHERE done!=1').fetchone()[0]:
            raise ValueError('Requires a completed frozen v2 source')
        expected = source.execute('SELECT sum(expected_rows) FROM inputs').fetchone()[0] or 0
        low, high = source.execute('SELECT min(rowid), max(rowid) FROM records').fetchone()
        if (low, high) != (1, expected):
            raise ValueError('Requires dense frozen source rowids')
        code = {Path(p).name: basic.file_digest(p) for p in (basic.__file__, noise.__file__, __file__)}
        stat = (source_dir / 'cleaned.sqlite').stat()
        spec = dict(schema_version=2, source=str(source_dir), scan_policy=_scan_policy(policy), code_sha256=code,
                    source_key=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino),
                    source_spec_sha256=basic.file_digest(source_dir / 'run-spec.json'), chunk_rows=chunk_rows)
        path = output_dir / 'run-spec.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Rules/source/chunking changed; use a new output directory')
        if not path.exists():
            basic.write_json(path, spec)
        (output_dir / 'source').mkdir(exist_ok=True)
        for p in (basic.__file__, noise.__file__, __file__):
            shutil.copyfile(p, output_dir / 'source' / Path(p).name)
        records = output_dir / 'records'
        records.mkdir(exist_ok=True)
        tasks = [(i, lo, min(lo + chunk_rows - 1, high), records / f'part-{i:05d}.parquet')
                 for i, lo in enumerate(range(1, high + 1, chunk_rows))]
        todo = [t for t in tasks if not t[3].exists()][:max_chunks]
        # 1. scan: the only stage that reads the 33 GB source; SHA-256 runs beside it.
        from concurrent.futures import ThreadPoolExecutor
        scan_start = time.perf_counter()
        stats = []
        with _TreeMemory() as memory, ThreadPoolExecutor(1) as hasher:
            identity = hasher.submit(_source_identity, source_dir / 'cleaned.sqlite', output_dir / 'source-identity.json')
            if todo:
                with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(source_dir), policy)) as pool:
                    for result in pool.map(_scan_chunk, todo):
                        if shutil.disk_usage(output_dir).free < policy['reserve_gib'] * 2**30:
                            raise ValueError('Disk reserve reached; completed shards retained')
                        stats.append(result)
            identity = identity.result()
        metrics['scan'] = dict(seconds=time.perf_counter() - scan_start, shards_written=len(stats),
                               rows_scanned=sum(s['rows'] for s in stats), workers=workers,
                               peak_tree_rss_mib=memory.peak_mib, peak_main_rss_mib=memory.peak_main_mib,
                               max_worker_rss_mib=max((s['worker_max_rss_mib'] for s in stats), default=None),
                               source_sha256_reused=identity['reused'])
        if any(not t[3].exists() for t in tasks):
            report = dict(status='partial', shards_done=sum(t[3].exists() for t in tasks), shards_total=len(tasks), metrics=metrics)
            basic.write_json(output_dir / 'manifest.json', report)
            return report
        shard_rows = sum(pl.scan_parquet(t[3]).select(pl.len()).collect().item() for t in tasks)
        if shard_rows != expected:
            raise ValueError('Shard row count differs from the source')
        # 2. dedupe over persisted keys (no source access).
        stage = time.perf_counter()
        glob = str(records / 'part-*.parquet')
        with _TreeMemory() as memory:
            candidates_before, duplicates = _dedupe(glob, policy, pl)
            duplicates.write_parquet(output_dir / 'duplicates.parquet', compression='zstd')
        metrics['dedupe'] = dict(seconds=time.perf_counter() - stage, record_level_candidates=candidates_before,
                                 duplicate_rows=duplicates.height, peak_tree_rss_mib=memory.peak_mib)
        # 3. finalize shard by shard: reasons, length floor, compact decisions, candidate export.
        stage = time.perf_counter()
        floor = policy['denoise']['min_lexical_words']
        for name in ('decisions', 'candidates'):
            target = output_dir / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir()
        join = duplicates.select('rowid', 'rule', 'duplicate_of', 'copies', 'seconds_after_previous')
        with _TreeMemory() as memory:
            for t in tasks:
                _finalize_shard(t[3], join, floor, output_dir / 'decisions' / t[3].name,
                                output_dir / 'candidates' / t[3].name, pl)
            decisions = pl.scan_parquet(str(output_dir / 'decisions' / '*.parquet'))
            reasons = dict(decisions.group_by('reason').len().collect().iter_rows())
            by_kind = {k: dict(v) for k, v in _nest(decisions.group_by('kind', 'reason').len().collect().iter_rows()).items()}
            lexical = dict(decisions.filter(pl.col('reason').is_in([READY, 'below_min_lexical_words']))
                           .group_by(pl.col('lexical_words').clip(0, 101)).len().collect().iter_rows())
        metrics['finalize'] = dict(seconds=time.perf_counter() - stage, peak_tree_rss_mib=memory.peak_mib)
        metrics['total_seconds'] = time.perf_counter() - started
        total = sum(reasons.values())
        report = dict(schema_version=2, generated_at=datetime.now(timezone.utc).isoformat(), status='full_denoise_complete',
                      policy_revision=policy['revision'], min_lexical_words=floor, source_records=expected,
                      candidates=reasons.get(READY, 0), excluded=total - reasons.get(READY, 0),
                      retention_pct=100 * reasons.get(READY, 0) / total, reasons=reasons, by_kind=by_kind,
                      candidates_by_min_lexical_words={x: sum(n for w, n in lexical.items() if w >= x) for x in range(0, 101)},
                      source_database_sha256=identity['sha256'], source_identity_key=identity['key'],
                      outputs=dict(records='records/part-*.parquet', duplicates='duplicates.parquet',
                                   decisions='decisions/part-*.parquet', candidates='candidates/part-*.parquet'),
                      metrics=metrics, independent_accuracy=None, clinical_eligibility='not_assessed',
                      semantic_screen='pending')
        basic.write_json(output_dir / 'manifest.json', report)
        return report


def _nest(rows):
    out = {}
    for kind, reason, n in rows:
        out.setdefault(kind, {})[reason] = n
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('.cache/reddit-cleaning/v2-strict'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-denoise.json'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--chunk-rows', type=int, default=100000)
    parser.add_argument('--max-chunks', type=int, help='scan at most this many new shards, then stop (resumable)')
    args = parser.parse_args()
    report = pipeline(args.source_dir, args.output_dir, json.loads(args.policy.read_text()),
                      args.workers, args.chunk_rows, args.max_chunks)
    print(json.dumps({k: v for k, v in report.items() if k != 'candidates_by_min_lexical_words'}, indent=2))


if __name__ == '__main__':
    main()
