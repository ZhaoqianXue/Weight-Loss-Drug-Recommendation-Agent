"""OpenAI Batch API steps for compact extraction, on the same run ledger and request bodies as the synchronous runner.

  prepare  write request files for records that need an attempt (first attempts and capped retries), <= 50,000
           requests and 200 MB per file (gzipped on disk; the sha256 is of the exact uploaded bytes); no network
  submit   per prepared part: verify its sha256; record a batch that already exists on the account (matched on run,
           part and payload sha in the metadata) instead of creating one; check the budget (actual spend, uncertain
           charges and the conservative caps of every batch still open or of unknown outcome, plus this part's cap); log
           intents, upload, create. After an uncertain create error the account is reconciled and the part is recorded if
           its batch exists; otherwise the step stops without re-creating. A create the API refuses with an HTTP 4xx
           created nothing: it is logged as rejected; a capacity refusal (enqueued-token limit, 429) is back-pressure.
           A failed upload created nothing either: a transient one (SDK connection loss or timeout, HTTP 408/409/429/5xx)
           is back-pressure and the same part is uploaded again; a definite 4xx or an unclassified error stops.
  poll     snapshot the current batch of every uncollected part (optionally with a thread pool)
  collect  download output and error files of finished batches (gzipped) and evaluate each line by custom_id: lines
           may arrive in any order, a custom_id is evaluated once, and requests without a line become missing_result
           anomalies. A batch that failed on the enqueued-token limit processed nothing: it is requeued, never
           evaluated, and the same part goes out again after a back-off. The next prepare turns retriable anomalies
           into a retry part while attempts remain.
  roll     a rolling window over the steps above: at most `window` batches of the run in flight; a finished batch is
           collected and the next prepared part submitted; retriable anomalies become retry parts. One driver per run
           (lock file); stops at a deadline or a STOP file and resumes from the ledger. The whole account is listed
           once per start; each submit checks only the most recent page.

Use a client with max_retries=0 for submit so the SDK never resends an upload or a paid create on its own.
Scale notes (2026-10-01, full-batch-p7c1-luna-low-20261001-01a0f4c2): the ledger is read incrementally (see runner.py),
a part's rows are written with one fsync per file, and the budget is actual spend plus open caps, so caps of collected
batches are never counted again.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import hashlib
import json
import os
import re
import time
import zlib

from .runner import custom_id, needs_attempt, now, split_custom_id
from .structured import (ExtractionAnomaly, batch_line, compact_json, http_retriable, mask, request_cap_usd,
                         sha256_text)

MAX_REQUESTS = 50_000
MAX_BYTES = 200 * 1024 * 1024
FINAL = ('completed', 'failed', 'expired', 'cancelled')
TIMESTAMPS = ('created_at', 'in_progress_at', 'finalizing_at', 'completed_at', 'failed_at', 'expired_at', 'cancelling_at', 'cancelled_at')
TAG = 'weightloss-compact'
CAPACITY_CODES = frozenset({'token_limit_exceeded'})
CAPACITY_TEXT = re.compile(r'enqueued token limit|token_limit_exceeded|too many (?:enqueued|queued)', re.I)


class Backpressure(RuntimeError):
    """The API refused new work for capacity (enqueued-token limit); nothing was created. Retry after a back-off."""


def parts(ledger):
    return ledger.rows('batch_parts')


def requeued_ids(ledger):
    return {r['batch_id'] for r in ledger.rows('batch_requeues')}


def current_batches(ledger):
    """part -> its current batch row: the latest batch created for the part, unless that batch was requeued after a
    capacity failure (the part then has no current batch and goes out again)."""
    gone, last = requeued_ids(ledger), {}
    for r in ledger.rows('batches'):
        last[r['part']] = r
    return {k: r for k, r in last.items() if r['batch_id'] not in gone}


def unresolved_creates(ledger):
    """part -> create intent whose outcome is unknown: no batch was recorded for its uploaded file and the API did not
    refuse it. Only a remote match can resolve it; it is never re-created automatically."""
    recorded = {r.get('input_file_id') for r in ledger.rows('batches')}
    intents = ledger.rows('batch_intents')
    refused = {r.get('input_file_id') for r in intents if r['step'] == 'create_rejected'}
    out = {}
    for r in intents:
        if r['step'] == 'create' and r.get('input_file_id') not in recorded and r.get('input_file_id') not in refused:
            out[r['part']] = r
    return out


def open_caps_usd(ledger):
    """Conservative caps of work that may still be charged: the current batches of uncollected parts, plus parts whose
    create was attempted with an unknown outcome. A recorded batch counts once; a collected part counts its actual
    usage instead (spent_and_uncertain)."""
    collected = {r['part'] for r in ledger.rows('batch_collected')}
    caps = {p['part']: p['cap_usd'] for p in parts(ledger)}
    current = current_batches(ledger)
    unknown = set(unresolved_creates(ledger)) - set(current)
    return sum(caps.get(k, 0.0) for k in current if k not in collected) + sum(caps.get(k, 0.0) for k in unknown)


def committed_usd(ledger):
    """Actual spend and uncertain charges of this run (all modes) plus open caps: the amount a new part is checked against."""
    spent, uncertain = ledger.spent_and_uncertain()
    return spent + uncertain + open_caps_usd(ledger)


def open_custom_ids(ledger):
    """Custom ids in parts whose batch has not been collected yet (prepared, submitted, running or requeued)."""
    collected = {r['part'] for r in ledger.rows('batch_collected')}
    return {cid for p in parts(ledger) if p['part'] not in collected for cid in p['custom_ids']}


def read_part(ledger, part):
    """The exact bytes of a prepared part, verified against its sha256; RuntimeError when unreadable or changed."""
    try:
        raw = (ledger.dir/part['path']).read_bytes()
        data = gzip.decompress(raw) if part.get('gzip') else raw
    except (OSError, EOFError, zlib.error) as error:
        raise RuntimeError(f"{part['part']}: request file unreadable ({type(error).__name__}); nothing submitted") from None
    if hashlib.sha256(data).hexdigest() != part['sha256']:
        raise RuntimeError(f"{part['part']}: request file changed since prepare; nothing submitted")
    return data


def _write_atomic(path, data):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def prepare(ledger, *, max_attempts=2, max_requests=MAX_REQUESTS, max_bytes=MAX_BYTES, record_ids=None, compress=True):
    if not 0 < max_requests <= MAX_REQUESTS or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError('batch files are limited to 50,000 requests and 200 MB')
    busy = {split_custom_id(cid)[0] for cid in open_custom_ids(ledger)}
    states = ledger.index()['states']
    wanted = set(record_ids) if record_ids is not None else None
    policy = ledger.manifest.get('retry_policy')
    todo = [inp for rid, inp in ledger.inputs().items()
            if rid not in busy and (wanted is None or rid in wanted) and needs_attempt(states[rid], max_attempts, policy)]
    folder = ledger.dir/'batch'
    folder.mkdir(exist_ok=True)
    number = len(parts(ledger))
    model = ledger.manifest['model']
    created, lines, size = [], [], 0
    checked = False

    def flush():
        nonlocal number, lines, size
        number += 1
        key = f'part-{number:04d}'
        data = ''.join(line for _, line, _, _ in lines).encode()
        name = f'{key}.jsonl.gz' if compress else f'{key}.jsonl'
        _write_atomic(folder/name, gzip.compress(data, compresslevel=3) if compress else data)
        row = {'part': key, 'path': f'batch/{name}', 'gzip': compress, 'sha256': hashlib.sha256(data).hexdigest(), 'requests': len(lines),
               'bytes': len(data), 'cap_usd': round(sum(cap for _, _, cap, _ in lines), 9),
               'custom_ids': [cid for cid, _, _, _ in lines], 'created_utc': now()}
        at = now()
        ledger.append_many('requests', [{'custom_id': cid, 'record_id': split_custom_id(cid)[0], 'attempt': split_custom_id(cid)[1],
                                         'mode': 'batch', 'body_sha256': body_sha, 'at': at, 'part': key}
                                        for cid, _, _, body_sha in lines])
        ledger.append('batch_parts', row)
        created.append({k: row[k] for k in ('part', 'requests', 'bytes', 'sha256', 'cap_usd')})
        lines, size = [], 0

    for inp in todo:
        cid = custom_id(inp.record_id, states[inp.record_id]['attempts'] + 1)
        body = ledger.request_body(inp)
        body_json = compact_json(body)  # serialized once: the line, the cap and the logged sha256 all use it
        line = '{"custom_id":' + json.dumps(cid, ensure_ascii=False) + ',"method":"POST","url":"/v1/responses","body":' + body_json + '}\n'
        if not checked:
            if line != batch_line(cid, body):
                raise RuntimeError('fast batch line differs from batch_line(); nothing prepared')
            checked = True
        width = len(line.encode())
        if width > max_bytes:
            raise ValueError(f'{inp.record_id}: one request exceeds the batch file size limit')
        if lines and (len(lines) >= max_requests or size + width > max_bytes):
            flush()
        lines.append((cid, line, request_cap_usd(model, body, batch=True, size=len(body_json.encode())), sha256_text(body_json)))
        size += width
    if lines:
        flush()
    return created


def _page(listing):
    return getattr(listing, 'data', listing)


def remote_batches(client, ledger, pages=1):
    """Batches on the account for this run, keyed by part and matched on run id and payload sha. pages=1 reads only the
    most recent page (a create of this driver is always recent); pages=None walks the whole account."""
    known = {p['part']: p['sha256'][:16] for p in parts(ledger)}
    listing = client.batches.list(limit=100)
    found = {}
    for b in (listing if pages is None else _page(listing)):
        md = b.metadata or {}
        if md.get('tag') == TAG and md.get('run') == ledger.run_id and known.get(md.get('part')) == md.get('sha256'):
            found.setdefault(md['part'], []).append({'batch_id': b.id, 'input_file_id': b.input_file_id, 'status': b.status})
    return found


def transient_error(error):
    """A failure that created nothing and may pass on a later try: SDK connection loss or timeout
    (openai.APIConnectionError, which includes APITimeoutError) or an HTTP 408/409/429/5xx answer. A definite 4xx, or
    any exception the SDK does not classify, is not transient."""
    try:
        import openai
    except ImportError:  # no SDK: only an explicit HTTP status can classify the error
        openai = None
    if openai is not None and isinstance(error, openai.APIConnectionError):
        return True
    status = getattr(error, 'status_code', None)
    return isinstance(status, int) and (status in (408, 409, 429) or status >= 500)


def capacity_refusal(error):
    status = getattr(error, 'status_code', None)
    text = f"{getattr(error, 'code', '') or ''} {getattr(error, 'message', '') or error}"
    return status == 429 or bool(CAPACITY_TEXT.search(text))


def submit(ledger, client, *, budget_usd, only=None, should_stop=None, remote=None):
    current = current_batches(ledger)
    by_part = {p['part']: p for p in parts(ledger)}
    committed = committed_usd(ledger)
    gone = requeued_ids(ledger)
    remote = remote_batches(client, ledger) if remote is None else remote
    live = {k: [b for b in v if b['batch_id'] not in gone] for k, v in remote.items()}
    repeated = {k: v for k, v in live.items() if len(v) > 1}
    if repeated:
        ledger.append('batch_errors', {'step': 'duplicate check', 'duplicates': repeated, 'at': now()})
        raise RuntimeError(f'the account holds several batches for {sorted(repeated)}; nothing submitted, decide manually')
    recorded = {r['batch_id'] for r in ledger.rows('batches')}
    unknown = unresolved_creates(ledger)
    done, stopped = [], False
    for key, part in by_part.items():
        if key in current or (only is not None and key not in only):
            continue
        data = read_part(ledger, part)
        row = {'part': key, 'sha256': part['sha256'], 'requests': part['requests'], 'cap_usd': part['cap_usd']}
        match = [b for b in live.get(key, []) if b['batch_id'] not in recorded]
        if match:
            ledger.append('batches', {**row, **match[0], 'recovered_from': 'account metadata (run, part, payload sha)', 'at': now()})
            if key not in unknown:  # a reconciled unknown create was already inside the open caps
                committed += part['cap_usd']
            done.append(key)
            continue
        if key in unknown:
            # an earlier create may have succeeded without being recorded; only a remote match can resolve it
            ledger.append('batch_errors', {**row, 'step': 'unresolved create intent', 'resolution': 'unknown', 'at': now()})
            raise RuntimeError(f'{key}: an earlier create was attempted and no matching batch is visible on the account; '
                               'state unknown, not re-created. Check the account before any manual resubmission.')
        if committed + part['cap_usd'] > budget_usd:
            raise RuntimeError(f'stop before {key}: committed {committed:.4f} USD (spend, uncertain and open caps) plus its cap '
                               f'{part["cap_usd"]:.4f} would exceed the budget {budget_usd}')
        if should_stop and should_stop():  # re-checked before every paid upload/create (deadline, STOP file)
            stopped = True
            break
        metadata = {'tag': TAG, 'run': ledger.run_id, 'part': key, 'sha256': part['sha256'][:16]}
        ledger.append('batch_intents', {'part': key, 'step': 'upload', 'at': now()})
        started = time.time()
        try:
            upload = client.files.create(file=(f'{ledger.run_id}-{key}.jsonl', data), purpose='batch')
        except Exception as error:  # no create was called, so nothing is billed
            transient = transient_error(error)
            ledger.append('batch_errors', {**row, 'step': 'upload', 'error': safe(error), 'transient': transient, 'at': now()})
            if transient:  # connection loss, timeout, 408/409/429/5xx: the same part is uploaded again after a back-off
                raise Backpressure(f'{key}: upload failed ({safe(error)["type"]}); nothing created; uploaded again after a back-off') from None
            raise RuntimeError(f'{key}: upload failed ({safe(error)["type"]}); nothing created') from None
        uploaded = time.time()
        if should_stop and should_stop():  # re-checked after a slow upload, before the paid create; an unused upload is not billed
            ledger.append('batch_intents', {'part': key, 'step': 'upload_unused', 'input_file_id': upload.id, 'at': now()})
            stopped = True
            break
        ledger.append('batch_intents', {'part': key, 'step': 'create', 'input_file_id': upload.id, 'at': now()})
        try:
            b = client.batches.create(input_file_id=upload.id, endpoint='/v1/responses', completion_window='24h',
                                      metadata={**metadata, 'input_file_id': upload.id})
        except Exception as error:
            status = getattr(error, 'status_code', None)
            if status is not None and 400 <= status < 500:  # the API answered and refused: nothing was created
                ledger.append('batch_intents', {'part': key, 'step': 'create_rejected', 'input_file_id': upload.id, 'status': status,
                                                'error': safe(error), 'at': now()})
                if capacity_refusal(error):
                    raise Backpressure(f'{key}: create refused for capacity (HTTP {status}); nothing created, retried after a back-off') from None
                raise RuntimeError(f'{key}: create refused (HTTP {status}, {safe(error)["message"][:120]}); nothing created') from None
            # the batch may exist (timeout, 5xx): reconcile, never re-create here
            try:
                match = [x for x in remote_batches(client, ledger).get(key, []) if x['input_file_id'] == upload.id]
            except Exception:
                match = []
            if match:
                ledger.append('batches', {**row, **match[0], 'recovered_from': 'reconciled after an uncertain create error',
                                          'create_error': safe(error), 'at': now()})
                committed += part['cap_usd']
                done.append(key)
                continue
            ledger.append('batch_errors', {**row, 'step': 'create', 'input_file_id': upload.id, 'error': safe(error), 'at': now()})
            raise RuntimeError(f'{key}: create outcome uncertain and not found on the account; not re-created. '
                               'Wait, then submit again: it checks the account before creating anything.') from None
        ledger.append('batches', {**row, 'batch_id': b.id, 'input_file_id': upload.id, 'status': b.status, 't': round(time.time(), 3),
                                  'upload_s': round(uploaded - started, 3), 'create_s': round(time.time() - uploaded, 3), 'at': now()})
        committed += part['cap_usd']
        done.append(key)
    return {'submitted': done, 'committed_usd': round(committed, 9), 'budget_usd': budget_usd, 'stopped_early': stopped}


def _errors(b):
    e = getattr(b, 'errors', None)
    data = e.get('data') if isinstance(e, dict) else getattr(e, 'data', None)
    out = []
    for x in data or []:
        get = x.get if isinstance(x, dict) else (lambda k, x=x: getattr(x, k, None))
        out.append({'code': get('code'), 'message': mask(get('message') or '')[:300], 'line': get('line')})
    return out


def poll(ledger, client, only=None, workers=1):
    """Snapshot the current batch of every uncollected part (or of `only`)."""
    collected = {r['part'] for r in ledger.rows('batch_collected')}
    rows = [r for k, r in current_batches(ledger).items() if k not in collected and (only is None or k in only)]

    def one(row):
        try:
            b = client.batches.retrieve(row['batch_id'])
            counts = getattr(b, 'request_counts', None)
            return {'part': row['part'], 'batch_id': b.id, 'status': b.status,
                    'request_counts': counts.model_dump() if hasattr(counts, 'model_dump') else counts,
                    'output_file_id': getattr(b, 'output_file_id', None), 'error_file_id': getattr(b, 'error_file_id', None),
                    'errors': _errors(b), **{k: getattr(b, k, None) for k in TIMESTAMPS}, 'at': now()}
        except Exception as error:
            return {'part': row['part'], 'batch_id': row['batch_id'], 'error': safe(error), 'at': now()}

    if workers > 1 and len(rows) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
            snapshots = list(pool.map(one, rows))
    else:
        snapshots = [one(r) for r in rows]
    ledger.append_many('batch_snapshots', snapshots)
    return snapshots


def capacity_failure(snap):
    """A batch that failed validation on the enqueued-token limit: nothing was processed or billed."""
    if snap.get('status') != 'failed' or snap.get('output_file_id'):
        return False
    counts = snap.get('request_counts') or {}
    if (counts.get('completed') or 0) or (counts.get('failed') or 0):
        return False
    return any(e.get('code') in CAPACITY_CODES or CAPACITY_TEXT.search(e.get('message') or '') for e in snap.get('errors') or [])


def collect(ledger, client, snapshots=None, recover=True, workers=1):
    """Download and evaluate every finished, uncollected batch; returns counts per part (a requeued capacity failure
    reports requeued=1). `snapshots` reuses a fresh poll; recover=False skips the recovery pass (a driver that already
    recovered at its start)."""
    if recover:
        ledger.recover()  # bodies saved by an interrupted collect are evaluated once, without a second usage row
    latest = {}
    for snap in (poll(ledger, client) if snapshots is None else snapshots):
        latest[snap['part']] = snap
    by_part = {p['part']: p for p in parts(ledger)}
    collected = {r['part'] for r in ledger.rows('batch_collected')}
    current = current_batches(ledger)
    report, todo = {}, []
    for key, snap in latest.items():
        if key in collected or snap.get('status') not in FINAL or current.get(key, {}).get('batch_id') != snap.get('batch_id'):
            continue
        if capacity_failure(snap):
            ledger.append('batch_requeues', {'part': key, 'batch_id': snap['batch_id'], 'status': snap['status'],
                                             'errors': snap.get('errors'), 'reason': 'capacity (enqueued-token limit)', 'at': now()})
            report[key] = {'ok': 0, 'anomaly': 0, 'missing': 0, 'duplicate_lines': 0, 'unexpected_lines': 0, 'requeued': 1}
            continue
        todo.append((key, snap))
    downloads = {r['file_id']: r for r in ledger.rows('batch_downloads')}

    def fetch(item):
        key, snap = item
        started, rows = time.time(), []
        for kind in ('output_file_id', 'error_file_id'):
            file_id = snap.get(kind)
            if not file_id or file_id in downloads:
                continue
            data = client.files.content(file_id).content
            path = ledger.dir/'batch'/f'{key}-{snap["batch_id"]}-{kind.split("_")[0]}.jsonl.gz'
            _write_atomic(path, gzip.compress(data, compresslevel=3))
            rows.append({'part': key, 'kind': kind, 'file_id': file_id, 'path': str(path.relative_to(ledger.dir)), 'gzip': True,
                         'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'at': now()})
        return key, rows, time.time() - started

    if workers > 1 and len(todo) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            fetched = list(pool.map(fetch, todo))
    else:
        fetched = [fetch(item) for item in todo]
    answered = set(ledger.index()['answered'])
    for (key, snap), (_, rows, seconds) in zip(todo, fetched):
        for r in rows:
            ledger.append('batch_downloads', r)
            downloads[r['file_id']] = r
        files = [ledger.dir/downloads[snap[k]]['path'] for k in ('output_file_id', 'error_file_id') if snap.get(k)]
        started = time.time()
        report[key] = evaluate_part(ledger, by_part[key], files, snap['status'], answered)
        ledger.append('batch_collected', {'part': key, 'batch_id': snap['batch_id'], 'status': snap['status'], **report[key],
                                          't': round(time.time(), 3), 'download_s': round(seconds, 3),
                                          'evaluate_s': round(time.time() - started, 3), 'at': now()})
    ledger.summary()
    return report


def evaluate_part(ledger, part, files, status, answered):
    expected = set(part['custom_ids'])
    counts = {'ok': 0, 'anomaly': 0, 'missing': 0, 'duplicate_lines': 0, 'unexpected_lines': 0}
    seen, replies, failed, errors = set(), [], [], []
    for path in files:
        opener = gzip.open if str(path).endswith('.gz') else open
        rel = str(path.relative_to(ledger.dir)) if path.is_relative_to(ledger.dir) else str(path)
        with opener(path, 'rb') as stream:  # streamed line by line; each reply keeps a pointer to its exact provider line
            for number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                line = raw.decode()
                item = json.loads(line)
                cid = item.get('custom_id')
                if cid not in expected:
                    counts['unexpected_lines'] += 1
                    errors.append({'part': part['part'], 'step': 'collect', 'unexpected_custom_id': mask(cid)[:120], 'at': now()})
                    continue
                if cid in seen or cid in answered:
                    counts['duplicate_lines'] += cid in seen
                    seen.add(cid)
                    continue
                seen.add(cid)
                response, error = item.get('response') or {}, item.get('error')
                if response.get('status_code') == 200 and isinstance(response.get('body'), dict):
                    replies.append((cid, response['body'], {'file': rel, 'line': number, 'line_sha256': hashlib.sha256(raw.rstrip(b'\n')).hexdigest()}))
                else:
                    code = response.get('status_code')
                    detail = f'status {code}; ' + mask(json.dumps(error or (response.get('body') or {}).get('error')))[:300]
                    failed.append((cid, ExtractionAnomaly('batch_error', detail, retriable=code is None or http_retriable(code))))
    ledger.append_many('batch_errors', errors)
    rows = ledger.record_batch_responses(replies)
    for cid in sorted(expected - seen - answered):
        failed.append((cid, ExtractionAnomaly('missing_result', f'batch {status} returned no line for this request')))
    anomalies = [{'custom_id': cid, 'record_id': split_custom_id(cid)[0], 'attempt': split_custom_id(cid)[1], 'mode': 'batch', 'at': now(),
                  'outcome': 'anomaly', 'anomaly': a.as_dict()} for cid, a in failed]
    ledger.append_many('results', anomalies)
    for row in rows + anomalies:
        answered.add(row['custom_id'])
        counts[row['outcome']] += 1
    counts['missing'] = sum(1 for r in anomalies if r['anomaly']['kind'] == 'missing_result')
    counts['anomaly'] -= counts['missing']
    return counts


def safe(error):
    return {'type': type(error).__name__, 'status': getattr(error, 'status_code', None),
            'message': mask(getattr(error, 'message', None) or error)[:300]}


ROLL_LOCK, STOP_FILE, HEARTBEAT = 'roll.lock', 'STOP', 'heartbeat.json'


class Busy(RuntimeError):
    """Another live process drives this run."""


def lock_owner(ledger):
    """PID recorded by a live process that holds the run's driver lock (flock), -1 if held with an unreadable PID, else
    None. The kernel drops the lock when its process ends, so a crashed driver never leaves a stale lock."""
    path = ledger.dir/ROLL_LOCK
    if not path.exists():
        return None
    with open(path, 'a+') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            try:
                return int(json.loads(stream.read())['pid'])
            except (ValueError, KeyError, TypeError):
                return -1
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return None


def acquire_lock(ledger):
    """Hold an exclusive flock on the run's lock file for the life of the driver; returns the open file."""
    stream = open(ledger.dir/ROLL_LOCK, 'a+')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.seek(0)
        holder = stream.read()[:200]
        stream.close()
        raise Busy(f'run {ledger.run_id} is driven by another live process ({holder or "pid not written yet"}); nothing started') from None
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps({'pid': os.getpid(), 'at': now()}))
    stream.flush()
    return stream


def release_lock(stream):
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def is_first(part):
    return all(split_custom_id(cid)[1] == 1 for cid in part['custom_ids'])


def reconcile(ledger, client):
    """At a driver start: walk the whole account once. Duplicate live batches for one part stop everything; a batch on
    the account that the ledger does not hold for a part without a current batch is recorded (never re-created)."""
    remote = remote_batches(client, ledger, pages=None)
    gone, recorded, current = requeued_ids(ledger), {r['batch_id'] for r in ledger.rows('batches')}, current_batches(ledger)
    live = {k: [b for b in v if b['batch_id'] not in gone] for k, v in remote.items()}
    repeated = {k: v for k, v in live.items() if len(v) > 1}
    if repeated:
        ledger.append('batch_errors', {'step': 'duplicate check (start)', 'duplicates': repeated, 'at': now()})
        raise RuntimeError(f'the account holds several batches for {sorted(repeated)}; nothing submitted, decide manually')
    by_part = {p['part']: p for p in parts(ledger)}
    added = []
    for key, found in live.items():
        new = [b for b in found if b['batch_id'] not in recorded]
        if new and key not in current and key in by_part:
            p = by_part[key]
            ledger.append('batches', {'part': key, 'sha256': p['sha256'], 'requests': p['requests'], 'cap_usd': p['cap_usd'], **new[0],
                                      'recovered_from': 'account metadata at driver start', 'at': now()})
            added.append(key)
    return remote, added


def roll(ledger, client, *, budget_usd, read_client=None, window=16, part_size=256, max_attempts=2, poll_seconds=15.0,
         deadline_seconds=None, eval_room=None, clock=time.time, sleep=time.sleep, echo=None, max_bytes=MAX_BYTES,
         poll_workers=1, download_workers=1, backoff_seconds=60.0, backoff_max_seconds=900.0, status_seconds=60.0,
         reconcile_at_start=False):
    """Rolling window over prepare/submit/poll/collect. Each tick polls the in-flight batches of this run, submits
    prepared parts into free slots (at most `window` in flight), then downloads and evaluates the finished ones.
    First attempts are prepared up front in parts of `part_size`; retriable anomalies become a retry part once a part's
    worth has accumulated or no first-attempt part is still open. Guards: those of submit (budget = actual spend,
    uncertain charges and open caps; an unknown create outcome halts submission and is never re-created, while in-flight
    batches are still collected), one driver per run (lock file), `eval_room` (request rows this call may add), and a
    deadline or STOP file, which end the call and leave running batches on the server; calling roll again resumes from
    the ledger. Failed batches and missing lines stay anomalies, except a capacity failure (enqueued-token limit) or a
    capacity refusal of create, which back off (backoff_seconds doubling to backoff_max_seconds) and resend the same part.
    A heartbeat file is rewritten every tick and status.json every `status_seconds`. Returns roll_report() plus the
    reason the call ended."""
    if window < 1 or not 0 < part_size <= MAX_REQUESTS:
        raise ValueError('window must be >= 1 and part_size 1..50,000')
    read_client = read_client or client
    lock = acquire_lock(ledger)
    t0 = clock()
    halted, room, stopped = None, eval_room, None
    backoff, paused_until, last_status = backoff_seconds, 0.0, None

    def event(kind, **extra):
        row = {'event': kind, 'epoch': round(clock(), 3), 't': round(clock() - t0, 3), 'at': now(), **extra}
        ledger.append('roll_events', row)
        if echo:
            echo(row)

    def ids(name):
        return {r['part'] for r in ledger.rows(name)}

    def inflight():
        return set(current_batches(ledger)) - ids('batch_collected')

    def halt_reason():
        if (ledger.dir/STOP_FILE).exists():
            return 'stop_file'
        if deadline_seconds is not None and clock() - t0 >= deadline_seconds:
            return 'deadline'
        return None

    def queued():
        current, collected = current_batches(ledger), ids('batch_collected')
        waiting = [p['part'] for p in parts(ledger) if p['part'] not in current and p['part'] not in collected]
        again = {r['part'] for r in ledger.rows('batch_requeues')}
        return sorted(waiting, key=lambda k: (k not in again, k))  # requeued parts first

    def heartbeat(active, note=None):
        data = {'pid': os.getpid(), 'at': now(), 'epoch': round(clock(), 3), 'started_epoch': round(t0, 3), 'window': window,
                'part_size': part_size, 'active': active, 'queued_parts': len(queued()), 'halted': halted, 'paused_until': paused_until or None,
                'budget_usd': budget_usd, 'note': note}
        tmp = ledger.dir/(HEARTBEAT + '.tmp')
        tmp.write_text(json.dumps(data) + '\n')
        os.replace(tmp, ledger.dir/HEARTBEAT)

    def fill(active):
        nonlocal halted, paused_until, backoff
        queue, space = queued(), window - active
        if halted or not queue or space <= 0 or halt_reason() or clock() < paused_until:
            return active
        before, started = {r['batch_id'] for r in ledger.rows('batches')}, clock()  # by batch id: a requeued part gets a new batch
        try:
            submit(ledger, client, budget_usd=budget_usd, only=set(queue[:space]), should_stop=lambda: halt_reason() is not None)
        except Backpressure as error:  # nothing created: wait for capacity, then send the same part again
            paused_until = clock() + backoff
            event('backpressure', reason=mask(str(error))[:300], pause_s=backoff, active=active)
            backoff = min(backoff * 2, backoff_max_seconds)
        except RuntimeError as error:  # budget stop, unresolved create, upload failure: no further submission
            halted = mask(str(error))[:300]
        new = [r['part'] for r in ledger.rows('batches') if r['batch_id'] not in before]
        active += len(new)
        if new:
            event('submit', parts=new, seconds=round(clock() - started, 3), active=active)
        if halted:
            event('halt', reason=halted, active=active)
        return active

    def backlog():
        busy = {split_custom_id(cid)[0] for cid in open_custom_ids(ledger)}
        policy = ledger.manifest.get('retry_policy')
        return [rid for rid, state in ledger.index()['states'].items()
                if rid not in busy and state['attempts'] > 0 and needs_attempt(state, max_attempts, policy)]

    def add_retries(active):
        nonlocal room
        if halted or active >= window or queued():
            return
        waiting = backlog()
        collected = ids('batch_collected')
        if not waiting or (len(waiting) < part_size and any(is_first(p) and p['part'] not in collected for p in parts(ledger))):
            return  # accumulate until a part fills or every first-attempt part is collected
        if room is not None:
            if room <= 0:
                return
            waiting = waiting[:room]
        started = clock()
        made = prepare(ledger, max_attempts=max_attempts, max_requests=part_size, max_bytes=max_bytes, record_ids=waiting)
        added = sum(p['requests'] for p in made)
        room = None if room is None else room - added
        event('prepare', attempt='retry', parts=len(made), requests=added, seconds=round(clock() - started, 3))

    try:
        event('start', pid=os.getpid(), window=window, part_size=part_size, max_attempts=max_attempts, poll_seconds=poll_seconds,
              deadline_seconds=deadline_seconds, budget_usd=budget_usd, eval_room=eval_room,
              spec=ledger.manifest.get('spec'), reasoning_effort=ledger.manifest.get('reasoning_effort'))
        ledger.recover()
        if reconcile_at_start:
            _, added = reconcile(ledger, read_client)
            if added:
                event('reconciled', parts=added)
        ix = ledger.index()
        in_parts = {cid for p in parts(ledger) for cid in p['custom_ids']}
        orphans = sorted(cid for cid, mode in ix['requested'].items() if mode == 'batch' and cid not in in_parts and cid not in ix['answered'])
        fresh = [rid for rid, state in ix['states'].items() if state['attempts'] == 0]
        if orphans:  # an interrupted prepare logged requests without their part: neither fresh nor retriable; stop, do not guess
            stopped = 'orphan_requests'
            event('orphan_requests', count=len(orphans), sample=[mask(c)[:80] for c in orphans[:5]],
                  detail='batch requests logged without a prepared part (interrupted prepare); nothing submitted; needs a manual recovery decision')
        elif fresh and room is not None and len(fresh) > room:
            stopped = 'eval_cap'
            event('eval_cap', detail=f'{len(fresh)} first attempts exceed the remaining {room} record-evaluations; nothing prepared')
        elif fresh:
            started = clock()
            made = prepare(ledger, max_attempts=max_attempts, max_requests=part_size, max_bytes=max_bytes, record_ids=fresh)
            added = sum(p['requests'] for p in made)
            room = None if room is None else room - added
            event('prepare', attempt='first', parts=len(made), requests=added, seconds=round(clock() - started, 3))
        first_done = any(r['event'] == 'first_pass_done' for r in ledger.rows('roll_events'))
        while stopped is None:
            tick = clock()
            stopped = halt_reason()
            if stopped:
                break
            live = inflight()
            snaps = poll(ledger, read_client, only=live, workers=poll_workers) if live else []
            stopped = halt_reason()  # a slow poll may cross the deadline or meet a STOP file: no new paid submission
            if stopped:
                break
            final = [s for s in snaps if s.get('status') in FINAL]
            if final:  # collected first, so a finished batch's cap is replaced by its actual usage before any new submission
                started = clock()
                try:
                    report = collect(ledger, read_client, snapshots=final, recover=False, workers=download_workers)
                    requeued = sorted(k for k, r in report.items() if r.get('requeued'))
                    if requeued:
                        paused_until = clock() + backoff
                        event('requeue', parts=requeued, pause_s=backoff, reason='capacity failure (enqueued-token limit); nothing processed')
                        backoff = min(backoff * 2, backoff_max_seconds)
                    elif report:
                        backoff = backoff_seconds  # capacity is back: reset the back-off
                    event('collect', parts=sorted(k for k in report if k not in requeued), seconds=round(clock() - started, 3),
                          outcome={k: sum(r[k] for r in report.values()) for k in ('ok', 'anomaly', 'missing')})
                except Exception as error:  # a failed download is retried next tick; the part stays uncollected
                    ledger.append('batch_errors', {'step': 'roll collect', 'error': safe(error), 'at': now()})
                    event('collect_error', error=safe(error))
            active = fill(len(inflight()))
            first = [p['part'] for p in parts(ledger) if is_first(p)]
            if not first_done and first and set(first) <= ids('batch_collected'):
                tally = ledger.index()['first']
                event('first_pass_done', requests=sum(p['requests'] for p in parts(ledger) if is_first(p)),
                      ok=tally['ok'], anomalies=tally['anomaly'])
                first_done = True
            add_retries(active)
            active = fill(active)
            if last_status is None or clock() - last_status >= status_seconds:
                ledger.summary()
                last_status = clock()
            heartbeat(active)
            stopped = halt_reason()
            if stopped:
                break
            if not inflight():
                if halted:  # nothing in flight and no further submission: queued parts stay prepared, unsubmitted
                    stopped = 'halted'
                elif not queued() and (not backlog() or (room is not None and room <= 0)):
                    stopped = 'done'
                if stopped:
                    break
            sleep(max(0.0, poll_seconds - (clock() - tick)))
        event('end', stopped=stopped, halted=halted, wall_s=round(clock() - t0, 3))
        ledger.summary()
        heartbeat(len(inflight()), note=f'ended: {stopped}')
        return {**roll_report(ledger), 'stopped': stopped, 'halted': halted}
    finally:
        release_lock(lock)


def roll_report(ledger):
    """Timeline and yield of a rolled run from its ledger; times are seconds from the first driver start."""
    events = ledger.rows('roll_events')
    starts = [e for e in events if e['event'] == 'start']
    if not starts:
        return {}
    t0 = starts[0]['epoch']

    def rel(x):
        return None if x is None else round(x - t0, 1)

    def span(a, z):
        return None if a is None or z is None else z - a

    last = {}
    for snap in ledger.rows('batch_snapshots'):
        if snap.get('status'):
            last[snap['part']] = snap
    batches = current_batches(ledger)
    collected = {r['part']: r for r in ledger.rows('batch_collected')}
    timeline = []
    for p in parts(ledger):
        key, b, s, c = p['part'], batches.get(p['part'], {}), last.get(p['part'], {}), collected.get(p['part'], {})
        created, started, finalizing = s.get('created_at'), s.get('in_progress_at'), s.get('finalizing_at')
        ended = next((s[k] for k in ('completed_at', 'failed_at', 'expired_at', 'cancelled_at') if s.get(k)), None)
        timeline.append({'part': key, 'requests': p['requests'], 'first_attempts': is_first(p), 'status': s.get('status'),
                         'submitted': rel(b.get('t')), 'upload_s': b.get('upload_s'), 'create_s': b.get('create_s'),
                         'created': rel(created), 'ended': rel(ended), 'collected': rel(c.get('t')),
                         'validating_s': span(created, started), 'in_progress_s': span(started, finalizing),
                         'finalizing_s': span(finalizing, ended), 'provider_s': span(created, ended),
                         'download_s': c.get('download_s'), 'evaluate_s': c.get('evaluate_s')})
    ix = ledger.index()

    def tally(counter):
        kinds = {k.split(':', 1)[1]: v for k, v in counter.items() if k.startswith('anomaly:')}
        return {'results': counter['ok'] + counter['anomaly'], 'ok': counter['ok'], 'anomalies': kinds}

    first_done = next((rel(e['epoch']) for e in events if e['event'] == 'first_pass_done'), None)
    end = next((rel(e['epoch']) for e in reversed(events) if e['event'] == 'end' and e.get('stopped') == 'done'), None)
    first, retry = tally(ix['first']), tally(ix['retry'])
    valid = sum(1 for s in ix['states'].values() if s['ok'])
    return {'invocations': len(starts), 'window': starts[-1].get('window'), 'part_size': starts[-1].get('part_size'),
            'max_active': max([e.get('active', 0) for e in events if e['event'] == 'submit'] or [0]),
            'records': len(ledger.inputs()), 'first_pass': {**first, 'done_s': first_done,
                                                           'valid_per_s': round(first['ok'] / first_done, 2) if first_done else None},
            'retries': retry, 'valid_records': valid, 'done_s': end,
            'valid_per_s': round(valid / end, 2) if end else None,
            'batch_cost_usd': round(ix['tokens'].get('batch', {}).get('cost_usd', 0.0), 6),
            'requeues': len(ledger.rows('batch_requeues')), 'parts': timeline}
