"""Full-corpus Jev screening worker (user-authorized 2026-09-29; Jev only, no Luna).

Frozen configuration: prompt strict400-v2a (file sha256 4f77a72a...), one binary Choice per record, state = full record text,
keep iff probabilities.keep >= 0.5, model jev-1.13.0, target 1150 requests/min, concurrency 8, no packing.
Input: the 85 s10 shards listed in input-manifest.json (1,030,309 unique ids). Each shard is extracted to a temporary JSONL
by a separate .venv-reddit-clean process, verified against the manifest (count + id hash), processed, then deleted.

Persistence and resume: one result line per record (results/<shard>.jsonl) and one attempt line per HTTP call
(attempts/<shard>.jsonl, with usage and cost); records already in results are never re-sent. A record whose attempts all
fail technically goes to errors/<shard>.jsonl, is NOT a drop, and is retried on the next start.
Guards: cumulative cost <= MAX_USD including the unknown reserve and in-flight caps; 401/403 stops at once; an error storm
or a STOP file stops gracefully; 429 honours Retry-After, pauses all workers and lowers the rate, which recovers slowly.
The API key is read from TYPESAFE_API_KEY (loaded by start.sh) and never written or printed.
"""
import collections
import datetime
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get('JEV_FULL_RES') or ROOT / 'results/jev-full-2026-09-29')
CACHE = Path(os.environ.get('JEV_FULL_CACHE') or ROOT / '.cache/jev-full-2026-09-29')
EXPORT = ROOT / '.cache/reddit-cleaning/reddit-strict-1m-s10-2026-09-26/records'
PROMPT = ROOT / 'results/jev-appetite-correction-2026-09-29/prompts/prompt-v2a.json'
PROMPT_SHA256 = '4f77a72aa5e91e2dd787f744b1ebd7046b8dbb87129f7cbe5c678bd70719c1a4'
EXTRACT_PY = ROOT / '.venv-reddit-clean/bin/python'
ENDPOINT, MODEL, QID = 'https://api.typesafe.ai/v1/systemone', 'jev-1.13.0', 'strict_keep'
USD_PER_M_INPUT = 0.042
THRESHOLD = 0.5
MAX_USD = float(os.environ.get('JEV_FULL_MAX_USD') or 75.0)
TARGET_RPM, MIN_RPM, CONCURRENCY, MAX_ATTEMPTS = 1150.0, 500.0, 8, 3
RETRIABLE = {408, 409, 429, 500, 502, 503, 504, 529}
STORM_CONSECUTIVE, STORM_WINDOW, STORM_RATE = 50, 2000, 0.05
LOCK = threading.RLock()  # re-entrant: Stats.record may call stop() while holding it
STOP = threading.Event()
PAUSE_REASON = []


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def mask(text, key):
    text = str(text).replace(key, '***') if key else str(text)
    return re.sub(r'(Bearer\s+)\S+', r'\1***', text)[:400]


def stop(reason):
    with LOCK:
        if not PAUSE_REASON:
            PAUSE_REASON.append(reason)
    STOP.set()


class Rate:
    """Global start spacing; 429 -> cooldown for all workers and a lower rate that recovers by 5% per quiet 10 minutes."""

    def __init__(self):
        self.rpm, self.next_at, self.cool_until, self.last_429, self.n429 = TARGET_RPM, 0.0, 0.0, 0.0, 0
        self.starts = collections.deque()

    def wait(self):
        with LOCK:
            t = time.monotonic()
            if self.rpm < TARGET_RPM and t - self.last_429 > 600:
                self.rpm, self.last_429 = min(TARGET_RPM, self.rpm * 1.05), t
            s = max(t, self.next_at, self.cool_until)
            self.next_at = s + 60.0 / self.rpm
            self.starts.append(s)
            while self.starts and self.starts[0] < s - 300:
                self.starts.popleft()
        d = s - time.monotonic()
        if d > 0:
            time.sleep(d)

    def hit_429(self, retry_after):
        with LOCK:
            t = time.monotonic()
            self.n429 += 1
            self.last_429 = t
            self.cool_until = max(self.cool_until, t + (retry_after or 5.0))
            self.rpm = max(MIN_RPM, self.rpm * 0.85)

    def recent_rpm(self):
        with LOCK:
            if len(self.starts) < 2:
                return None
            span = self.starts[-1] - self.starts[0]
            return round((len(self.starts) - 1) / span * 60, 1) if span > 0 else None


class Ledger:
    def __init__(self):
        self.usd = self.unknown = self.committed = 0.0
        self.attempts = self.retries = 0
        for p in (CACHE / 'attempts').glob('*.jsonl'):
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.usd += r.get('usd') or 0.0
                    self.unknown += r.get('unknown_usd') or 0.0
                    self.attempts += 1
                    self.retries += r.get('attempt', 1) > 1

    def reserve(self, cap):
        with LOCK:
            if self.usd + self.unknown + self.committed + cap > MAX_USD:
                return False
            self.committed += cap
            return True

    def settle(self, cap, row, path):
        with LOCK:
            self.committed -= cap
            self.usd += row['usd']
            self.unknown += row['unknown_usd']
            self.attempts += 1
            self.retries += row['attempt'] > 1
            with path.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_prompt():
    raw = PROMPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROMPT_SHA256, 'prompt file changed'
    return json.loads(raw)


def body_for(text, prompt):
    return {'model': MODEL, 'state': text,
            'questions': {QID: {'type': 'choice', 'instructions': prompt['instructions'], 'criteria': prompt['criteria']}}}


def extract(shard, dst):
    out = subprocess.run([str(EXTRACT_PY), str(RES / 'scripts/extract_shard.py'), str(EXPORT / shard), str(dst)], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f'extract failed for {shard}: {out.stderr[-300:]}')


def rebuild_from_attempts(shard):
    """A successful attempt logged without its result line (crash in between) becomes a result, so it is never paid again."""
    ap, rp = CACHE / 'attempts' / (shard + '.jsonl'), CACHE / 'results' / (shard + '.jsonl')
    if not ap.exists():
        return 0
    have = done_ids(shard)
    add = {}
    for x in ap.read_text().splitlines():
        if not x.strip():
            continue
        a = json.loads(x)
        pk = a.get('p_keep')
        if a.get('error') is None and a.get('choice') in ('keep', 'drop') and isinstance(pk, (int, float)) and a['record_id'] not in have:
            add[a['record_id']] = {'record_id': a['record_id'], 'shard': shard, 'decision': 'keep' if pk >= THRESHOLD else 'drop', 'p_keep': pk,
                                   'choice': a['choice'], 'confidence': a.get('confidence'), 'input_tokens': a.get('input_tokens'), 'attempts': a.get('attempt'),
                                   'request_id': a.get('request_id'), 'at': a.get('at'), 'rebuilt_from_attempt': True}
    if add:
        with rp.open('a') as f:
            for r in add.values():
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(add)


def done_ids(shard):
    p = CACHE / 'results' / (shard + '.jsonl')
    if not p.exists():
        return set()
    return {json.loads(x)['record_id'] for x in p.read_text().splitlines() if x.strip()}


class Stats:
    def __init__(self, manifest):
        self.total = manifest['records']
        self.done = self.keep = self.pending_errors = 0
        self.session_done = 0
        self.t0 = time.monotonic()
        self.recent = collections.deque(maxlen=STORM_WINDOW)
        self.consecutive_fail = 0
        for sh in manifest['shards']:
            p = CACHE / 'results' / (sh['shard'] + '.jsonl')
            if p.exists():
                for x in p.read_text().splitlines():
                    if x.strip():
                        self.done += 1
                        self.keep += json.loads(x)['decision'] == 'keep'

    def record(self, ok, keep=False):
        with LOCK:
            self.recent.append(ok)
            if ok:
                self.done += 1
                self.session_done += 1
                self.keep += keep
                self.consecutive_fail = 0
            else:
                self.pending_errors += 1
                self.consecutive_fail += 1
            if self.consecutive_fail >= STORM_CONSECUTIVE:
                stop(f'error storm: {self.consecutive_fail} consecutive failed records')
            if len(self.recent) >= 500 and self.recent.count(False) / len(self.recent) > STORM_RATE:
                stop(f'error storm: {self.recent.count(False)}/{len(self.recent)} recent records failed')


def write_status(phase, stats, ledger, rate, manifest, shards_done, current):
    el = time.monotonic() - stats.t0
    rpm_rec = stats.session_done / el * 60 if el > 0 else None
    left = stats.total - stats.done
    st = {'phase': phase, 'pid': os.getpid(), 'updated_utc': now(), 'pause_reason': PAUSE_REASON[0] if PAUSE_REASON else None,
          'records_total': stats.total, 'records_done': stats.done, 'records_left': left, 'keep': stats.keep, 'drop': stats.done - stats.keep,
          'keep_rate': round(stats.keep / stats.done, 4) if stats.done else None, 'error_records_this_session': stats.pending_errors,
          'attempts': ledger.attempts, 'retries': ledger.retries, 'http_429': rate.n429, 'usd_known': round(ledger.usd, 6),
          'usd_unknown_reserve': round(ledger.unknown, 6), 'usd_cap': MAX_USD, 'target_rpm_now': round(rate.rpm, 1),
          'request_rpm_last_5min': rate.recent_rpm(), 'records_per_min_session': round(rpm_rec, 1) if rpm_rec else None,
          'eta_hours': round(left / rpm_rec / 60, 2) if rpm_rec else None, 'shards_done': shards_done, 'shards_total': len(manifest['shards']),
          'current_shard': current, 'config': {'model': MODEL, 'prompt_sha256': PROMPT_SHA256, 'threshold': THRESHOLD, 'target_rpm': TARGET_RPM, 'concurrency': CONCURRENCY}}
    tmp = RES / 'run-status.json.tmp'
    tmp.write_text(json.dumps(st, indent=1) + '\n')
    os.replace(tmp, RES / 'run-status.json')


def run():
    key = os.environ.get('TYPESAFE_API_KEY', '')
    if not key:
        sys.exit('TYPESAFE_API_KEY not loaded (use start.sh)')
    for sub in ('results', 'attempts', 'errors', 'tmp'):
        (CACHE / sub).mkdir(parents=True, exist_ok=True)
    lockp = CACHE / 'worker.lock'
    try:
        fd = os.open(lockp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f'{lockp} exists: another worker may be running (remove only after confirming it is not)')
    os.write(fd, f'{os.getpid()} {now()}\n'.encode())
    os.close(fd)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *a: stop('SIGTERM received'))
    try:
        manifest = json.loads((RES / 'input-manifest.json').read_text())
        assert manifest['records'] == manifest['unique_ids'] == 1_030_309
        prompt = load_prompt()
        ledger, rate, stats = Ledger(), Rate(), Stats(manifest)
        local = threading.local()
        with (RES / 'run-log.jsonl').open('a') as f:
            f.write(json.dumps({'event': 'start', 'at': now(), 'pid': os.getpid(), 'records_done_before': stats.done, 'usd_before': round(ledger.usd, 6), 'max_usd': MAX_USD}) + '\n')

        def client():
            if not hasattr(local, 'c'):
                local.c = httpx.Client(timeout=httpx.Timeout(60.0, connect=15.0))
            return local.c

        def one(row, shard):
            body = body_for(row['text'], prompt)
            cap = len(json.dumps(body, ensure_ascii=False).encode()) * USD_PER_M_INPUT / 1e6
            last = None
            for n in range(1, MAX_ATTEMPTS + 1):
                if STOP.is_set():
                    return
                if not ledger.reserve(cap):
                    stop(f'budget cap ${MAX_USD} reached (known + unknown reserve + in flight)')
                    return
                rate.wait()
                t0 = time.monotonic()
                sc = rid = err = None
                out = None
                retry_after = None
                try:
                    r = client().post(ENDPOINT, json=body, headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
                    sc, rid = r.status_code, r.headers.get('x-typesafe-request-id')
                    if sc == 200:
                        j = r.json()
                        a = (j.get('answers') or {}).get(QID) or {}
                        pr, u = a.get('probabilities'), j.get('usage') or {}
                        pk = pr.get('keep') if isinstance(pr, dict) else None
                        if a.get('choice') in ('keep', 'drop') and isinstance(pk, (int, float)) and not isinstance(pk, bool) and math.isfinite(pk) and 0.0 <= pk <= 1.0 and isinstance(u.get('input_tokens'), int):
                            out = {'choice': a['choice'], 'p_keep': pr['keep'], 'confidence': a.get('confidence'), 'input_tokens': u['input_tokens'], 'model': j.get('model')}
                        else:
                            err = 'bad_response'
                    else:
                        err = f'http_{sc}: {mask(r.text, key)}'
                        ra = r.headers.get('retry-after')
                        try:
                            retry_after = min(float(ra), 30.0) if ra else None
                        except ValueError:
                            retry_after = None
                except (httpx.HTTPError, ValueError) as e:
                    err = f'{type(e).__name__}: {mask(e, key)}'
                known = out is not None
                att = {'record_id': row['record_id'], 'attempt': n, 'at': now(), 'status_code': sc, 'request_id': rid, 'latency_s': round(time.monotonic() - t0, 3),
                       'usd': round(out['input_tokens'] * USD_PER_M_INPUT / 1e6, 12) if known else 0.0, 'unknown_usd': 0.0 if known else round(cap, 12),
                       'input_tokens': out['input_tokens'] if known else None, 'choice': out['choice'] if known else None,
                       'p_keep': out['p_keep'] if known else None, 'confidence': out['confidence'] if known else None,
                       'model': out['model'] if known else None, 'error': err}
                ledger.settle(cap, att, CACHE / 'attempts' / (shard + '.jsonl'))
                if known:
                    res = {'record_id': row['record_id'], 'shard': shard, 'decision': 'keep' if out['p_keep'] >= THRESHOLD else 'drop', 'p_keep': out['p_keep'],
                           'choice': out['choice'], 'confidence': out['confidence'], 'input_tokens': out['input_tokens'], 'attempts': n, 'request_id': rid, 'at': att['at']}
                    with LOCK:
                        with (CACHE / 'results' / (shard + '.jsonl')).open('a') as f:
                            f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    stats.record(True, res['decision'] == 'keep')
                    return
                last = err
                if sc in (401, 403):
                    stop(f'authentication/permission error http_{sc}')
                    break
                if sc == 429:
                    rate.hit_429(retry_after)
                elif sc is not None and sc not in RETRIABLE:
                    break
                time.sleep(retry_after or 2.0 * n)
            with LOCK:
                with (CACHE / 'errors' / (shard + '.jsonl')).open('a') as f:
                    f.write(json.dumps({'record_id': row['record_id'], 'at': now(), 'last_error': last}, ensure_ascii=False) + '\n')
            stats.record(False)

        shards_done = sum(1 for sh in manifest['shards'] if len(done_ids(sh['shard'])) == sh['records'])
        last_status = 0.0
        for sh in manifest['shards']:
            if STOP.is_set() or (CACHE / 'STOP').exists():
                stop(PAUSE_REASON[0] if PAUSE_REASON else 'STOP file present')
                break
            name = sh['shard']
            rebuilt = rebuild_from_attempts(name)
            if rebuilt:
                with (RES / 'run-log.jsonl').open('a') as f:
                    f.write(json.dumps({'event': 'rebuilt_results_from_attempts', 'shard': name, 'records': rebuilt, 'at': now()}) + '\n')
                stats.done += rebuilt
            have = done_ids(name)
            if len(have) == sh['records']:
                continue
            tmp = CACHE / 'tmp' / (name + '.jsonl')
            extract(name, tmp)
            rows = [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()]
            ids = sorted(r['record_id'] for r in rows)
            if len(rows) != sh['records'] or hashlib.sha256('\n'.join(ids).encode()).hexdigest() != sh['ids_sha256']:
                stop(f'input verification failed for {name}')
                break
            pending = [r for r in rows if r['record_id'] not in have]
            del rows, ids
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futs = []
                for r in pending:
                    if STOP.is_set() or (CACHE / 'STOP').exists():
                        stop(PAUSE_REASON[0] if PAUSE_REASON else 'STOP file present')
                        break
                    futs.append(pool.submit(one, r, name))
                    if len(futs) >= 64:  # bounded queue: never materialise more than a small window of futures
                        futs.pop(0).result()
                    if time.monotonic() - last_status > 30:
                        write_status('running', stats, ledger, rate, manifest, shards_done, name)
                        last_status = time.monotonic()
                for f in futs:
                    f.result()
            if len(done_ids(name)) == sh['records']:
                shards_done += 1
                tmp.unlink(missing_ok=True)
            write_status('running', stats, ledger, rate, manifest, shards_done, name)
        complete = all(len(done_ids(sh['shard'])) == sh['records'] for sh in manifest['shards'])
        phase = 'paused' if PAUSE_REASON else ('completed' if complete else 'incomplete (errors pending; restart retries them)')
        write_status(phase, stats, ledger, rate, manifest, shards_done, None)
        with (RES / 'run-log.jsonl').open('a') as f:
            f.write(json.dumps({'event': 'end', 'at': now(), 'phase': phase, 'reason': PAUSE_REASON[0] if PAUSE_REASON else None,
                                'records_done': stats.done, 'usd_known': round(ledger.usd, 6)}) + '\n')
    finally:
        lockp.unlink(missing_ok=True)


if __name__ == '__main__':
    run()
