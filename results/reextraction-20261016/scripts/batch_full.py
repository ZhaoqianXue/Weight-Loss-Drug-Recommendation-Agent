"""Full Batch run of the frozen v9 extraction (v9-prompt-7, freeze-v7.json) over the frozen full inputs.

Usage:
  python batch_full.py --run-id full-v7 prepare      # write the request parts (no network)
  python batch_full.py --run-id full-v7 drive [--budget-usd 60] [--poll-seconds 300] [--max-parts N]

`drive` loops until every record has a final result. It submits the first part (1,000 requests) alone and waits for it;
the rest is submitted only if that part's real Batch cost projects the whole run within the budget and its share of
records needing a retry stays under --gate-retry. It then submits the remaining parts, polls them, downloads and
evaluates finished ones, and after the last first-attempt part turns retriable anomalies and failed wording checks into
retry parts (attempt 2). On the retry, items that still fail the wording check are dropped; this is the same procedure as
the frozen synchronous runner (run.py).

Submitted batches run on the provider's servers, so a lost connection only pauses submitting, polling and downloading;
every step resumes from the files in the run folder, and a restarted driver continues where the last one stopped. Before
creating a batch the account is checked for one with the same run, part and payload hash, so a create is never repeated.
A capacity refusal (enqueued-token limit) backs off and resends later. A STOP file in the run folder ends the loop and
leaves running batches on the server.
"""
import argparse, fcntl, gzip, hashlib, io, json, os, sys, time, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from weightloss.extraction import reextract as rx
from weightloss.extraction.structured import (ExtractionAnomaly, build_body, cost_usd, final_output_text, parse_json_object,
                                              schema_sha256, sha256_text, strict_json_schema, token_split, validate_model)

STAGE = os.path.join(ROOT, 'results/reextraction-20261016')
FREEZE = json.load(open(os.path.join(STAGE, 'freeze-v7.json')))
INPUTS = os.path.join(ROOT, '.cache/reextraction-20261016/full/inputs.jsonl')
MODEL, EFFORT, MAX_OUT = FREEZE['model'], FREEZE['reasoning_effort'], FREEZE['max_output_tokens']
PART_SIZE, FIRST_PART = 5000, 1000
FINAL_STATES = ('completed', 'failed', 'expired', 'cancelled')
CAPACITY = ('token_limit_exceeded', 'enqueued token limit', 'enqueued_token', 'rate limit', 'rate_limit')
NO_RETRY = ('refusal', 'content_filter')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def utc(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


class Run:
    def __init__(self, run_id):
        self.id = run_id
        self.dir = os.path.join(ROOT, '.cache/reextraction-20261016/full', run_id)
        for sub in ('parts', 'downloads', 'evaluated'):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.prompts = {p: open(os.path.join(STAGE, 'prompts', f"{FREEZE['prompt_version']}-{p}.txt")).read() for p in ('reddit', 'webmd')}
        self.schemas = {p: strict_json_schema(rx.output_model(p)) for p in ('reddit', 'webmd')}
        for p in ('reddit', 'webmd'):
            if sha256_text(self.prompts[p]) != FREEZE['prompt_sha256'][p] or schema_sha256(self.schemas[p]) != FREEZE['schema_sha256'][p]:
                raise SystemExit(f'{p} prompt or schema differs from freeze-v7.json; nothing done')

    def path(self, name):
        return os.path.join(self.dir, name)

    def read_jsonl(self, name):
        p = self.path(name)
        return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

    def append(self, name, rows):
        with open(self.path(name), 'a') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush(); os.fsync(f.fileno())

    def write_atomic(self, name, text):
        tmp = self.path(name + '.tmp')
        with open(tmp, 'w') as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.path(name))

    def write_json(self, name, data):
        self.write_atomic(name, json.dumps(data, indent=1, ensure_ascii=False) + '\n')

    def event(self, kind, **kw):
        self.append('events.jsonl', [{'at': now(), 'event': kind, **kw}])

    def body(self, rec):
        p = rec['platform']
        return build_body(model=MODEL, instructions=self.prompts[p], user_content=rec['model_input'], schema_name=f'v9_{p}',
                          schema=self.schemas[p], max_output_tokens=MAX_OUT, reasoning_effort=EFFORT, store=False)


def load_inputs():
    inputs = {}
    for line in open(INPUTS):
        r = json.loads(line)
        k = f"{r['platform']}:{r['record_id']}"
        if k in inputs:
            raise SystemExit(f'duplicate input key {k}; nothing done')
        inputs[k] = r
    return inputs


def make_part(run, name, attempt, keys, inputs):
    """Write one gzipped request file and return its ledger row (the caller appends rows once all files exist)."""
    lines = [json.dumps({'custom_id': f'{k}:a{attempt}', 'method': 'POST', 'url': '/v1/responses', 'body': run.body(inputs[k])},
                        ensure_ascii=False, separators=(',', ':')) for k in keys]
    data = ('\n'.join(lines) + '\n').encode()
    with open(run.path(f'parts/{name}.jsonl.gz'), 'wb') as f:
        f.write(gzip.compress(data, compresslevel=3))
    return {'part': name, 'attempt': attempt, 'requests': len(keys), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'keys': keys, 'prepared_utc': now()}


def prepare(run):
    if run.read_jsonl('parts.jsonl'):
        raise SystemExit('parts already prepared for this run; nothing done')
    inputs = load_inputs()
    keys = list(inputs)
    chunks = [keys[:FIRST_PART]] + [keys[i:i + PART_SIZE] for i in range(FIRST_PART, len(keys), PART_SIZE)]
    rows = [make_part(run, f'a1-{n:04d}', 1, chunk, inputs) for n, chunk in enumerate(chunks, 1)]
    run.append('parts.jsonl', rows)
    run.write_json('manifest.json', {
        'run_id': run.id, 'created_utc': now(), 'freeze': FREEZE, 'inputs': os.path.relpath(INPUTS, ROOT),
        'inputs_sha256': hashlib.sha256(open(INPUTS, 'rb').read()).hexdigest(), 'records': len(keys),
        'reddit': sum(k.startswith('reddit:') for k in keys), 'webmd': sum(k.startswith('webmd:') for k in keys),
        'parts': len(rows), 'part_size': PART_SIZE, 'first_part': FIRST_PART,
        'request_bytes': sum(r['bytes'] for r in rows), 'max_part_bytes': max(r['bytes'] for r in rows),
        'code_sha256': {os.path.relpath(f, ROOT): hashlib.sha256(open(f, 'rb').read()).hexdigest()
                        for f in (os.path.abspath(__file__), os.path.join(ROOT, 'src/weightloss/extraction/reextract.py'),
                                  os.path.join(ROOT, 'src/weightloss/extraction/structured.py'))}})
    print(f'prepared {len(rows)} parts, {len(keys)} records, {sum(r["bytes"] for r in rows) / 1e6:.0f} MB of requests, '
          f'largest part {max(r["bytes"] for r in rows) / 1e6:.1f} MB')


def judge(rec, attempt, item):
    """One batch output line -> (row, retry?). Same acceptance as run.py: one retry for anything but a refusal or a
    content filter; on the retry, items failing the wording check are dropped and the record is kept."""
    resp = item.get('response') or {}
    body = resp.get('body') or {}
    usage = body.get('usage')
    row = {'key': f"{rec['platform']}:{rec['record_id']}", 'attempt': attempt, 'usage': usage, 'usd': cost_usd(MODEL, usage, batch=True) or 0.0}
    if resp.get('status_code') != 200:
        detail = item.get('error') or body.get('error') or f"status_code {resp.get('status_code')}"
        row.update(ok=False, anomaly={'kind': 'batch_error', 'detail': str(detail)[:300]})
    else:
        try:
            parsed = validate_model(rx.output_model(rec['platform']), parse_json_object(final_output_text(body)))
            problems = rx.check_output(parsed, rec['sentences'])
            if not problems:
                row.update(ok=True, output=parsed.model_dump(), dropped_items=[], removed_items=[])
            elif attempt == 1:
                row.update(ok=False, anomaly={'kind': 'check_failed', 'detail': problems[:5]})
            else:
                out, dropped = rx.drop_items(parsed, problems)
                row.update(ok=True, output=out, dropped_items=dropped, removed_items=out.pop('_removed', []))
        except ExtractionAnomaly as e:
            row.update(ok=False, anomaly=e.as_dict())
    retry = not row['ok'] and attempt == 1 and row['anomaly']['kind'] not in NO_RETRY
    return row, retry


def _count(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def evaluate(run, inputs, part, out_lines, err_lines):
    """Evaluate one finished batch into evaluated/<part>.jsonl (rewritten whole, so a repeat is harmless)."""
    keys, attempt, rows = set(part['keys']), part['attempt'], {}
    for raw in out_lines + err_lines:
        item = json.loads(raw)
        key = item['custom_id'].rsplit(':a', 1)[0]
        if key in keys and key not in rows:
            row, retry = judge(inputs[key], attempt, item)
            rows[key] = {**row, 'part': part['part'], 'retry': retry}
    for key in part['keys']:
        if key not in rows:
            rows[key] = {'key': key, 'attempt': attempt, 'part': part['part'], 'usage': None, 'usd': 0.0, 'ok': False,
                         'anomaly': {'kind': 'missing_result', 'detail': 'no line in the batch output'}, 'retry': attempt == 1}
    ordered = [rows[k] for k in part['keys']]
    run.write_atomic(f"evaluated/{part['part']}.jsonl", ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ordered))
    tokens = {}
    for r in ordered:
        for k, v in token_split(r['usage']).items():
            tokens[k] = tokens.get(k, 0) + v
    return {'part': part['part'], 'attempt': attempt, 'requests': len(ordered), 'billed': sum(1 for r in ordered if r['usage']),
            'final': sum(1 for r in ordered if not r['retry']), 'ok': sum(1 for r in ordered if r['ok']),
            'retry': sum(1 for r in ordered if r['retry']), 'anomaly_final': sum(1 for r in ordered if not r['ok'] and not r['retry']),
            'dropped_items': sum(len(r.get('dropped_items') or []) for r in ordered), 'usd': sum(r['usd'] for r in ordered),
            'tokens': tokens, 'anomaly_kinds': _count(r['anomaly']['kind'] for r in ordered if not r['ok'])}


def capacity_error(text):
    t = (text or '').lower()
    return any(c in t for c in CAPACITY)


def remote_match(client, run, part):
    """A batch already on the account for this run, part and payload hash that has not failed, or None."""
    for b in client.batches.list(limit=100).data:
        md = b.metadata or {}
        if md.get('run') == run.id and md.get('part') == part['part'] and md.get('sha') == part['sha256'][:32] and b.status != 'failed':
            return b
    return None


def submit(client, run, part):
    """Create the batch for one part; returns its ledger row, or None after a capacity refusal."""
    existing = remote_match(client, run, part)
    if existing is not None:
        return {'part': part['part'], 'batch_id': existing.id, 'input_file_id': existing.input_file_id, 'created_utc': now(), 'recovered': True}
    with open(run.path(f"parts/{part['part']}.jsonl.gz"), 'rb') as f:
        data = gzip.decompress(f.read())
    if hashlib.sha256(data).hexdigest() != part['sha256']:
        raise SystemExit(f"{part['part']}: request file changed since prepare; stopping")
    up = client.files.create(file=(f"{run.id}-{part['part']}.jsonl", io.BytesIO(data)), purpose='batch')
    try:
        b = client.batches.create(input_file_id=up.id, endpoint='/v1/responses', completion_window='24h',
                                  metadata={'run': run.id, 'part': part['part'], 'sha': part['sha256'][:32]})
    except Exception as e:
        if capacity_error(str(e)):
            run.event('capacity_refusal', part=part['part'], detail=str(e)[:300])
            return None
        b = remote_match(client, run, part)  # a create whose reply was lost may still have succeeded
        if b is None:
            raise
    return {'part': part['part'], 'batch_id': b.id, 'input_file_id': up.id, 'created_utc': now()}


def finish(run, inputs, parts):
    """Merge the final row of every record into final.jsonl (input order) and return the summary."""
    final = {}
    for p in sorted(parts.values(), key=lambda p: (p['attempt'], p['part'])):
        for r in run.read_jsonl(f"evaluated/{p['part']}.jsonl"):
            if not r['retry']:
                final[r['key']] = r
    missing = [k for k in inputs if k not in final]
    if missing:
        raise RuntimeError(f'{len(missing)} records have no final row')
    with open(run.path('final.jsonl.tmp'), 'w') as f:
        for k, rec in inputs.items():
            r = final[k]
            f.write(json.dumps({'platform': rec['platform'], 'record_id': rec['record_id'], 'ok': r['ok'], 'output': r.get('output'),
                                'anomaly': None if r['ok'] else r['anomaly'], 'attempt': r['attempt'],
                                'dropped_items': r.get('dropped_items') or [], 'removed_items': r.get('removed_items') or []},
                               ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(run.path('final.jsonl.tmp'), run.path('final.jsonl'))
    return {'records': len(inputs), 'ok': sum(r['ok'] for r in final.values()), 'anomaly': sum(not r['ok'] for r in final.values()),
            'ok_on_retry': sum(r['ok'] and r['attempt'] == 2 for r in final.values()),
            'dropped_items': sum(len(r.get('dropped_items') or []) for r in final.values()),
            'anomaly_kinds': _count(r['anomaly']['kind'] for r in final.values() if not r['ok']),
            'final_sha256': hashlib.sha256(open(run.path('final.jsonl'), 'rb').read()).hexdigest()}


def ledger(parts, collected, budget, gate_retry):
    """Spend so far, Batch cost per billed request, projected total, and the first-part gate (None until it is collected)."""
    spent = sum(c['usd'] for c in collected.values())
    billed = sum(c['billed'] for c in collected.values())
    per_request = spent / billed if billed else None
    open_requests = sum(p['requests'] for n, p in parts.items() if n not in collected)
    projected = spent + per_request * open_requests if per_request is not None else None
    first, gate = collected.get('a1-0001'), None
    if first is not None:
        gate = 'budget' if projected is None or projected > budget else 'retry_share' if first['retry'] / first['requests'] > gate_retry else 'open'
    return spent, per_request, projected, gate


def drive(run, client, budget, max_parts, poll_seconds, gate_retry):
    inputs = load_inputs()
    lock = open(run.path('driver.lock'), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another driver holds the lock; nothing done')
    if not run.read_jsonl('parts.jsonl'):
        raise SystemExit('no parts prepared; run prepare first')
    backoff_until, started = 0.0, now()
    run.event('driver_start', pid=os.getpid(), budget_usd=budget, max_parts=max_parts, gate_retry=gate_retry)
    while True:
        if os.path.exists(run.path('STOP')):
            run.event('stop_file'); break
        parts = {p['part']: p for p in run.read_jsonl('parts.jsonl')}
        batches = {}
        for b in run.read_jsonl('batches.jsonl'):
            batches[b['part']] = b  # the latest row per part
        collected = {c['part']: c for c in run.read_jsonl('collected.jsonl')}
        spent, per_request, projected, gate = ledger(parts, collected, budget, gate_retry)
        server, last_error, note = {}, None, None
        try:
            # 1. submit: the first part alone, the rest once the first part passed the gate
            live = [n for n in batches if batches[n].get('status') != 'failed_capacity']
            for name, part in parts.items():
                if name in batches and batches[name].get('status') != 'failed_capacity':
                    continue
                if time.time() < backoff_until or (max_parts is not None and len(live) >= max_parts):
                    break
                if name != 'a1-0001' and gate != 'open':
                    break
                row = submit(client, run, part)
                if row is None:
                    backoff_until = time.time() + 900; break
                run.append('batches.jsonl', [row]); batches[name] = row; live.append(name)
                run.event('submitted', part=name, batch_id=row['batch_id'], requests=part['requests'], recovered=row.get('recovered', False))
            # 2. poll and collect
            for name, row in batches.items():
                if name in collected or row.get('status') == 'failed_capacity':
                    continue
                b = client.batches.retrieve(row['batch_id'])
                rc = b.request_counts
                server[name] = {'status': b.status, 'completed': rc.completed if rc else None, 'failed': rc.failed if rc else None,
                                'total': rc.total if rc else None}
                if b.status not in FINAL_STATES:
                    continue
                if b.status == 'failed':
                    detail = json.dumps([e.model_dump() for e in (b.errors.data if b.errors else [])])[:600]
                    if capacity_error(detail):
                        run.append('batches.jsonl', [{**row, 'status': 'failed_capacity', 'at': now(), 'detail': detail}])
                        batches[name] = {**row, 'status': 'failed_capacity'}
                        backoff_until = time.time() + 900
                        run.event('batch_failed_capacity', part=name, detail=detail); continue
                    run.event('batch_failed', part=name, detail=detail)
                out_lines, err_lines = [], []
                for fid, kind, lines in ((b.output_file_id, 'output', out_lines), (b.error_file_id, 'error', err_lines)):
                    if fid:
                        data = client.files.content(fid).read()
                        with open(run.path(f'downloads/{name}-{kind}.jsonl.gz'), 'wb') as f:
                            f.write(gzip.compress(data, compresslevel=3))
                        lines.extend(l for l in data.decode().splitlines() if l.strip())
                summary = {**evaluate(run, inputs, parts[name], out_lines, err_lines), 'status': b.status, 'batch_id': b.id, 'at': now()}
                run.append('collected.jsonl', [summary]); collected[name] = summary
                run.event('collected', **{k: summary[k] for k in ('part', 'status', 'final', 'ok', 'retry', 'usd')})
            # 3. retry parts once every first-attempt part is collected
            if all(n in collected for n, p in parts.items() if p['attempt'] == 1) and not os.path.exists(run.path('retry_done')):
                queue = [r['key'] for n in sorted(parts) for r in run.read_jsonl(f'evaluated/{n}.jsonl') if r['retry']]
                rows = [make_part(run, f'a2-{n:04d}', 2, queue[i:i + PART_SIZE], inputs) for n, i in enumerate(range(0, len(queue), PART_SIZE), 1)]
                run.append('parts.jsonl', rows)
                run.write_atomic('retry_done', f'{len(queue)} records in {len(rows)} parts\n')
                run.event('retry_parts', records=len(queue), parts=len(rows))
        except SystemExit:
            raise
        except Exception as e:  # network loss or a provider error: logged, tried again next tick
            last_error = f'{type(e).__name__}: {str(e)[:300]}'
            run.event('error', detail=last_error, trace=traceback.format_exc()[-1200:])
        # 4. status
        parts = {p['part']: p for p in run.read_jsonl('parts.jsonl')}
        spent, per_request, projected, gate = ledger(parts, collected, budget, gate_retry)
        done = all(n in collected for n in parts) and os.path.exists(run.path('retry_done'))
        live = [n for n in batches if batches[n].get('status') != 'failed_capacity']
        if gate in ('budget', 'retry_share') and all(n in collected for n in live):
            note = f'held at the first-part gate ({gate}); nothing further submitted'
        if max_parts is not None and len(live) >= max_parts and all(n in collected for n in live) and not done:
            note = f'--max-parts {max_parts} reached and collected'
        status = {
            'run_id': run.id, 'updated_utc': now(), 'driver_started_utc': started, 'pid': os.getpid(), 'records': len(inputs),
            'records_final': sum(c['final'] for c in collected.values()), 'records_ok': sum(c['ok'] for c in collected.values()),
            'records_anomaly': sum(c['anomaly_final'] for c in collected.values()),
            'records_retry_queued': sum(c['retry'] for c in collected.values() if c['attempt'] == 1),
            'items_dropped': sum(c['dropped_items'] for c in collected.values()),
            'parts': len(parts), 'parts_submitted': len(live), 'parts_collected': len(collected),
            'server_requests_completed': sum(s['completed'] or 0 for s in server.values()),
            'server_requests_open': sum((s['total'] or 0) - (s['completed'] or 0) - (s['failed'] or 0) for s in server.values()),
            'server': server, 'gate': gate, 'spent_usd': round(spent, 4), 'per_request_usd': per_request,
            'projected_total_usd': round(projected, 2) if projected is not None else None, 'budget_usd': budget,
            'last_error': last_error, 'backoff_until_utc': utc(backoff_until) if backoff_until > time.time() else None,
            'note': note, 'finished': False}
        if done:
            summary = finish(run, inputs, parts)
            status.update(finished=True, final=summary)
            run.write_json('status.json', status)
            run.write_json('summary.json', {**summary, 'spent_usd': round(spent, 4), 'parts': len(parts), 'finished_utc': now(),
                                            'tokens': {k: sum(c['tokens'].get(k, 0) for c in collected.values())
                                                       for k in ('input_tokens', 'cached_tokens', 'cache_write_tokens', 'output_tokens', 'reasoning_tokens')}})
            run.event('finished', ok=summary['ok'], anomaly=summary['anomaly'], spent_usd=round(spent, 4)); break
        run.write_json('status.json', status)
        if note:
            run.event('driver_exit', note=note); break
        time.sleep(poll_seconds)
    lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('step', choices=('prepare', 'drive'))
    ap.add_argument('--max-parts', type=int)
    ap.add_argument('--budget-usd', type=float, default=60.0)
    ap.add_argument('--poll-seconds', type=float, default=300.0)
    ap.add_argument('--gate-retry', type=float, default=0.05, help='largest share of first-part records needing a retry')
    a = ap.parse_args()
    run = Run(a.run_id)
    if a.step == 'prepare':
        prepare(run); return
    if not os.environ.get('OPENAI_API_KEY'):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('"').strip("'")
    from openai import OpenAI
    drive(run, OpenAI(max_retries=2, timeout=900), a.budget_usd, a.max_parts, a.poll_seconds, a.gate_retry)


if __name__ == '__main__':
    main()
