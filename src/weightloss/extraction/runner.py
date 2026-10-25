"""Run ledger and bounded synchronous runner for compact extraction; Batch steps live in batch.py.

A run directory pins everything a result depends on: manifest.json (model, reasoning effort, schema and prompt
fingerprints, prices, SDK version), schema.json and inputs.jsonl, the exact text snapshot that evidence spans refer to.
It holds source text and model answers, so keep it in a private location such as .cache/. All ledgers are append-only
JSON lines:

  requests.jsonl   one row per request built (custom_id '<record_id>@<attempt>', sync or batch), written before sending
  responses.jsonl  raw response bodies, written before anything is derived from them
  attempts.jsonl   synchronous HTTP outcomes, including uncertain charges for timeouts, connection loss and 5xx
  usage.jsonl      tokens and cost per response
  results.jsonl    per custom_id: ok + compact record, or anomaly {kind, detail, retriable}
  held.jsonl       requests that may have been answered but whose body is missing; never re-sent automatically

A record's final state is its earliest ok result; otherwise its latest anomaly. Restarting never re-sends a record with
an ok result or a held request.

Scale (2026-10-01, full-batch-p7c1-luna-low-20261001-01a0f4c2): the ledger files are append-only, and one driver writes
them, so a ledger object reads them incrementally. Small control files (batch parts, batches, intents, events, ...)
are cached and only the appended bytes are parsed. requests, results, usage, attempts and held feed a slim incremental
index (per-record attempts, earliest ok, latest anomaly, held; answered and costed custom ids; spend). summary(),
spent_and_uncertain() and recover() read only that index and new bytes, never all 256k rows on every call.
append_many() writes a batch of rows with one fsync. Batch-mode responses rows hold the raw body with the echoed
instructions and JSON schema replaced by their sha256 ("slim"), but only when both equal the run's request. The exact
provider bytes stay in the gzipped batch output file that the row names. states() still reads the full rows (export,
final delivery).
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import threading
import time

from . import compact
from .structured import (BATCH_FACTOR, PRICES_PER_1M, PROVIDER_KINDS, ExtractionAnomaly, compact_json, cost_usd, http_retriable, mask,
                         request_cap_usd, sha256_text, token_split)

RUN_ID = re.compile(r'^[A-Za-z0-9._-]{1,64}$')
# canonical_version (2026-09-30): a run of an opt-in canonical spec must reopen with the same canonical contract; manifests
# and specs without one both read as absent, so older runs reopen unchanged.
FINGERPRINT_KEYS = ('schema_name', 'schema_version', 'schema_sha256', 'prompt_version', 'prompt_sha256', 'text_verbosity',
                    'canonical_version', 'contract_version')
# retry_policy 'provider-only' (2026-09-30): an ordinary retry only when no usable reply exists (provider/protocol stages);
# a reply that fails local validation is kept for path-only correction instead of being re-extracted. None = historical.
RETRY_POLICIES = ('provider-only',)
DRIVER_LOCK = 'roll.lock'  # the same lock file as batch.roll: one live driver (sync, repair, roll) per run
# control files read through the tail cache (only appended bytes are parsed)
CACHED = frozenset({'batch_parts', 'batches', 'batch_collected', 'batch_intents', 'batch_downloads', 'batch_errors', 'batch_requeues',
                    'roll_events', 'held', 'attempts'})
TOKEN_KEYS = ('input_tokens', 'cached_tokens', 'cache_write_tokens', 'output_tokens', 'reasoning_tokens')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def custom_id(record_id, attempt):
    return f'{record_id}@{attempt}'


def split_custom_id(value):
    record_id, _, attempt = value.rpartition('@')
    return record_id, int(attempt)


def sdk_version():
    try:
        import openai
        return openai.__version__
    except ImportError:
        return None


class TransportError(Exception):
    """Timeout or connection loss: the request may have been processed and billed."""


class RunLedger:
    def __init__(self, run_dir, manifest):
        self.dir, self.manifest = Path(run_dir), manifest
        self.spec = compact.get_spec(manifest.get('spec', compact.CANONICAL.name))
        self.lock = threading.RLock()
        self._inputs = None
        self._tails = {}  # name -> {'key': (inode, device), 'offset': bytes consumed, 'rows': parsed rows (CACHED only)}
        self._index = None

    @property
    def run_id(self):
        return self.manifest['run_id']

    @classmethod
    def create(cls, run_dir, run_id, inputs, *, model=compact.DEFAULT_MODEL, reasoning_effort=compact.DEFAULT_REASONING,
               max_output_tokens=compact.DEFAULT_MAX_OUTPUT_TOKENS, source=None, spec=None, localize=None, retry_policy=None):
        run_dir = Path(run_dir)
        spec = spec or compact.CANONICAL
        if not RUN_ID.match(run_id):
            raise ValueError('run_id must be 1-64 characters from A-Z a-z 0-9 . _ -')
        if (run_dir/'manifest.json').exists():
            raise FileExistsError(f'{run_dir} already holds a run')
        inputs = list(inputs)
        ids = [i.record_id for i in inputs]
        if not inputs or len(set(ids)) != len(ids):
            raise ValueError('inputs must be non-empty with unique record_id values')
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = ''.join(compact_json({'record_id': i.record_id, 'kind': i.kind, 'fields': dict(i.fields)}) + '\n' for i in inputs)
        (run_dir/'inputs.jsonl').write_text(lines)
        (run_dir/'schema.json').write_text(json.dumps(spec.schema, indent=1) + '\n')
        (run_dir/'instructions.txt').write_text(spec.instructions)
        manifest = {'run_id': run_id, 'created_utc': now(), 'spec': spec.name, 'model': model, 'reasoning_effort': reasoning_effort,
                    'max_output_tokens': max_output_tokens, **spec.fingerprint(),
                    'inputs': {'records': len(inputs), 'sha256': sha256_text(lines), 'source': source},
                    'prices_per_1m': PRICES_PER_1M.get(model), 'batch_factor': BATCH_FACTOR, 'openai_sdk': sdk_version()}
        if localize is not None:
            from . import localize as localize_module
            if localize not in localize_module.POLICIES:
                raise ValueError(f'unknown localization policy {localize!r}')
            manifest['localize'] = localize
        if retry_policy is not None:
            if retry_policy not in RETRY_POLICIES:
                raise ValueError(f'unknown retry policy {retry_policy!r}')
            manifest['retry_policy'] = retry_policy
        tmp = run_dir/'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n')
        os.replace(tmp, run_dir/'manifest.json')
        return cls(run_dir, manifest)

    @classmethod
    def open(cls, run_dir):
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir/'manifest.json').read_text())
        current = compact.get_spec(manifest.get('spec', compact.CANONICAL.name)).fingerprint()
        changed = [k for k in FINGERPRINT_KEYS if manifest.get(k) != current.get(k)]
        if changed:
            raise ValueError(f'run {manifest["run_id"]} was created with a different {", ".join(changed)}; start a new run')
        if sha256_text((run_dir/'inputs.jsonl').read_text()) != manifest['inputs']['sha256']:
            raise ValueError('inputs.jsonl no longer matches the run manifest')
        return cls(run_dir, manifest)

    def inputs(self):
        if self._inputs is None:
            self._inputs = {}
            for row in self.rows('inputs'):
                self._inputs[row['record_id']] = compact.ExtractionInput(row['record_id'], row['fields'], row['kind'])
        return self._inputs

    def request_body(self, inp):
        m = self.manifest
        return compact.build_request(inp, spec=self.spec, model=m['model'], reasoning_effort=m['reasoning_effort'],
                                     max_output_tokens=m['max_output_tokens'])

    def append(self, name, row):
        self.append_many(name, [row])

    def append_many(self, name, rows):
        """Append rows to one ledger file with a single fsync (a torn tail from an interrupted writer is closed first)."""
        if not rows:
            return
        data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        with self.lock, open(self.dir/f'{name}.jsonl', 'ab+') as stream:
            if stream.seek(0, 2):
                stream.seek(-1, 2)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')  # a torn write from an interrupted process ends without newline
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _new_rows(self, name, keep=False):
        """Rows appended to name.jsonl since the last call for this name (complete lines only; a torn line carries no
        fact). keep=True also keeps every parsed row in memory (control files)."""
        path = self.dir/f'{name}.jsonl'
        with self.lock:
            tail = self._tails.setdefault(name, {'key': None, 'offset': 0, 'rows': []})
            if not path.exists():
                if tail['offset']:
                    tail.update(key=None, offset=0, rows=[])
                    if not keep:
                        self._index = None  # a file vanished: rebuild the index from scratch
                return []
            st = path.stat()
            key = (st.st_ino, st.st_dev)
            if tail['key'] != key or st.st_size < tail['offset']:  # replaced or truncated: read it again from the start
                if tail['offset'] and not keep:
                    self._index = None
                tail.update(key=key, offset=0, rows=[])
            if st.st_size == tail['offset']:
                return []
            with open(path, 'rb') as stream:
                stream.seek(tail['offset'])
                data = stream.read(st.st_size - tail['offset'])
            end = data.rfind(b'\n') + 1
            new = []
            for line in data[:end].splitlines():
                try:
                    new.append(json.loads(line))
                except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn write carries no fact
                    continue
            tail['offset'] += end
            if keep:
                tail['rows'].extend(new)
            return new

    def rows(self, name):
        """Committed rows; a line that does not parse is a torn write (rows are fsynced whole) and carries no fact.
        Control files come from the tail cache; the large files are read in full (use index() on hot paths)."""
        if name in CACHED:
            with self.lock:
                self._new_rows(name, keep=True)
                return list(self._tails[name]['rows'])
        path = self.dir/f'{name}.jsonl'
        if not path.exists():
            return []
        out = []
        with open(path, 'rb') as stream:
            for line in stream:
                try:
                    out.append(json.loads(line))
                except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn write carries no fact
                    continue
        return out

    def index(self):
        """Slim incremental view of requests, results, usage, attempts and held (only rows appended since the last
        call are read): {'states': {record_id: {attempts, ok, last, held}}, 'answered', 'requested', 'retry_requests',
        'used', 'spent', 'uncertain', 'tokens', 'first'}. ok/last keep custom_id, attempt and (ok) keep or (last) the
        anomaly kind, retriable flag and stage, as states() does with full rows."""
        with self.lock:
            if self._index is None:
                for name in ('requests', 'results', 'usage', 'attempts', 'held'):
                    self._tails.pop(f'ix:{name}', None)
                self._index = {'states': {rid: {'attempts': 0, 'ok': None, 'last': None, 'held': False} for rid in self.inputs()},
                               'answered': set(), 'requested': {}, 'retry_requests': 0, 'used': set(), 'spent': 0.0, 'uncertain': 0.0,
                               'tokens': {}, 'first': Counter(), 'retry': Counter()}
            ix = self._index
            states = ix['states']
            for row in self._ix_rows('requests'):
                cid, rid = row['custom_id'], row['record_id']
                if cid not in ix['requested']:
                    ix['requested'][cid] = row.get('mode')
                    ix['retry_requests'] += row['attempt'] > 1
                states[rid]['attempts'] = max(states[rid]['attempts'], row['attempt'])
            for row in self._ix_rows('results'):
                ix['answered'].add(row['custom_id'])
                state = states[row['record_id']]
                tally = ix['first'] if row['attempt'] == 1 else ix['retry']
                tally[row['outcome']] += 1
                if row['outcome'] == 'anomaly':
                    tally['anomaly:' + row['anomaly']['kind']] += 1
                if row['outcome'] == 'ok':
                    if state['ok'] is None or row['attempt'] < state['ok']['attempt']:
                        state['ok'] = {'custom_id': row['custom_id'], 'attempt': row['attempt'], 'keep': row.get('keep')}
                elif state['last'] is None or row['attempt'] >= state['last']['attempt']:
                    a = row['anomaly']
                    state['last'] = {'custom_id': row['custom_id'], 'attempt': row['attempt'],
                                     'anomaly': {'kind': a['kind'], 'retriable': a['retriable'], 'stage': a.get('stage')}}
            for row in self._ix_rows('usage'):
                key = (row.get('mode'), row['custom_id'])
                if key in ix['used']:
                    continue  # one costed row per response (a replayed collect never counts twice)
                ix['used'].add(key)
                ix['spent'] += row.get('cost_usd') or 0
                t = ix['tokens'].setdefault(row.get('mode'), {**dict.fromkeys(TOKEN_KEYS, 0), 'responses': 0, 'cost_usd': 0.0})
                for k in TOKEN_KEYS:
                    t[k] += row.get(k) or 0
                t['responses'] += 1
                t['cost_usd'] += row.get('cost_usd') or 0
            for row in self._ix_rows('attempts'):
                ix['uncertain'] += row.get('uncertain_charge_usd', 0) or 0
            for row in self._ix_rows('held'):
                states[split_custom_id(row['custom_id'])[0]]['held'] = True
            return ix

    def _ix_rows(self, name):
        """New rows of name for the index, on a tail of their own (independent of the row cache)."""
        path = self.dir/f'{name}.jsonl'
        tail = self._tails.setdefault(f'ix:{name}', {'key': None, 'offset': 0, 'rows': []})
        if not path.exists():
            return []
        st = path.stat()
        if tail['key'] != (st.st_ino, st.st_dev) or st.st_size < tail['offset']:
            if tail['offset']:
                raise RuntimeError(f'{name}.jsonl was replaced or truncated under a live index; reopen the run')
            tail.update(key=(st.st_ino, st.st_dev), offset=0)
        if st.st_size == tail['offset']:
            return []
        with open(path, 'rb') as stream:
            stream.seek(tail['offset'])
            data = stream.read(st.st_size - tail['offset'])
        end = data.rfind(b'\n') + 1
        tail['offset'] += end
        out = []
        for line in data[:end].splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn write carries no fact
                continue
        return out

    def log_request(self, cid, mode, body, **extra):
        record_id, attempt = split_custom_id(cid)
        self.append('requests', {'custom_id': cid, 'record_id': record_id, 'attempt': attempt, 'mode': mode,
                                 'body_sha256': sha256_text(compact_json(body)), 'at': now(), **extra})

    def record_response(self, cid, mode, response):
        """Store the raw body first, then usage and the evaluated result; returns the result row."""
        record_id, attempt = split_custom_id(cid)
        self.append('responses', {'custom_id': cid, 'mode': mode, 'response': response})
        self.append('usage', self.usage_row(cid, mode, response))
        return self.evaluate(cid, mode, response)

    def usage_row(self, cid, mode, response):
        usage = response.get('usage') or {}
        return {'custom_id': cid, 'mode': mode, 'response_id': response.get('id'), 'status': response.get('status'),
                'model': response.get('model'), **token_split(usage),
                'cost_usd': cost_usd(self.manifest['model'], usage, batch=mode == 'batch'), 'at': now()}

    def slim(self, response):
        """A response body without the echoed instructions and JSON schema when both equal this run's request (their
        sha256 is kept); otherwise the body unchanged. Nothing the evaluation reads is removed."""
        if getattr(self, '_slim_ref', None) is None:
            body = self.request_body(next(iter(self.inputs().values())))
            self._slim_ref = (body['instructions'], json.dumps(body['text']['format']['schema'], sort_keys=True),
                              {'instructions_sha256': sha256_text(body['instructions']),
                               'schema_sha256': sha256_text(json.dumps(body['text']['format']['schema'], sort_keys=True))})
        instructions, schema, marker = self._slim_ref
        fmt = (response.get('text') or {}).get('format') or {}
        if response.get('instructions') != instructions or json.dumps(fmt.get('schema'), sort_keys=True) != schema:
            return response
        out = {k: v for k, v in response.items() if k != 'instructions'}
        out['text'] = {**response['text'], 'format': {k: v for k, v in fmt.items() if k != 'schema'}}
        out['_slim'] = dict(marker)
        return out

    def record_batch_responses(self, items):
        """One batch part's 200 replies [(custom_id, raw body, source)]: slim raw bodies first (one fsync), then the usage
        rows not booked yet, then the evaluated results (one fsync each). Returns the result rows in order. `source`
        ({'file', 'line', 'line_sha256'}) points at the exact provider line in the gzipped output file, which is written
        and fsynced before this call."""
        if not items:
            return []
        ix = self.index()
        self.append_many('responses', [{'custom_id': cid, 'mode': 'batch', 'response': self.slim(body), **({'source': src} if src else {})}
                                       for cid, body, src in items])
        self.append_many('usage', [self.usage_row(cid, 'batch', body) for cid, body, _ in items if ('batch', cid) not in ix['used']])
        rows = [self.evaluate_row(cid, 'batch', body) for cid, body, _ in items]
        self.append_many('results', rows)
        return rows

    def evaluate(self, cid, mode, response):
        row = self.evaluate_row(cid, mode, response)
        self.append('results', row)
        return row

    def evaluate_row(self, cid, mode, response):
        record_id, attempt = split_custom_id(cid)
        row = {'custom_id': cid, 'record_id': record_id, 'attempt': attempt, 'mode': mode, 'at': now()}
        policy, repairs = self.manifest.get('localize'), []
        try:
            record = compact.evaluate_response(response, self.inputs()[record_id], self.run_id, self.spec,
                                               localize_policy=policy, repairs=repairs)
            row.update(outcome='ok', keep=compact.retention(record)['keep'], record=record)
            if repairs:
                row['localization'] = {'policy': policy, 'repairs': repairs}
        except ExtractionAnomaly as anomaly:
            row.update(outcome='anomaly', anomaly=anomaly.as_dict())
            if repairs:  # audit partial repairs too; the raw reply is kept in responses.jsonl either way
                row['localization'] = {'policy': policy, 'repairs': repairs, 'applied': False}
        return row

    def record_anomaly(self, cid, mode, anomaly):
        record_id, attempt = split_custom_id(cid)
        row = {'custom_id': cid, 'record_id': record_id, 'attempt': attempt, 'mode': mode, 'at': now(),
               'outcome': 'anomaly', 'anomaly': anomaly.as_dict()}
        self.append('results', row)
        return row

    def states(self):
        states = {rid: {'attempts': 0, 'ok': None, 'last': None, 'held': False} for rid in self.inputs()}
        for row in self.rows('requests'):
            states[row['record_id']]['attempts'] = max(states[row['record_id']]['attempts'], row['attempt'])
        for row in self.rows('results'):
            state = states[row['record_id']]
            if row['outcome'] == 'ok':
                if state['ok'] is None or row['attempt'] < state['ok']['attempt']:
                    state['ok'] = row
            elif state['last'] is None or row['attempt'] >= state['last']['attempt']:
                state['last'] = row
        for row in self.rows('held'):
            states[split_custom_id(row['custom_id'])[0]]['held'] = True
        return states

    def spent_and_uncertain(self):
        ix = self.index()
        return ix['spent'], ix['uncertain']

    def recover(self):
        """Finish work an interrupted process left: saved bodies without a result are evaluated; synchronous requests
        with no outcome at all count as transport errors (uncertain charge); HTTP 200 without a saved body is held."""
        with self.lock:
            ix = self.index()
            answered = set(ix['answered'])
            used = {cid for _, cid in ix['used']}
            saved, path = set(), self.dir/'responses.jsonl'
            if path.exists():
                with open(path, 'rb') as stream:  # streamed: bodies are never all held in memory
                    for line in stream:
                        try:
                            row = json.loads(line)
                        except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn write carries no fact
                            continue
                        cid = row['custom_id']
                        if cid in saved:
                            continue
                        saved.add(cid)
                        if cid not in answered:
                            if cid not in used:
                                self.record_response(cid, row['mode'], row['response'])
                            else:
                                self.evaluate(cid, row['mode'], row['response'])
                            answered.add(cid)
            attempts = self.rows('attempts')
            ended = {a['custom_id'] for a in attempts if a.get('http_status') != 429}
            rate_limited = {a['custom_id'] for a in attempts if a.get('http_status') == 429}
            held = {h['custom_id'] for h in self.rows('held')}
            for a in attempts:
                if a.get('http_status') == 200 and a['custom_id'] not in saved and a['custom_id'] not in held:
                    self.append('held', {'custom_id': a['custom_id'], 'reason': 'HTTP 200 recorded but body not saved', 'at': now()})
                    held.add(a['custom_id'])
            for row in self.rows('requests'):
                cid = row['custom_id']
                if row['mode'] == 'sync' and cid in rate_limited and cid not in ended and cid not in answered:
                    # interrupted while waiting after HTTP 429: nothing was processed, so no charge; retriable
                    self.record_anomaly(cid, 'sync', ExtractionAnomaly('http_error', 'HTTP 429 (interrupted during the wait)', retriable=True))
                    answered.add(cid)
                elif row['mode'] == 'sync' and cid not in ended and cid not in answered and cid not in held:
                    cap = request_cap_usd(self.manifest['model'], self.request_body(self.inputs()[row['record_id']]))
                    self.append('attempts', {'custom_id': cid, 'error': 'no outcome recorded (interrupted)', 'uncertain_charge_usd': round(cap, 9), 'at': now()})
                    self.record_anomaly(cid, 'sync', ExtractionAnomaly('transport_error', 'interrupted before an outcome was recorded'))
                    ended.add(cid)

    def summary(self):
        ix = self.index()
        states = ix['states']
        spent, uncertain = ix['spent'], ix['uncertain']
        final = {'ok_keep': 0, 'ok_no_keep': 0, 'held': 0, 'pending': 0}
        anomalies = {}
        for state in states.values():
            if state['ok']:
                final['ok_keep' if state['ok']['keep'] else 'ok_no_keep'] += 1
            elif state['held']:
                final['held'] += 1
            elif state['last']:
                kind = state['last']['anomaly']['kind']
                anomalies[kind] = anomalies.get(kind, 0) + 1
            else:
                final['pending'] += 1
        tokens = {}
        for mode in ('sync', 'batch'):
            if mode in ix['tokens']:
                t = ix['tokens'][mode]
                tokens[mode] = {**{k: t[k] for k in TOKEN_KEYS}, 'responses': t['responses'], 'cost_usd': round(t['cost_usd'], 9)}
        out = {'run_id': self.run_id, 'updated_utc': now(), 'records': len(states), 'final': final, 'final_anomalies': anomalies,
               'requests': len(ix['requested']), 'retry_requests': ix['retry_requests'],
               'spent_usd': round(spent, 9), 'uncertain_charges_usd': round(uncertain, 9), 'usage': tokens}
        tmp = self.dir/'status.json.tmp'
        tmp.write_text(json.dumps(out, indent=1) + '\n')
        os.replace(tmp, self.dir/'status.json')
        return out

    def unresolved(self):
        """{record_id: reason} for every input record without a validated result: its last anomaly kind, 'held' or
        'pending'. Empty means complete delivery."""
        return {rid: 'held' if s['held'] else (s['last']['anomaly']['kind'] if s['last'] else 'pending')
                for rid, s in self.states().items() if not s['ok']}

    def export_complete(self, path):
        """Final-delivery export (opt-in; export() still writes partial analysis sets). Refuses with ValueError and
        writes nothing unless every frozen input record has exactly one validated result; nothing is dropped or
        emptied to pass."""
        missing = self.unresolved()
        if missing:
            shown = ', '.join(f'{rid} ({why})' for rid, why in sorted(missing.items())[:5])
            raise ValueError(f'final export refused: {len(missing)} of {len(self.inputs())} records have no validated result: {shown}')
        path = Path(path)
        tmp = path.with_name(path.name + '.tmp')
        count = self.export(tmp)
        written = [json.loads(line)['record_id'] for line in tmp.read_text().splitlines()]
        if count != len(self.inputs()) or written != list(self.inputs()):
            tmp.unlink()
            raise ValueError('final export refused: exported records do not match the frozen inputs one to one')
        os.replace(tmp, path)
        return count

    def export(self, path):
        """ANALYSIS-ONLY partial export: each record's final ok compact record, silently skipping records without one
        (retention computed at read time, not stored). Never a final delivery: use export_complete, or
        `xopt.py export-final`, which refuses unless every frozen input validates."""
        states = self.states()
        count = 0
        with open(path, 'w') as stream:
            for rid in self.inputs():
                if states[rid]['ok']:
                    stream.write(compact_json(states[rid]['ok']['record']) + '\n')
                    count += 1
        return count


def needs_attempt(state, max_attempts, retry_policy=None):
    if state['ok'] or state['held'] or state['attempts'] >= max_attempts:
        return False
    if state['attempts'] == 0:
        return True
    last = state['last']
    if not (last and last['attempt'] == state['attempts'] and last['anomaly']['retriable']):
        return False
    return retry_policy is None or last['anomaly']['kind'] in PROVIDER_KINDS


def acquire_driver_lock(run_dir):
    """Exclusive flock on the run's driver lock (shared with batch.roll); raises RuntimeError if another driver holds it."""
    import fcntl
    stream = open(Path(run_dir)/DRIVER_LOCK, 'a+')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f'{run_dir} is driven by another live process; nothing started') from None
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps({'pid': os.getpid(), 'at': now()}))
    stream.flush()
    return stream


def release_driver_lock(stream):
    import fcntl
    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


class Budget:
    """Every attempt reserves its conservative cap first; settle() books the outcome in one locked step."""

    def __init__(self, limit_usd, spent=0.0, uncertain=0.0):
        self.limit, self.spent, self.uncertain, self.reserved = limit_usd, spent, uncertain, 0.0
        self.stopped = False
        self.lock = threading.Lock()

    def reserve(self, cap):
        with self.lock:
            if self.stopped or self.spent + self.uncertain + self.reserved + cap > self.limit:
                self.stopped = True
                return False
            self.reserved += cap
            return True

    def settle(self, cap, spent=0.0, uncertain=0.0):
        with self.lock:
            self.reserved -= cap
            self.spent += spent
            self.uncertain += uncertain


def openai_send(client):
    """Transport over the official SDK. The client should use max_retries=0 so retries stay bounded here."""
    import openai

    def send(body):
        try:
            raw = client.responses.with_raw_response.create(**body)
        except openai.APIStatusError as error:
            return error.status_code, dict(error.response.headers), error.response.text
        except openai.APIConnectionError as error:  # includes timeouts
            raise TransportError(type(error).__name__) from None
        return raw.status_code, dict(raw.headers), raw.text
    return send


def make_client(max_retries=0, timeout=600):
    """Official client; the key comes from OPENAI_API_KEY and is never logged."""
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is required; nothing was sent.')
    from openai import OpenAI
    return OpenAI(max_retries=max_retries, timeout=timeout)


def run_sync(ledger, send, *, budget_usd, max_attempts=2, concurrency=4, limit=None, record_ids=None,
             max_rate_limit_waits=6, sleep=time.sleep):
    """Send every record that needs an attempt, at most max_attempts requests per record in total, while the
    conservative caps stay within budget_usd (spend and uncertain charges already in the ledger included)."""
    if max_attempts < 1 or concurrency < 1:
        raise ValueError('max_attempts and concurrency must be positive')
    lock = acquire_driver_lock(ledger.dir)
    try:
        return _run_sync(ledger, send, budget_usd=budget_usd, max_attempts=max_attempts, concurrency=concurrency, limit=limit,
                         record_ids=record_ids, max_rate_limit_waits=max_rate_limit_waits, sleep=sleep)
    finally:
        release_driver_lock(lock)


def _run_sync(ledger, send, *, budget_usd, max_attempts, concurrency, limit, record_ids, max_rate_limit_waits, sleep):
    policy = ledger.manifest.get('retry_policy')
    ledger.recover()
    states = ledger.states()
    inputs = ledger.inputs()
    wanted = set(record_ids) if record_ids is not None else None
    todo = [inp for rid, inp in inputs.items() if (wanted is None or rid in wanted) and needs_attempt(states[rid], max_attempts, policy)]
    if limit is not None:
        todo = todo[:limit]
    spent, uncertain = ledger.spent_and_uncertain()
    budget = Budget(budget_usd, spent, uncertain)
    model = ledger.manifest['model']

    def backoff(n):
        sleep(min(60, 2 ** n) + random.random())

    def one(inp):
        attempt, waits = states[inp.record_id]['attempts'] + 1, 0
        body = ledger.request_body(inp)
        cap = request_cap_usd(model, body)
        while attempt <= max_attempts:
            cid = custom_id(inp.record_id, attempt)
            if not budget.reserve(cap):
                ledger.append('attempts', {'custom_id': cid, 'skipped': 'budget', 'at': now()})
                return 'budget'
            ledger.log_request(cid, 'sync', body)
            started = time.monotonic()
            try:
                status, headers, text = send(body)
            except TransportError as error:
                budget.settle(cap, uncertain=cap)
                ledger.append('attempts', {'custom_id': cid, 'error': str(error), 'uncertain_charge_usd': round(cap, 9), 'at': now()})
                ledger.record_anomaly(cid, 'sync', ExtractionAnomaly('transport_error', str(error)))
                attempt += 1
                backoff(attempt)
                continue
            base = {'custom_id': cid, 'http_status': status, 'request_id': (headers or {}).get('x-request-id'),
                    'latency_s': round(time.monotonic() - started, 2), 'at': now()}
            if status == 429 and waits < max_rate_limit_waits:
                budget.settle(cap)
                ledger.append('attempts', base)
                waits += 1
                retry_after = (headers or {}).get('retry-after')
                sleep(float(retry_after) if retry_after else min(60, 2 ** waits) + random.random())
                continue  # rate limited before any processing: same attempt number
            if status != 200:
                uncertain_charge = cap if status >= 500 else 0.0
                budget.settle(cap, uncertain=uncertain_charge)
                ledger.append('attempts', {**base, 'error': mask(text)[:500], 'uncertain_charge_usd': round(uncertain_charge, 9)})
                retriable = http_retriable(status)
                ledger.record_anomaly(cid, 'sync', ExtractionAnomaly('http_error', f'HTTP {status}', retriable=retriable))
                if retriable:
                    attempt += 1
                    backoff(attempt)
                    continue
                return 'http_error'
            try:
                response = json.loads(text)
            except ValueError:  # JSONDecodeError or UnicodeDecodeError: a torn write carries no fact
                response = None
                error = 'HTTP 200 body is not JSON'
            else:
                error = None if isinstance(response, dict) else 'HTTP 200 body is JSON but not an object'
            if error:  # answered and possibly billed but unreadable: book the cap, hold, never re-send (nothing is saved to replay)
                budget.settle(cap, uncertain=cap)
                ledger.append('attempts', {**base, 'error': error, 'uncertain_charge_usd': round(cap, 9)})
                ledger.append('held', {'custom_id': cid, 'reason': f'{error}; held, not re-sent', 'at': now()})
                return 'held'
            row = ledger.record_response(cid, 'sync', response)
            budget.settle(cap, spent=cost_usd(model, response.get('usage')) or 0.0)
            ledger.append('attempts', base)
            if row['outcome'] == 'ok':
                return 'ok'
            if not row['anomaly']['retriable'] or (policy is not None and row['anomaly']['kind'] not in PROVIDER_KINDS):
                return row['anomaly']['kind']  # provider-only: a usable but locally invalid reply goes to path-only correction
            attempt += 1
        return 'attempts_exhausted'

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(one, todo))
    summary = ledger.summary()
    summary['this_invocation'] = {'records_tried': len(todo), 'stopped_by_budget': outcomes.count('budget'),
                                  'budget_usd': budget_usd, 'max_attempts': max_attempts}
    return summary
