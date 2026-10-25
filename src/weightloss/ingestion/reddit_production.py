"""Persistent, fair scheduling for the FDA-bounded Reddit production collection."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from .reddit import (ArcticClient, Pilot, VERIFICATION_VERSION, target_spec, utc,
                     verify_collection, verification_passed, hydrate_context, repair_from_reverse, normalize_reset_timestamp)

from .reddit_transport import RequestGate, KeepAliveTransport

PRIORITY = ('victoza', 'saxendaforweightloss', 'trulicity', 'rybelsuspill',
            'ozempict2d', 'wegovymaintenance', 'zepbound_maintenance', 'ozempicmaintenance')


def request_policy(directory):
    """Transport tuning is separate from the immutable cohort and page cursors."""
    path = Path(directory)/'request-policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    interval = policy.get('interval_seconds', 1.0)
    if (isinstance(interval, bool) or not isinstance(interval, (int, float))
            or not math.isfinite(interval) or not 0.5 <= interval <= 5):
        raise ValueError('Request interval must be finite and between 0.5 and 5 seconds')
    workers = policy.get('workers', 1)
    keep_alive = policy.get('keep_alive', False)
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError('Workers must be 1 or 2')
    if type(keep_alive) is not bool:
        raise ValueError('keep_alive must be boolean')
    return dict(policy, interval_seconds=float(interval), workers=workers, keep_alive=keep_alive)


def write_json(path, value):
    # The worker and read-only supervisor may export status simultaneously.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as out:
        temp = Path(out.name)
        try:
            out.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
            out.flush()
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def source_done(manifest):
    verification = manifest.get('verification', {})
    context = manifest.get('context_audit', {})
    return (verification_passed(verification)
            and context.get('status') in ('complete', 'partial')
            and not context.get('error_type')
            and not any(r['status'] == 'pending' for r in context.get('counts', [])))


def prune_old_exports(directory):
    """Only this production run's superseded derived exports; never SQLite/raw rows."""
    for folder in (directory, directory/'reverse-audit'):
        manifest_path = folder/'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        retained = {(folder/p).resolve() for p in manifest.get('export_files', {}).values()}
        if not retained:
            continue
        for path in (folder/'exports').glob('*.jsonl'):
            if re.fullmatch(r'(posts|comments)-[0-9a-f]{64}\.jsonl', path.name) and path.resolve() not in retained:
                path.unlink()


class DiskGuardClient(ArcticClient):
    def __init__(self, directory, reserve_bytes, **kwargs):
        super().__init__(**kwargs)
        self.directory, self.reserve_bytes = directory, reserve_bytes

    def request(self, path, params):
        if shutil.disk_usage(self.directory).free < self.reserve_bytes:
            raise ValueError('Disk reserve reached; collection checkpoint retained')
        return super().request(path, params)


def _collect_community(directory, spec, end, client, per_community):
    """Thread owns its Pilot, request log, direction, connection and export files."""
    pilot = None
    try:
        pilot = Pilot(directory, spec, end=end)
        pilot.run(client, max_windows=per_community, export=False)
        if pilot.get_meta('not_before', 0) <= time.time():
            verified = pilot.get_meta('verification', {})
            if not verification_passed(verified):
                verify_collection(pilot, client, per_community, defer_exports=True)
            if pilot.get_meta('verification', {}).get('status') == 'mismatch' and client.calls < client.budget:
                repair_from_reverse(pilot, client)
                if pilot.get_meta('not_before', 0) <= time.time():
                    verify_collection(pilot, client, per_community, defer_exports=True)
            if verification_passed(pilot.get_meta('verification', {})):
                hydrate_context(pilot, client)
        manifest = pilot.export()
        result = {'source_done': source_done(manifest),
                  **{k: manifest[k] for k in ('status', 'counts', 'verification', 'context_audit')},
                  'last_served_at': utc(time.time())}
        # Also propagate a cooldown saved before this process started.
        if pilot.get_meta('not_before', 0) > time.time():
            client.on_rate_limit(pilot.get_meta('not_before'))
        prune_old_exports(directory)
        return result
    except KeyboardInterrupt:
        if pilot is not None:
            pilot.export()
        raise
    except Exception as error:
        result = {'source_done': False, 'error_type': type(error).__name__, 'error': str(error)[:300]}
        if pilot is not None:
            result.update(pilot.summary())
        return result
    finally:
        if pilot is not None:
            pilot.close()


def collect_production(output_dir, max_requests=1200, per_community=200, reserve_gib=15):
    from weightloss.settings import get_settings
    if max_requests < 1 or per_community < 1 or reserve_gib < 1:
        raise ValueError('Budgets and reserve must be positive')
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'.production.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {'status': 'already_running', 'output_dir': str(directory)}
    try:
        if (directory/'scope-removal-pending.json').exists():
            raise ValueError('Finish pending scope removal before collecting')
        config = json.loads((get_settings().root/'configs/reddit.json').read_text())
        specs = {name: target_spec(config, name) for name in sorted(config['targets'])}
        pinned = directory/'production-spec.json'
        if pinned.exists():
            spec = json.loads(pinned.read_text())
            if spec['targets'] != specs:
                raise ValueError('Production scope changed; reconcile before resuming')
        else:
            order = [n for n in PRIORITY if n in specs]
            order += [n for n in specs if n not in order]
            spec = {'targets': specs, 'order': order, 'snapshot_end_exclusive': int(time.time()),
                    'download_keyword_filter': False, 'export_retention': 'latest generation; all records remain in SQLite'}
            write_json(pinned, spec)
        state_path = directory/'production-state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'cursor': 0, 'not_before': 0, 'rounds': 0, 'communities': {}, 'requests_total': 0}
        state['not_before'] = normalize_reset_timestamp(state.get('not_before', 0))
        state['snapshot_end_exclusive_utc'] = utc(spec['snapshot_end_exclusive'])
        state['targets_total'] = len(specs)
        if state['not_before'] > time.time():
            return dict(state, status='cooldown')
        cooldown_path = directory/'source-cooldown.json'
        if cooldown_path.exists():
            state['not_before'] = max(state['not_before'], json.loads(cooldown_path.read_text())['not_before'])
            if state['not_before'] > time.time():
                write_json(state_path, state)
                return dict(state, status='cooldown')
        policy = request_policy(directory)
        def persist_cooldown(until):
            write_json(cooldown_path, {'not_before': until, 'updated_at': utc(time.time())})
            if policy['interval_seconds'] < 1 or policy['workers'] > 1:
                write_json(directory/'request-policy.json', dict(
                    policy, interval_seconds=1.0, workers=1,
                    reason='automatic fallback after source cooldown', updated_at=utc(time.time())))
        gate = RequestGate(max_requests, policy['interval_seconds'], persist_cooldown)
        clients = []
        started = time.monotonic()
        initial_requests = state['requests_total']
        state.update(status='running', round_started_at=utc(time.time()), requests_this_round=0,
                     request_interval_seconds=policy['interval_seconds'], workers=policy['workers'])
        state['rounds'] += 1
        write_json(state_path, state)
        visited = 0
        executor = ThreadPoolExecutor(max_workers=policy['workers']) if policy['workers'] > 1 else None
        batch = []
        try:
            while visited < len(spec['order']) and gate.calls < max_requests:
                if shutil.disk_usage(directory).free < reserve_gib*2**30:
                    state['status'] = 'disk_reserve_reached'
                    break
                batch = []
                while visited < len(spec['order']) and len(batch) < policy['workers']:
                    index = state['cursor']
                    name = spec['order'][index]
                    state['cursor'] = (index+1) % len(spec['order'])
                    visited += 1
                    previous = state['communities'].get(name, {})
                    if previous.get('source_done') and source_done(previous):
                        continue
                    client = DiskGuardClient(directory, int(reserve_gib*2**30), budget=per_community,
                                             interval=0)
                    client.before_request, client.on_rate_limit = gate.acquire, gate.cooldown
                    if policy['keep_alive']:
                        client.opener = KeepAliveTransport()
                    clients.append(client)
                    batch.append((index, name, client))
                if not batch:
                    break
                state['active_subreddits'] = [name for _, name, _ in batch]
                state['active_subreddit'] = batch[0][1]
                write_json(state_path, state)
                futures = []
                def collect_one(name, client):
                    return _collect_community(directory/name, specs[name], spec['snapshot_end_exclusive'],
                                              client, per_community)
                if executor:
                    futures = [executor.submit(collect_one, name, client) for _, name, client in batch]
                try:
                    for position, (index, name, client) in enumerate(batch):
                        result = futures[position].result() if executor else collect_one(name, client)
                        state['communities'][name] = result
                except KeyboardInterrupt:
                    gate.stop()
                    state['cursor'] = batch[0][0]
                    # In-flight requests may commit; every thread closes its own
                    # SQLite connection before releasing the production lock.
                    if executor:
                        for (_, name, _), future in zip(batch, futures):
                            try:
                                state['communities'][name] = future.result()
                            except KeyboardInterrupt:
                                state['communities'].setdefault(name, {}).update(source_done=False)
                    raise
                finally:
                    state['not_before'] = max(state['not_before'], gate.not_before)
                    if gate.not_before > time.time():
                        state['cursor'] = batch[0][0]
                        state['status'] = 'cooldown'
                    state['requests_total'] = initial_requests + sum(c.calls for c in clients)
                    state['requests_this_round'] = sum(c.calls for c in clients)
                    state['rows_collected'] = sum(sum(v['count'] for v in r.get('counts', {}).values()) for r in state['communities'].values())
                    state['communities_verified'] = sum(r.get('source_done', False) for r in state['communities'].values())
                    state['updated_at'] = utc(time.time())
                    write_json(state_path, state)
                if state['not_before'] > time.time():
                    break
        finally:
            gate.stop()
            if executor:
                executor.shutdown(wait=True)
            for client in clients:
                if client.opener:
                    client.opener.close()
            if (policy['interval_seconds'] < 1 or policy['workers'] > 1) and state.get('not_before', 0) > time.time():
                state['request_policy_fallback'] = 'source_cooldown'
            state['round_metrics'] = {
                'elapsed_seconds': round(time.monotonic()-started, 3),
                'requests': sum(c.calls for c in clients),
                'returned_rows': sum(r.get('rows', 0) for c in clients for r in c.history),
                'http_errors': sum(r.get('http_status', 200) >= 400 for c in clients for r in c.history),
            }
            if state['status'] == 'running':
                state['status'] = 'source_reconciled' if state.get('communities_verified', 0) == len(specs) else 'partial'
            state['active_subreddit'] = None
            state['active_subreddits'] = []
            state['round_finished_at'] = utc(time.time())
            state['disk_free_gib'] = round(shutil.disk_usage(directory).free/2**30, 2)
            write_json(state_path, state)
        return state
    finally:
        lock.close()


def production_status(output_dir, report_path=None):
    """Read live checkpoint counts without acquiring a writer lock or exposing text."""
    directory = Path(output_dir).expanduser().resolve()
    spec = json.loads((directory/'production-spec.json').read_text())
    state = json.loads((directory/'production-state.json').read_text())
    counts, issues = {}, {}
    for name in spec['targets']:
        checkpoint = directory/name/'checkpoint.sqlite'
        if not checkpoint.exists():
            continue
        try:
            db = sqlite3.connect(checkpoint.as_uri()+'?mode=ro', uri=True, timeout=2)
            try:
                counts[name] = dict(db.execute('SELECT kind,count(*) FROM items GROUP BY kind'))
            finally:
                db.close()
        except sqlite3.Error as error:
            issues[name] = type(error).__name__
    occupied = False
    lock = (directory/'.production.lock').open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            occupied = True
    finally:
        lock.close()
    report = {'generated_at': utc(time.time()), 'production_directory': str(directory),
              'state': state, 'writer_active': occupied,
              'live_database_counts': counts,
              'live_rows_downloaded': sum(sum(v.values()) for v in counts.values()),
              'checkpoint_read_errors': issues,
              'note': 'Formal run only. Live counts can exceed per-community summaries during a round. Source completion still requires reconciliation.'}
    with (directory/'.worker.lock').open('a') as worker_lock:
        try:
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report['worker_active'] = False
        except BlockingIOError:
            report['worker_active'] = True
    worker_state = directory/'worker-state.json'
    report['worker_state'] = json.loads(worker_state.read_text()) if worker_state.exists() else None
    if report_path:
        target = Path(report_path).expanduser().resolve()
        if target == directory or directory in target.parents:
            raise ValueError('Write progress reports outside the raw checkpoint directory')
        # A status export is a metadata report, never a replacement for a database.
        if target.suffix != '.json':
            raise ValueError('Status report must be JSON')
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, report)
    return report
