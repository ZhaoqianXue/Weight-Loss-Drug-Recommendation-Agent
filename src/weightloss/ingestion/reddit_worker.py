"""Single local worker: continue bounded production rounds between agent inspections."""
import fcntl
import os
from pathlib import Path
import time

from .reddit import utc
from .reddit_production import collect_production, production_status, write_json


def run_worker(output_dir, report_file, max_requests=1200, per_community=200,
               reserve_gib=15, max_rounds=None):
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if max_rounds is not None and max_rounds < 1:
        raise ValueError('max_rounds must be positive')
    lock = (directory/'.worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {'status': 'already_running'}
    state = {'pid': os.getpid(), 'started_at': utc(time.time()),
             'rounds_completed': 0, 'status': 'starting'}
    stop_file = directory/'.worker-stop'

    def save(status, **values):
        state.update(status=status, updated_at=utc(time.time()), **values)
        write_json(directory/'worker-state.json', state)

    def wait_until(deadline):
        while time.time() < deadline:
            if stop_file.exists():
                return False
            save('waiting', not_before=deadline)
            time.sleep(min(60, max(0, deadline-time.time())))
        return not stop_file.exists()

    try:
        while max_rounds is None or state['rounds_completed'] < max_rounds:
            if stop_file.exists():
                save('stopped')
                break
            save('collecting', not_before=0)
            result = collect_production(directory, max_requests, per_community, reserve_gib)
            status = result['status']
            if status == 'already_running':
                save('other_collector_running')
                break
            production_status(directory, report_file)
            if status != 'cooldown':
                state['rounds_completed'] += 1
            save('round_finished', production_status=status,
                 rows_collected=result.get('rows_collected', 0),
                 communities_verified=result.get('communities_verified', 0))
            if status in ('source_reconciled', 'disk_reserve_reached'):
                save(status)
                break
            if max_rounds is not None and state['rounds_completed'] >= max_rounds:
                save('round_budget_reached')
                break
            # Cooling down is a source-wide deadline, never a reason to change
            # pages, credentials or download scope. Poll the stop file each minute.
            delay = 10 if result.get('requests_this_round', 0) else 120
            if any(c.get('error') for c in result.get('communities', {}).values()):
                delay = max(delay, 120)
            deadline = max(time.time()+delay, result.get('not_before', 0))
            if not wait_until(deadline):
                save('stopped')
                break
    except KeyboardInterrupt:
        save('stopped')
    except Exception as error:
        # Surface unexpected configuration or filesystem failures to the agent;
        # do not conceal them in an endless restart loop.
        save('error', error_type=type(error).__name__, error=str(error)[:300])
        raise
    finally:
        lock.close()
    return state
