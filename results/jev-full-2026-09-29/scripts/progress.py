"""Hourly progress check for the full Jev run (offline; no API calls, no key). Appends one line to hourly-progress.jsonl
and one row to hourly-progress.md: phase, records done/left, keep rate, errors, retries, 429s, cost, rate, ETA, worker
liveness and stall detection. Exit code 0 = running normally, 3 = finished or paused (monitoring can stop), 2 = problem."""
import datetime, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RES, CACHE = ROOT / 'results/jev-full-2026-09-29', ROOT / '.cache/jev-full-2026-09-29'
now = datetime.datetime.now(datetime.timezone.utc)
st = json.loads((RES / 'run-status.json').read_text()) if (RES / 'run-status.json').exists() else {}
def alive(pid):
    try:
        os.kill(int(pid), 0); return True
    except Exception:
        return False
pid = st.get('pid')
worker_alive = bool(pid) and alive(pid)
upd = st.get('updated_utc')
age_min = round((now - datetime.datetime.strptime(upd, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)).total_seconds() / 60, 1) if upd else None
prev = None
hp = RES / 'hourly-progress.jsonl'
if hp.exists():
    lines = [x for x in hp.read_text().splitlines() if x.strip()]
    prev = json.loads(lines[-1]) if lines else None
errs = sum(len(p.read_text().splitlines()) for p in (CACHE / 'errors').glob('*.jsonl')) if (CACHE / 'errors').exists() else 0
row = {'checked_utc': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'phase': st.get('phase'), 'worker_pid': pid, 'worker_alive': worker_alive,
       'status_age_min': age_min, 'records_done': st.get('records_done'), 'records_left': st.get('records_left'), 'keep': st.get('keep'),
       'keep_rate': st.get('keep_rate'), 'error_lines_total': errs, 'retries': st.get('retries'), 'http_429': st.get('http_429'),
       'usd_known': st.get('usd_known'), 'usd_unknown_reserve': st.get('usd_unknown_reserve'), 'usd_cap': st.get('usd_cap'),
       'request_rpm_last_5min': st.get('request_rpm_last_5min'), 'target_rpm_now': st.get('target_rpm_now'), 'eta_hours': st.get('eta_hours'),
       'shards_done': st.get('shards_done'), 'pause_reason': st.get('pause_reason')}
problems = []
if st.get('phase') == 'running' and not worker_alive:
    problems.append('status says running but the worker process is gone')
if st.get('phase') == 'running' and age_min is not None and age_min > 10:
    problems.append(f'status not updated for {age_min} min')
if prev and prev.get('records_done') == row['records_done'] and st.get('phase') == 'running':
    problems.append('no progress since the previous check')
row['problems'] = problems
with hp.open('a') as f:
    f.write(json.dumps(row) + '\n')
md = RES / 'hourly-progress.md'
if not md.exists():
    md.write_text('| checked (UTC) | phase | done | left | keep rate | errors | retries | 429 | USD | RPM | ETA h | problems |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n')
with md.open('a') as f:
    f.write(f"| {row['checked_utc']} | {row['phase']} | {row['records_done']} | {row['records_left']} | {row['keep_rate']} | {errs} | {row['retries']} | "
            f"{row['http_429']} | {row['usd_known']} | {row['request_rpm_last_5min']} | {row['eta_hours']} | {'; '.join(problems) or '-'} |\n")
print(json.dumps(row))
sys.exit(3 if st.get('phase') in ('completed', 'paused') or (st.get('phase', '').startswith('incomplete')) else (2 if problems else 0))
