"""Final reconciliation of the full Jev run: reads the private results/attempts/errors, writes final-reconciliation.json.

Read-only over .cache/jev-full-2026-09-29 (no API calls). Residual errors are manifest records without a result line;
error lines are history, not decisions. Costs: usd = billed input tokens of successful attempts; unknown_usd = per-request
upper-bound reserve for attempts whose billing is unknown (failed requests), kept separately and never netted.
"""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / 'results/jev-full-2026-09-29'
CACHE = ROOT / '.cache/jev-full-2026-09-29'
RETRY_SESSION_START = '2026-09-29T22:31:59Z'  # launch of the bounded retry of the 31 HTTP 520 failures
THRESHOLD = 0.5

man = json.loads((RES / 'input-manifest.json').read_text())
ids, dup, invalid, rule_mismatch, ties, rebuilt = set(), 0, 0, 0, 0, 0
keep = drop = 0
missing_by_shard = {}
for sh in man['shards']:
    name, got = sh['shard'], set()
    p = CACHE / 'results' / (name + '.jsonl')
    for x in (p.read_text().splitlines() if p.exists() else []):
        if not x.strip():
            continue
        r = json.loads(x)
        rid, pk = r['record_id'], r.get('p_keep')
        if rid in ids or rid in got:
            dup += 1
            continue
        got.add(rid)
        if r.get('decision') not in ('keep', 'drop') or not isinstance(pk, (int, float)) or not 0.0 <= pk <= 1.0:
            invalid += 1
            continue
        rule_mismatch += r['decision'] != ('keep' if pk >= THRESHOLD else 'drop')
        ties += pk == THRESHOLD
        rebuilt += bool(r.get('rebuilt_from_attempt'))
        keep += r['decision'] == 'keep'
        drop += r['decision'] == 'drop'
    if len(got) != sh['records']:
        missing_by_shard[name] = sh['records'] - len(got)
    ids |= got

att = collections.Counter()
usd = unknown = 0.0
usd_retry = unknown_retry = 0.0
fail_codes = collections.Counter()
for p in sorted((CACHE / 'attempts').glob('*.jsonl')):
    for x in p.read_text().splitlines():
        if not x.strip():
            continue
        a = json.loads(x)
        retry = a['at'] >= RETRY_SESSION_START
        att['attempts'] += 1
        att['retry_session_attempts'] += retry
        att['later_attempts_same_session'] += a.get('attempt', 1) > 1
        if a.get('error') is None:
            att['successful_attempts'] += 1
        else:
            att['failed_attempts'] += 1
            fail_codes[str(a.get('status_code'))] += 1
        usd += a.get('usd') or 0.0
        unknown += a.get('unknown_usd') or 0.0
        if retry:
            usd_retry += a.get('usd') or 0.0
            unknown_retry += a.get('unknown_usd') or 0.0

complete = not missing_by_shard and len(ids) == man['records']
out = {
    'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'config': {'model': 'jev-1.13.0', 'prompt_sha256': '4f77a72aa5e91e2dd787f744b1ebd7046b8dbb87129f7cbe5c678bd70719c1a4',
               'rule': 'keep iff p_keep >= 0.5', 'target_rpm': 1150, 'concurrency': 8, 'packing': False},
    'records_total': man['records'],
    'records_with_result': len(ids),
    'residual_errors': man['records'] - len(ids),
    'residual_by_shard': missing_by_shard,
    'all_ids_sha256_match': complete and hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest() == man['all_ids_sha256'],
    'keep': keep, 'drop': drop, 'keep_rate': round(keep / len(ids), 4) if ids else None,
    'duplicate_result_lines': dup, 'invalid_result_lines': invalid, 'decision_rule_mismatches': rule_mismatch,
    'p_keep_exactly_0_5_kept_by_rule': ties, 'results_rebuilt_from_attempts': rebuilt,
    'attempts': dict(att), 'failed_attempt_status_codes': dict(fail_codes),
    'cost_usd': {'known_billed_total': round(usd, 6), 'unknown_reserve_total': round(unknown, 6),
                 'known_plus_reserve_upper_bound': round(usd + unknown, 6),
                 'retry_session_known': round(usd_retry, 6), 'retry_session_unknown_reserve': round(unknown_retry, 6),
                 'cap': 75.0},
    'history': {'main_run': 'launch 2026-09-29T02:12:08Z, end 17:33:07Z, 1,030,278 results, 31 records failed with HTTP 520 '
                            '(Cloudflare, not in the runner retry set, so one attempt each), usd_known 41.521078, unknown reserve 0.005521',
                'retry_run': 'same frozen runner and config relaunched with start.sh at ' + RETRY_SESSION_START + '; only records without results were sent'},
    'status': 'completed' if complete else 'incomplete',
}
(RES / 'final-reconciliation.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
