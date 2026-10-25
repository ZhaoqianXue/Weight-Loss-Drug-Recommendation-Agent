"""Input manifest for the full Jev run: every s10 shard, its record count and a sha256 of its sorted record ids.
Reads only the fullname column (runs in .venv-reddit-clean, which has polars). Offline; no API calls."""
import hashlib, json, datetime
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / '.cache/reddit-cleaning/reddit-strict-1m-s10-2026-09-26/records'
shards, seen, total = [], set(), 0
for p in sorted(EXPORT.glob('part-*.parquet')):
    ids = pl.read_parquet(p, columns=['fullname'])['fullname'].to_list()
    rids = ['reddit:' + x for x in ids]
    assert len(rids) == len(set(rids)), p
    dup = seen.intersection(rids); assert not dup, (p, list(dup)[:3])
    seen.update(rids); total += len(rids)
    shards.append({'shard': p.name, 'records': len(rids), 'ids_sha256': hashlib.sha256('\n'.join(sorted(rids)).encode()).hexdigest(), 'bytes': p.stat().st_size})
assert total == len(seen) == 1_030_309, total
all_sha = hashlib.sha256('\n'.join(sorted(seen)).encode()).hexdigest()
man = {'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'export': str(EXPORT.relative_to(ROOT)),
       'records': total, 'unique_ids': len(seen), 'all_ids_sha256': all_sha, 'shards': shards, 'text_field': 'text (exact downstream extraction input, full, not truncated)'}
(ROOT / 'results/jev-full-2026-09-29/input-manifest.json').write_text(json.dumps(man, indent=1) + '\n')
print(json.dumps({k: man[k] for k in ('records', 'unique_ids', 'all_ids_sha256')} | {'shards': len(shards)}, indent=1))
