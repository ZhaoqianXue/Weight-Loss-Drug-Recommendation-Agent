"""Extract one s10 shard to a temporary JSONL (record_id, text) for the Jev worker (runs in .venv-reddit-clean)."""
import json, sys
import polars as pl
src, dst = sys.argv[1], sys.argv[2]
df = pl.read_parquet(src, columns=['fullname', 'text'])
with open(dst, 'w') as f:
    for fn, tx in zip(df['fullname'].to_list(), df['text'].to_list()):
        f.write(json.dumps({'record_id': 'reddit:' + fn, 'text': tx}, ensure_ascii=False) + '\n')
print(len(df))
