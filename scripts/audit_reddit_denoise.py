"""Summarize a finished reddit_denoise pipeline run from its Parquet outputs.

Reads only the run directory (never the 33 GB source): rule contributions, the
x-floor effect, Jev request/token planning estimates, runtime/memory metrics,
and an optional row-by-row equivalence check against the earlier two-pass
overlay (SQLite, x=0). Token counts use chars/4; they are estimates, not usage.
"""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import time

import polars as pl

NEW_RULES = ['moderation_or_bot_template', 'bot_command_only', 'media_without_text', 'social_phrase_combination',
             'duplicate_long_text', 'duplicate_quick_repeat']
READY = 'ready_for_semantic_screen'
JEV_USD_PER_MILLION_INPUT = 0.042   # docs.typesafe.ai/models, checked 2026-09-25
JEV_REQUESTS_PER_MINUTE = 1200
CONTEXT_CAP = 2000                  # v2 context excerpt cap


def layer(reason):
    if reason == READY:
        return 'candidate'
    if reason == 'below_min_lexical_words':
        return 'length_floor'
    return 'new_denoise_coverage_1' if reason in NEW_RULES else 'inherited_literature_noise_1'


def equivalence(decisions, old_sqlite, floor):
    """New x-floor reasons must equal the old x=0 overlay except READY->below_min for short rows."""
    started = time.perf_counter()
    db = sqlite3.connect(f'file:{Path(old_sqlite).resolve()}?mode=ro', uri=True)
    old = pl.DataFrame(db.execute('SELECT source_rowid, reason FROM decisions').fetchall(),
                       schema=['rowid', 'old_reason'], orient='row')
    joined = decisions.select('rowid', 'reason', 'lexical_words').collect().join(old, on='rowid', how='full', coalesce=True)
    expected = pl.when((pl.col('old_reason') == READY) & (pl.col('lexical_words') < floor)).then(
        pl.lit('below_min_lexical_words')).otherwise(pl.col('old_reason'))
    mismatches = joined.filter(pl.col('reason').is_null() | pl.col('old_reason').is_null() | (pl.col('reason') != expected))
    return dict(rows_compared=joined.height, mismatches=mismatches.height,
                examples=mismatches.head(5).to_dicts(), seconds=time.perf_counter() - started)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=Path('.cache/reddit-cleaning/denoise-x10-2026-09-25'))
    ap.add_argument('--old-overlay', type=Path, default=Path('.cache/reddit-cleaning/denoise-full-2026-09-25/decisions.sqlite'))
    ap.add_argument('--question-tokens', type=int, default=300)
    ap.add_argument('--output', type=Path, default=Path('results/reddit-denoise-coverage-2026-09-25'))
    args = ap.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['status'] != 'full_denoise_complete':
        raise ValueError('Pipeline run incomplete')
    floor = manifest['min_lexical_words']
    decisions = pl.scan_parquet(str(args.run / 'decisions' / '*.parquet'))
    total = manifest['source_records']
    reasons = manifest['reasons']
    args.output.mkdir(parents=True, exist_ok=True)
    rows = sorted(reasons.items(), key=lambda kv: -kv[1])
    with (args.output / 'rule-contributions.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reason', 'layer', 'records', 'pct_of_corpus'])
        for reason, n in rows:
            writer.writerow([reason, layer(reason), n, round(100 * n / total, 4)])
    by_layer = {}
    for reason, n in rows:
        by_layer[layer(reason)] = by_layer.get(layer(reason), 0) + n
    # Old rules on the same lexical definition: literature-noise-1 candidates = everything the
    # new record rules, duplicates and floor excluded, plus final candidates.
    lexical = (decisions.filter(pl.col('reason').is_in([READY, 'below_min_lexical_words'] + NEW_RULES))
               .group_by(pl.col('lexical_words').clip(0, 101)).len().collect())
    old_counts = dict(lexical.iter_rows())
    new_counts = {int(k): v for k, v in manifest['candidates_by_min_lexical_words'].items()}
    thresholds = [0, 3, 5, 10, 15, 20, 30, 50]
    curve = [dict(min_lexical_words=x,
                  literature_noise_1_candidates=sum(n for w, n in old_counts.items() if w >= x),
                  denoise_coverage_1_candidates=new_counts[x],
                  denoise_coverage_1_retention_pct=100 * new_counts[x] / total) for x in range(0, 101)]
    with (args.output / 'word-threshold-curve.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    # Jev planning estimate for the final candidates, with real parent and thread-title sizes.
    records = pl.scan_parquet(str(args.run / 'records' / '*.parquet'))
    parents = records.select(pl.col('fullname').alias('context_parent'), pl.col('own_characters').alias('parent_chars'))
    titles = records.select(pl.col('fullname').alias('context_thread'), pl.col('title').str.len_chars().alias('title_chars'))
    sizes = (decisions.filter(pl.col('decision') == 'candidate').select('context_parent', 'context_thread', 'own_characters')
             .join(parents, on='context_parent', how='left').join(titles, on='context_thread', how='left')
             .select(pl.len().alias('requests'), pl.col('own_characters').cast(pl.Int64).sum().alias('own'),
                     pl.col('parent_chars').fill_null(0).clip(0, CONTEXT_CAP).cast(pl.Int64).sum().alias('parent'),
                     pl.col('title_chars').fill_null(0).cast(pl.Int64).sum().alias('title')).collect().to_dicts()[0])
    requests = sizes['requests']
    own_only = sizes['own'] / 4 + args.question_tokens * requests
    with_context = (sizes['own'] + sizes['parent'] + sizes['title']) / 4 + args.question_tokens * requests
    cost = dict(requests=requests, own_characters=sizes['own'], parent_characters_capped=sizes['parent'],
                title_characters=sizes['title'], question_tokens_per_request=args.question_tokens, tokens_per_char=0.25,
                own_text_tokens=own_only, own_text_usd=own_only * JEV_USD_PER_MILLION_INPUT / 1e6,
                with_parent_and_title_tokens=with_context,
                with_parent_and_title_usd=with_context * JEV_USD_PER_MILLION_INPUT / 1e6,
                hours_at_request_limit=requests / JEV_REQUESTS_PER_MINUTE / 60,
                note='Planning estimate: one Jev request per candidate; state billed once per request.')
    result = dict(schema_version=2, policy_revision=manifest['policy_revision'], min_lexical_words=floor,
                  source_database_sha256=manifest['source_database_sha256'], source_records=total,
                  candidates=manifest['candidates'], excluded=manifest['excluded'], retention_pct=manifest['retention_pct'],
                  reasons=reasons, by_layer=by_layer, by_kind=manifest['by_kind'],
                  selected_thresholds=[curve[x] for x in thresholds], jev_estimate=cost, metrics=manifest['metrics'],
                  equivalence_with_two_pass_x0=equivalence(decisions, args.old_overlay, floor) if args.old_overlay.exists() else None)
    (args.output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('candidates', 'excluded', 'retention_pct', 'by_layer', 'jev_estimate',
                                              'equivalence_with_two_pass_x0')}, indent=2, default=str))


if __name__ == '__main__':
    main()
