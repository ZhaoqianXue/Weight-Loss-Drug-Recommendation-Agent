"""Draw private, reproducible QA samples from the retained x=10 candidates.

R200: equal-probability random retained records (fixed seed), excluding an external
acceptance sample if given. Strata: seeded draws from each candidate-rule hit set.
Texts stay under .cache (private); only IDs and aggregate labels leave it.
"""
import argparse
import json
from pathlib import Path

import polars as pl

RUN = Path('.cache/reddit-cleaning/denoise-x10-2026-09-25')
QA = Path('.cache/reddit-cleaning/denoise-x10-qa-2026-09-25')


def strata(f):
    ratio = pl.col('unique_tokens') / pl.col('tokens')
    thread = pl.when(pl.col('kind') == 'posts').then(pl.col('fullname')).otherwise(pl.col('context_thread'))
    dup = (f.sort('skeleton', 'created_utc', 'rowid')
           .with_columns(pos=pl.int_range(pl.len()).over('skeleton'), copies=pl.len().over('skeleton')))
    masked = f.with_columns(copies=pl.len().over('masked_key', 'masked_key2'), threads=thread.n_unique().over('masked_key', 'masked_key2'))
    return {
        'S1-exact-dup-later-copy': dup.filter(pl.col('pos') > 0),
        'S2-masked-template-20plus': masked.filter(pl.col('copies') >= 20),
        'S3-url-share-50pct': f.filter(pl.col('url_chars') / pl.col('chars') >= 0.5),
        'S4-token-repetition': f.filter(ratio < 0.3),
        'S4-char-run': f.filter(pl.col('char_run')),
        'S4-alpha-lt-30pct': f.filter(pl.col('letters') / pl.col('visible') < 0.3),
        'S5-non-english-10-19': f.filter((pl.col('lang') != 'en') & (pl.col('lang_score') >= 0.95)),
        'S5-non-latin-script': f.filter(pl.col('latin') / pl.col('letters').clip(1, None) < 0.5),
        'S6-bot-mod-words': f.filter(pl.col('bot_probe')),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--external', type=Path, default=Path('/tmp/reddit-retained-qa-20260925.jsonl'))
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--per-stratum', type=int, default=20)
    args = ap.parse_args()
    f = pl.read_parquet(str(QA / 'features' / '*.parquet'))
    external = set()
    if args.external.exists():
        external = {json.loads(l)['fullname'] for l in args.external.read_text().splitlines()}
    pool = f.filter(~pl.col('fullname').is_in(list(external)))
    picks = [pool.sample(200, seed=args.seed).with_columns(stratum=pl.lit('R200'))]
    for i, (name, hits) in enumerate(strata(f).items()):
        picks.append(hits.sample(min(args.per_stratum, hits.height), seed=args.seed + i + 1)
                     .select(f.columns).with_columns(stratum=pl.lit(name)))
    sample = pl.concat(picks)
    if external:
        ext = f.filter(pl.col('fullname').is_in(list(external))).with_columns(stratum=pl.lit('EXT104'))
        sample = pl.concat([sample, ext])
    ids = sample.select('rowid').unique()
    text = (pl.scan_parquet(str(RUN / 'candidates' / '*.parquet')).join(ids.lazy(), on='rowid')
            .select('rowid', 'text', 'quoted_text', 'title').collect())
    parents = (pl.scan_parquet(str(RUN / 'records' / '*.parquet')).select('fullname', 'text')
               .join(sample.select(pl.col('context_parent').alias('fullname')).unique().lazy(), on='fullname')
               .rename({'fullname': 'context_parent', 'text': 'parent_text'}).collect())
    out = (sample.join(text, on='rowid', how='left').join(parents, on='context_parent', how='left')
           .with_columns(pl.col('parent_text').str.slice(0, 300), pl.col('text').str.slice(0, 700))
           .select('stratum', 'rowid', 'fullname', 'kind', 'subreddit', 'lexical_words', 'text', 'quoted_text', 'parent_text',
                   'lang', 'lang_score'))
    (QA / 'review').mkdir(exist_ok=True)
    path = QA / 'review' / 'samples.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in out.to_dicts()))
    counts = {k: v for k, v in out.group_by('stratum').len().iter_rows()}
    hit_sizes = {k: v.height for k, v in strata(f).items()}
    (QA / 'review' / 'strata.json').write_text(json.dumps(dict(sample_counts=counts, hit_population=hit_sizes,
                                                               seed=args.seed, external=str(args.external)), indent=2))
    print(json.dumps(dict(sample_counts=counts, hit_population=hit_sizes), indent=2))


if __name__ == '__main__':
    main()
