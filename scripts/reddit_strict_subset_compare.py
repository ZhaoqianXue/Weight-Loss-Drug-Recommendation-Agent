"""Compare ~1M-record strict-subset selections on the a2 kept set (all selections are lossy; nothing here is denoising).

step 'compare': thresholds nearest to the target for each family, kept records/characters, overlaps, round-5 label retention
                -> results/.../strict-subset-comparison.{json,md}
step 'examples FAMILY_A FAMILY_B': private text samples kept by one selection and dropped by the other (scratch only)
"""
import argparse
import json
import sys
from pathlib import Path

import polars as pl

FEAT = Path('.cache/reddit-cleaning/strict-features-a2-2026-09-26/features')
BASE = Path('.cache/reddit-cleaning/denoise-x10-2026-09-25/candidates')
ROUNDS = Path('.cache/reddit-cleaning/denoise-iterations-2026-09-25')
RESULTS = Path('results/reddit-denoise-iterations-2026-09-25')
TARGET = 1_000_000


def load():
    return pl.scan_parquet(str(FEAT / '*.parquet')).collect()


def families(f):
    """name -> (score expression, gate expression or None); a record is kept when gate passes and score >= threshold."""
    rep = (pl.col('content_n') >= 30) & (pl.col('top_share') > 0.2)
    borrowed = (pl.col('content_n') >= 20) & (pl.col('original_n') < 0.5 * pl.col('content_n'))
    return {
        'A_words': (pl.col('n_tok'), None),
        'B_content_words': (pl.col('content_n'), None),
        'C_content_types': (pl.col('content_types'), None),
        'D_original_words': (pl.col('original_n'), None),
        'E_original_types': (pl.col('original_types'), None),
        'F_original_types_gated': (pl.col('original_types'), ~rep & ~borrowed),
    }


def nearest(f, score, gate):
    s = f.select(v=score, g=gate if gate is not None else pl.lit(True))
    counts = (s.filter(pl.col('g')).group_by('v').len().sort('v', descending=True)
              .with_columns(kept=pl.col('len').cum_sum()))
    best = counts.with_columns(d=(pl.col('kept') - TARGET).abs()).sort('d').row(0, named=True)
    return int(best['v']), int(best['kept'])


def kept_mask(f, score, gate, t):
    m = score >= t
    return (m & gate) if gate is not None else m


def round5_labels(f):
    rows = [json.loads(line) for p in sorted((ROUNDS / 'round-05').glob('labels-*.jsonl')) for line in open(p)]
    lab = pl.DataFrame({'rowid': [r['rowid'] for r in rows], 'label': [r['label'] for r in rows]}).with_columns(pl.col('rowid').cast(pl.Int64))
    return lab.join(f.select(pl.col('rowid').cast(pl.Int64)), on='rowid', how='semi')


def compare():
    f = load()
    fam = families(f)
    total, chars = f.height, int(f['chars'].sum())
    out = dict(pool=dict(version='denoise-x20-a2-2026-09-26', records=total, characters=chars), target=TARGET, families={})
    masks = {}
    lab = round5_labels(f)
    for name, (score, gate) in fam.items():
        t, _ = nearest(f, score, gate)
        m = f.select(k=kept_mask(f, score, gate, t))['k']
        masks[name] = m
        k = f.filter(m)
        lk = lab.join(f.filter(m).select(pl.col('rowid').cast(pl.Int64)), on='rowid', how='semi')
        by = {lb: [int(lk.filter(pl.col('label') == lb).height), int(lab.filter(pl.col('label') == lb).height)] for lb in ('N', 'S', 'J')}
        out['families'][name] = dict(
            threshold=t, gated=gate is not None, kept=k.height, kept_pct=round(100 * k.height / total, 2),
            kept_characters=int(k['chars'].sum()), char_pct=round(100 * int(k['chars'].sum()) / chars, 1),
            token_proxy_millions=round(int(k['chars'].sum()) / 4 / 1e6, 1),
            posts=int(k.filter(pl.col('kind') == 'posts').height), comments=int(k.filter(pl.col('kind') == 'comments').height),
            median_words=int(k['n_tok'].median()), median_original_types=int(k['original_types'].median()),
            round5_kept_of_labeled=by)
    names = list(masks)
    out['jaccard_vs_E'] = {n: round(float((masks[n] & masks['E_original_types']).sum() / (masks[n] | masks['E_original_types']).sum()), 3)
                           for n in names}
    out['jaccard_vs_F'] = {n: round(float((masks[n] & masks['F_original_types_gated']).sum()
                                          / (masks[n] | masks['F_original_types_gated']).sum()), 3) for n in names}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'strict-subset-comparison.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1))


def examples(a, b, n=12, seed=2026092610):
    """Records kept by selection a but not b (and a few margin records just below a's threshold); scratch output only."""
    f = load()
    fam = families(f)
    sel = {}
    for name in (a, b):
        score, gate = fam[name]
        t, _ = nearest(f, score, gate)
        sel[name] = (t, f.select(k=kept_mask(f, score, gate, t))['k'])
    only_a = f.filter(sel[a][1] & ~sel[b][1]).sample(n, seed=seed)
    only_b = f.filter(sel[b][1] & ~sel[a][1]).sample(n, seed=seed)
    score, gate = fam[a]
    margin = f.filter(f.select(v=score)['v'].is_between(sel[a][0] - 3, sel[a][0] - 1)).sample(n, seed=seed)
    for title, part in ((f'kept by {a} only', only_a), (f'kept by {b} only', only_b), (f'just below {a} threshold {sel[a][0]}', margin)):
        ids = part['rowid'].to_list()
        text = {r['rowid']: r for r in pl.scan_parquet(str(BASE / '*.parquet')).select('rowid', 'fullname', 'subreddit', 'text')
                .filter(pl.col('rowid').is_in(ids)).collect().iter_rows(named=True)}
        print(f'===== {title} ({part.height})')
        for r in part.iter_rows(named=True):
            t = text[r['rowid']]
            print(f"[{r['rowid']} {t['fullname']} r/{t['subreddit']} {r['kind']} words={r['n_tok']} content={r['content_n']} "
                  f"types={r['content_types']} orig={r['original_n']} orig_types={r['original_types']} top={r['top_share']:.2f}]")
            print('  ' + t['text'][:1500].replace('\n', ' ⏎ ') + (' …' if len(t['text']) > 1500 else ''))
        print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=['compare', 'examples'])
    p.add_argument('names', nargs='*')
    args = p.parse_args()
    if args.step == 'compare':
        compare()
    else:
        examples(*args.names)
    sys.exit(0)
