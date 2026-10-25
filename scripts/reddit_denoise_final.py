"""Final aggregates for the frozen basic-denoise version; no text leaves .cache.

summary:  reads the frozen refine version, same-rule comparison versions at other floors, the
          r5 -> r6 paragraph-containment fix and the round status, then writes
          results/reddit-denoise-iterations-2026-09-25/final-summary.json and CSV tables.
examples: writes a seeded private sample of removed records per reason, with text and the
          kept representative or host, to the private iterations directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weightloss.cleaning import reddit_denoise_refine as refine  # noqa: E402

CACHE = Path('.cache/reddit-cleaning')
BASE = CACHE / 'denoise-x10-2026-09-25'
FROZEN = CACHE / 'denoise-x20-r6-2026-09-25'
PREVIOUS = CACHE / 'denoise-x20-r5-2026-09-25'
COMPARE = {x: CACHE / f'denoise-x{x}-r6-compare-2026-09-25' for x in (30, 40, 50)}
ITER = CACHE / 'denoise-iterations-2026-09-25'
RESULTS = Path('results/reddit-denoise-iterations-2026-09-25')
REFINE_REASONS = ('mechanical_template', 'social_template', 'non_latin_script_scope', 'token_repetition_spam',
                  'duplicate_body_text', 'contained_paragraph_copy')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version_row(path, base_candidates):
    m = json.loads((path / 'manifest.json').read_text())
    floor_only = base_candidates - m['standalone_rule_hits_on_base_candidates']['hit_floor']
    net = m['net_beyond_floor']
    return dict(min_lexical_words=m['min_lexical_words'], policy_revision=m['policy_revision'], version_dir=str(path),
                manifest_sha256=digest(path / 'manifest.json'), floor_only_retained=floor_only,
                final_rules_retained=m['candidates'], removed_beyond_floor=floor_only - m['candidates'],
                characters=m['candidate_characters'], token_proxy=m['candidate_token_proxy'],
                net_by_reason={k: net.get(k, {}).get('records', 0) for k in REFINE_REASONS},
                seconds=round(m['metrics']['seconds'], 1), peak_tree_rss_mib=round(m['metrics']['peak_tree_rss_mib']))


def paragraph_fix():
    """r5 (paragraph sets) -> r6 (ordered subsequences): changed decisions and a text recheck of every r6 copy."""
    if not (PREVIOUS / 'manifest.json').exists():
        return None
    cols = ['rowid', 'decision', 'characters']
    old = pl.scan_parquet(str(PREVIOUS / 'decisions/*.parquet')).select(cols)
    new = pl.scan_parquet(str(FROZEN / 'decisions/*.parquet')).select(cols)
    changed = old.join(new, on='rowid', suffix='_r6').filter(pl.col('decision') != pl.col('decision_r6')).collect()
    restored = changed.filter(pl.col('decision_r6') == 'candidate')
    c5 = pl.read_parquet(PREVIOUS / 'paragraph_copies.parquet', columns=['rowid', 'host_rowid'])
    c6 = pl.read_parquet(FROZEN / 'paragraph_copies.parquet', columns=['rowid', 'host_rowid'])
    ids = pl.concat([c6['rowid'], c6['host_rowid']]).unique().to_list()
    text = dict(pl.scan_parquet(str(BASE / 'candidates/*.parquet')).select('rowid', 'text')
                .filter(pl.col('rowid').is_in(ids)).collect().iter_rows())
    failing = sum(not refine.ordered_within(refine.paragraph_list(text[a]), refine.paragraph_list(text[b]))
                  for a, b in c6.iter_rows())
    hosts = (pl.scan_parquet(str(FROZEN / 'decisions/*.parquet')).select('rowid', 'decision')
             .join(c6.select(pl.col('host_rowid').alias('rowid')).unique().lazy(), on='rowid').collect())
    return dict(r5_copies=c5.height, r6_copies=c6.height, restored=restored.height,
                restored_characters=int(restored['characters'].cast(pl.Int64).sum() or 0),
                newly_removed=changed.height - restored.height,
                host_reassigned=c5.join(c6, on='rowid', suffix='_r6').filter(pl.col('host_rowid') != pl.col('host_rowid_r6')).height,
                net_restored=restored.height - (changed.height - restored.height),
                r6_copies_failing_text_recheck=failing, r6_hosts=hosts.height,
                r6_hosts_retained=hosts.filter(pl.col('decision') == 'candidate').height)


def by_subreddit(floor):
    base = pl.scan_parquet(str(BASE / 'candidates/*.parquet')).select('rowid', 'subreddit', 'lexical_words')
    keep = pl.scan_parquet(str(FROZEN / 'decisions/*.parquet')).filter(pl.col('decision') == 'candidate').select('rowid')
    floor_only = base.filter(pl.col('lexical_words') >= floor).group_by('subreddit').len().rename({'len': 'floor_only_x20'})
    final = base.join(keep, on='rowid', how='semi').group_by('subreddit').len().rename({'len': 'final_r6'})
    return (floor_only.join(final, on='subreddit', how='full', coalesce=True).fill_null(0)
            .with_columns(removed_beyond_floor=pl.col('floor_only_x20') - pl.col('final_r6'))
            .sort('floor_only_x20', descending=True).collect())


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)


def summary(tests):
    base = json.loads((BASE / 'manifest.json').read_text())
    base_candidates = base['reasons'][refine.READY]
    m = json.loads((FROZEN / 'manifest.json').read_text())
    spec = json.loads((FROZEN / 'run-spec.json').read_text())
    rows = [version_row(FROZEN, base_candidates)] + [version_row(p, base_candidates) for p in COMPARE.values()
                                                      if (p / 'manifest.json').exists()]
    status = json.loads((RESULTS / 'status.json').read_text())
    subs = by_subreddit(m['min_lexical_words'])
    out = dict(
        schema_version=1,
        frozen_version=dict(dir=str(FROZEN), policy_revision=m['policy_revision'], config='configs/reddit-denoise-refine.json',
                            manifest_sha256=digest(FROZEN / 'manifest.json'),
                            policy_sha256=hashlib.sha256(json.dumps(spec['policy'], sort_keys=True).encode()).hexdigest(),
                            code_sha256=spec['code_sha256'], base_run=m['base_run'], base_manifest_sha256=spec['base_manifest_sha256']),
        source_records=m['source_records'], base_candidates=base_candidates,
        retained=dict(records=m['candidates'], characters=m['candidate_characters'], token_proxy=m['candidate_token_proxy'],
                      token_proxy_note=m['token_proxy_note'], by_kind=m['candidates_by_kind'],
                      retention_pct_of_source=round(m['retention_pct'], 3)),
        net_beyond_floor=m['net_beyond_floor'], standalone_rule_hits_on_base_candidates=m['standalone_rule_hits_on_base_candidates'],
        template_hits=m['template_hits'], body_duplicates=m['body_duplicates'], paragraph_containment=m['paragraph_containment'],
        paragraph_fix_r5_to_r6=paragraph_fix(), threshold_comparison=rows,
        retained_by_subreddit_top=subs.head(15).to_dicts(),
        review=dict(rounds=[{k: r[k] for k in ('round', 'state', 'policy_revision', 'seed', 'sample_size', 'reviewed', 'label_counts',
                                               'retained')} for r in status['rounds']],
                    voided_samples=status.get('voided_samples', []),
                    final_validation=('round 4: independent 1000 drawn from frozen r6 excluding every seen ID; M 0, template-handleable S 0, U 0; '
                                      'S and J are counted with boundary evidence, not claimed to be zero')),
        tests=tests,
        commands=dict(
            reproduce=('.venv-reddit-clean/bin/python -m weightloss.cleaning.reddit_denoise_refine '
                       '--output-dir .cache/reddit-cleaning/<new-dir> [--policy configs/reddit-denoise-refine.json]'),
            read=("PYTHONPATH=src .venv-reddit-clean/bin/python -c \"from weightloss.cleaning.reddit_denoise_refine import scan_candidates; "
                  "print(scan_candidates('.cache/reddit-cleaning/denoise-x20-r6-2026-09-25').select('rowid').collect().height)\""),
            export=('add --export-candidates with a NEW --output-dir; never rerun it in the frozen directory, because that rewrites '
                    'its manifest.json'),
            summary='.venv-reddit-clean/bin/python scripts/reddit_denoise_final.py summary && .venv-reddit-clean/bin/python scripts/reddit_denoise_rounds.py status'),
        notes=['net_beyond_floor attributes each removed record to the first applying reason (template > script > repetition > floor > '
               'duplicate_body_text > contained_paragraph_copy); it is not a leave-one-out ablation, since a record a template removes '
               'may also be a verbatim duplicate.',
               'floor_only_retained = base candidates with lexical_words >= x under the accepted denoise-coverage-1 rules.',
               'Identical or contained texts carry no author field, so they are described as same-text or expanded copies; '
               'authorship is not verified.'])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'final-summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    fields = ['min_lexical_words', 'policy_revision', 'floor_only_retained', 'final_rules_retained', 'removed_beyond_floor',
              'characters', 'token_proxy', *REFINE_REASONS, 'seconds', 'peak_tree_rss_mib']
    write_csv(RESULTS / 'threshold-comparison.csv', [{**r, **r['net_by_reason']} for r in rows], fields)
    write_csv(RESULTS / 'net-beyond-floor.csv', [dict(reason=k, **v) for k, v in m['net_beyond_floor'].items()],
              ['reason', 'records', 'characters'])
    subs.write_csv(RESULTS / 'retained-by-subreddit.csv')
    return out


def examples(per_reason, seed):
    dec = pl.scan_parquet(str(FROZEN / 'decisions/*.parquet')).filter(pl.col('reason').is_in(list(REFINE_REASONS)))
    picked = []
    for reason in REFINE_REASONS:
        rows = dec.filter(pl.col('reason') == reason).select('rowid', 'reason', 'detail').collect()
        if reason == 'mechanical_template':  # a few per template so every rule is shown
            rows = pl.concat([g.sample(min(3, g.height), seed=seed) for _, g in rows.group_by('detail', maintain_order=True)])
        else:
            rows = rows.sample(min(per_reason, rows.height), seed=seed)
        picked.append(rows)
    picked = pl.concat(picked).with_columns(
        other=pl.col('detail').str.extract(r'"(?:representative|host)_rowid": (\d+)').cast(pl.Int64))
    ids = pl.concat([picked['rowid'], picked['other'].drop_nulls()]).unique().to_list()
    base = {r['rowid']: r for r in pl.scan_parquet(str(BASE / 'candidates/*.parquet'))
            .select('rowid', 'fullname', 'kind', 'subreddit', 'title', 'text').filter(pl.col('rowid').is_in(ids)).collect().iter_rows(named=True)}
    out = ITER / 'r6-removed-examples.jsonl'
    with open(out, 'w') as f:
        for r in picked.iter_rows(named=True):
            row = dict(reason=r['reason'], detail=r['detail'], **base[r['rowid']])
            if r['other'] is not None:
                kept = base[r['other']]
                row['kept'] = dict(rowid=r['other'], fullname=kept['fullname'], kind=kept['kind'], subreddit=kept['subreddit'], text=kept['text'])
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return out, picked.group_by('reason').len().sort('reason').rows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--tests', default='', help='test command and outcome to record')
    e = sub.add_parser('examples')
    e.add_argument('--per-reason', type=int, default=10)
    e.add_argument('--seed', type=int, default=2026092607)
    args = parser.parse_args()
    if args.command == 'summary':
        out = summary(args.tests)
        print(json.dumps({k: out[k] for k in ('retained', 'threshold_comparison', 'paragraph_fix_r5_to_r6')}, ensure_ascii=False, indent=1))
    else:
        print(examples(args.per_reason, args.seed))


if __name__ == '__main__':
    main()
