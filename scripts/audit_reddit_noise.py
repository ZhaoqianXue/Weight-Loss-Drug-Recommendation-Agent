"""Compare basic noise policies and length ablations on saved random samples.

Run from the repository root with the reddit-clean environment. Public output
contains aggregates only; inspection text stays under the private cache.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sqlite3

from weightloss.cleaning import reddit_basic as basic, reddit_noise as noise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('.cache/reddit-cleaning/v2-strict/cleaned.sqlite'))
    ap.add_argument('--old-run', type=Path, default=Path('.cache/reddit-cleaning/basic-audit-2026-09-25'))
    ap.add_argument('--new-run', type=Path, default=Path('.cache/reddit-cleaning/basic-literature-audit-2026-09-25'))
    ap.add_argument('--output', type=Path, default=Path('results/reddit-basic-literature-2026-09-25/benchmark.json'))
    args = ap.parse_args()
    old_spec = json.loads((args.old_run / 'run-spec.json').read_text())
    new_spec = json.loads((args.new_run / 'run-spec.json').read_text())
    if old_spec['source_database_sha256'] != new_spec['source_database_sha256']:
        raise ValueError('Samples have different source snapshots')
    if new_spec['code_sha256'] != basic.file_digest(basic.__file__):
        raise ValueError('Basic implementation differs from new run')
    if new_spec['supporting_code_sha256'].get(str(Path(noise.__file__).resolve())) != basic.file_digest(noise.__file__):
        raise ValueError('Noise implementation differs from new run')
    source = sqlite3.connect(f'file:{args.source.resolve()}?mode=ro', uri=True)
    dbs = [sqlite3.connect(f'file:{(p / "candidates.sqlite").resolve()}?mode=ro', uri=True)
           for p in [args.old_run, args.new_run]]
    old_ids = {r[0] for r in dbs[0].execute('SELECT source_rowid FROM decisions')}
    report = {}; inspections = []; overlap = 0
    for index, db in enumerate(dbs):
        counts = Counter(); transitions = Counter(); reasons = Counter(); ablations = Counter(); chars = Counter()
        groups = defaultdict(Counter); new_exclusions = []; short_examples = []; remaining = []
        for rid, ident, stored_decision, stored_reason, feature_sha in db.execute(
                'SELECT source_rowid,fullname,decision,reason,source_feature_sha256 FROM decisions ORDER BY source_rowid'):
            if index == 1 and rid in old_ids:
                overlap += 1
                continue
            payload = source.execute('SELECT features_json FROM records WHERE rowid=?', (rid,)).fetchone()[0]
            if basic.digest(payload) != feature_sha:
                raise ValueError('Source row changed')
            f = json.loads(payload)
            old = basic.prepare(f, old_spec['policy'])
            new = noise.prepare(f, new_spec['policy'])
            stored = old if index == 0 else new
            if (stored['decision'], stored['reason']) != (stored_decision, stored_reason):
                raise ValueError('Saved classification does not replay')
            counts['total'] += 1
            counts['old_candidates'] += old['decision'] == 'candidate'
            counts['new_candidates'] += new['decision'] == 'candidate'
            transitions[f"{old['decision']} -> {new['decision']}"] += 1
            reasons[new['reason']] += 1
            groups[f['subreddit']]['total'] += 1
            groups[f['subreddit']]['new_candidates'] += new['decision'] == 'candidate'
            text = '\n\n'.join(s['text'] for i, s in enumerate(new['segments'])
                               if i not in new.get('noise_placeholder_segments', []))
            words = len(text.split())
            chars['source_segments'] += sum(len(s['text']) for s in f['segments'])
            if old['decision'] == 'candidate':
                chars['old_candidate_segments'] += sum(len(s['text']) for s in old['segments'])
            item = dict(source_rowid=rid, fullname=ident, subreddit=f['subreddit'], reason=new['reason'],
                        words=words, text='\n\n'.join(s['text'] for s in f['segments']))
            if old['decision'] == 'candidate' and new['decision'] == 'exclude':
                new_exclusions.append(item)
            if new['decision'] == 'candidate':
                chars['new_candidate_segments'] += sum(len(s['text']) for i, s in enumerate(new['segments'])
                                                   if i not in new.get('noise_placeholder_segments', []))
                for minimum in [3, 5, 10, 20]:
                    ablations[f'min_{minimum}_whitespace_words'] += words >= minimum
                for minimum in [15, 30, 100]:
                    ablations[f'min_{minimum}_chars'] += len(text) >= minimum
                ablations['min_10_words_and_half_alpha'] += words >= 10 and sum(c.isalpha() for c in text) / max(len(text), 1) >= .5
                if words < 5:
                    short_examples.append(item)
                if 'unquoted_automation_disclosure' in f['flags']:
                    counts['residual_unquoted_automation_flag'] += 1
                remaining.append(item)
        label = 'development_10000' if index == 0 else 'holdout_excluding_development'
        report[label] = dict(counts=counts, transitions=transitions, reasons=reasons,
                             ablation_candidate_counts=ablations, characters=chars, per_community=groups)
        rng = random.Random(20260926)
        for kind, items, size in [('newly_excluded', new_exclusions, 100), ('short_retained', short_examples, 100),
                                  ('random_retained', remaining, 100)]:
            for item in rng.sample(items, min(size, len(items))):
                inspections.append(dict(cohort=label, inspection_stratum=kind, **item))
    result = dict(schema_version=1, source_database_sha256=new_spec['source_database_sha256'],
                  code_sha256=new_spec['code_sha256'], supporting_code_sha256=new_spec['supporting_code_sha256'],
                  policy_revision=new_spec['policy']['revision'], overlapping_sample_rows_excluded=overlap,
                  cohorts=report, independent_human_precision=None, independent_human_recall=None,
                  limitations=['Length ablations apply local whitespace/character counts; not exact paper replications.',
                               'No Jev calls or end-to-end extraction validation.',
                               'Agent qualitative inspection is not independent human gold-standard validation.',
                               'No preset retention target or inference that all retained records are relevant.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.new_run / 'inspection.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False) + '\n' for i in inspections))
    print(json.dumps({k: {n: v for n, v in cohort.items() if n != 'per_community'} for k, cohort in report.items()}, indent=2))
    print('Excluded overlapping rows:', overlap)
    for db in [source, *dbs]:
        db.close()


if __name__ == '__main__':
    main()
