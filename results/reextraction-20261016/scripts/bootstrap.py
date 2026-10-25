"""Record-level stratified bootstrap 95% intervals for the primary metrics of one run.

Usage: python bootstrap.py --set test300 --gold <gold.jsonl> --run <run-id> [--n 2000]
Reddit metrics are weighted by the stratum weights in the sample; WebMD is unweighted. Resampling is within strata.
"""
import argparse, collections, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import items, match, SAMPLES, RUNS, ROOT


def per_record(sample, gold, run):
    rows = []
    for k, rec in sample.items():
        out = run[k]['output'] if run[k]['ok'] else None
        g, p = items(gold[k]), items(out)
        pairs = match(g, p)
        c = collections.Counter()
        for kind in ('purpose', 'side_effect'):
            tp = sum(1 for i, j in pairs if g[i]['kind'] == kind)
            c[kind, 'tp'] += tp
            c[kind, 'fn'] += sum(1 for x in g if x['kind'] == kind) - tp
            c[kind, 'fp'] += sum(1 for x in p if x['kind'] == kind) - tp
        gd = {(d['brand'], d['ingredient']) for d in gold[k].get('drugs', [])}
        pd = {(d['brand'], d['ingredient']) for d in (out or {}).get('drugs', [])}
        c['drug', 'tp'] += len(gd & pd); c['drug', 'fp'] += len(pd - gd); c['drug', 'fn'] += len(gd - pd)
        if k[0] == 'reddit':
            gw, pw = gold[k]['writer'], (out or {}).get('writer') or {}
            for f in ('age_group', 'sex'):
                gv, pv = gw[f], pw.get(f)
                if gv is None and pv is None:
                    continue
                if gv == pv:
                    c[f, 'tp'] += 1
                else:
                    c[f, 'fp'] += pv is not None; c[f, 'fn'] += gv is not None
        rows.append((k[0], rec.get('stratum', ''), rec.get('weight', 1.0), c))
    return rows


def metric(rows, plat, name):
    tp = sum(w * c[name, 'tp'] for p, s, w, c in rows if p == plat)
    fp = sum(w * c[name, 'fp'] for p, s, w, c in rows if p == plat)
    fn = sum(w * c[name, 'fn'] for p, s, w, c in rows if p == plat)
    return (tp / (tp + fp) if tp + fp else None, tp / (tp + fn) if tp + fn else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True); ap.add_argument('--gold', required=True); ap.add_argument('--run', required=True)
    ap.add_argument('--n', type=int, default=2000)
    a = ap.parse_args()
    sample = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(SAMPLES, a.set + '.jsonl')))}
    gold = {(g['platform'], str(g['record_id'])): g['gold'] for g in map(json.loads, open(a.gold))}
    run = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(RUNS, a.run, 'responses.jsonl')))}
    rows = per_record(sample, gold, run)
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r[0], r[1])].append(r)
    targets = [('reddit', n) for n in ('drug', 'side_effect', 'purpose', 'age_group', 'sex')] + [('webmd', n) for n in ('drug', 'side_effect', 'purpose')]
    point = {f'{p}/{n}': metric(rows, p, n) for p, n in targets}
    rng = random.Random(1)
    draws = collections.defaultdict(lambda: ([], []))
    for _ in range(a.n):
        boot = [r for rs in strata.values() for r in (rng.choice(rs) for _ in rs)]
        for p, n in targets:
            pr, rc = metric(boot, p, n)
            if pr is not None: draws[f'{p}/{n}'][0].append(pr)
            if rc is not None: draws[f'{p}/{n}'][1].append(rc)
    def ci(v):
        v = sorted(v); return (round(v[int(0.025 * len(v))], 3), round(v[int(0.975 * len(v)) - 1], 3)) if v else None
    out = {k: {'precision': round(point[k][0], 3) if point[k][0] is not None else None, 'precision_ci': ci(draws[k][0]),
               'recall': round(point[k][1], 3) if point[k][1] is not None else None, 'recall_ci': ci(draws[k][1])} for k in point}
    dest = os.path.join(ROOT, 'results/reextraction-20261016/scores', a.run, 'bootstrap.json')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
