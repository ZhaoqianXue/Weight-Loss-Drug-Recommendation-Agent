"""Score a v9 run against a gold file and estimate the full-corpus Batch cost and time.

Usage: python score.py --set dev500 --gold <gold.jsonl> --run <run-id> [--out <dir>]
Matching: within a record, a predicted item matches a gold item of the same kind (purpose or side effect) when the drug
identity (brand, ingredient) is equal and the wording matches (equal or contained after normalization, very similar, or
sharing an evidence sentence and either similar wording or a content word). Matching is one-to-one. Item metrics without the drug condition are reported for diagnosis.
"""
import argparse, collections, difflib, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLES = os.path.join(ROOT, '.cache/reextraction-20261016/samples')
RUNS = os.path.join(ROOT, '.cache/reextraction-20261016/runs')
FULL = {'reddit': 256273, 'webmd': 2681}
PRICE = {'input': 0.10, 'cached': 0.01, 'cache_write': 0.125, 'output': 0.50}  # USD per 1M, standard tier
BATCH_REF = {'responses': 256280, 'hours': 14.4, 'output_tokens_per_response': 195928099 / 256280}  # 2026-10-01 first pass
DETAILS = ('dose', 'duration', 'intensity', 'impact', 'action_taken')


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'^[\W_]+|[\W_]+$', '', (s or '').casefold())).strip()


def text_match(a, b, ev_a, ev_b):
    x, y = norm(a), norm(b)
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    if x in y or y in x:
        return 0.9
    r = difflib.SequenceMatcher(None, x, y).ratio()
    if r >= 0.75:
        return r
    if set(ev_a) & set(ev_b) and (r >= 0.5 or content_words(x) & content_words(y)):
        return max(r, 0.5)
    return 0.0


STOP = set('a an the my i me of to in on at and or with from for it its this that was is be been have had has very so really just all'.split())


def content_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 2}


def items(out):
    res = []
    for d in (out or {}).get('drugs', []):
        key = (d.get('brand'), d.get('ingredient'))
        for p in d.get('purposes', []):
            res.append({'kind': 'purpose', 'drug': key, 'text': p['purpose'], 'ev': p['evidence']})
        for s in d.get('side_effects', []):
            res.append({'kind': 'side_effect', 'drug': key, 'text': s['effect'], 'ev': s['evidence'], **{k: s[k] for k in DETAILS}})
    return res


def match(gold, pred, use_drug=True):
    pairs = []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if g['kind'] != p['kind'] or (use_drug and g['drug'] != p['drug']):
                continue
            s = text_match(g['text'], p['text'], g['ev'], p['ev'])
            if s:
                pairs.append((s, i, j))
    used_g, used_p, out = set(), set(), []
    for s, i, j in sorted(pairs, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i); used_p.add(j); out.append((i, j))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p and round(p, 3), 'recall': r and round(r, 3), 'f1': f and round(f, 3)}


def detail_equal(k, a, b):
    if k == 'action_taken':
        return sorted(a or []) == sorted(b or [])
    if k in ('dose', 'duration'):
        x, y = norm(a), norm(b)
        return x == y or (x and y and (x in y or y in x))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--gold', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    a = ap.parse_args()
    sample = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(SAMPLES, a.set + '.jsonl')))}
    W = {k: r.get('weight', 1.0) for k, r in sample.items()}
    ST = {k: r.get('stratum', '') for k, r in sample.items()}
    gold = {(g['platform'], str(g['record_id'])): g['gold'] for g in map(json.loads, open(a.gold))}
    run = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(RUNS, a.run, 'responses.jsonl')))}
    keys = [k for k in sample if k in gold and k in run]

    c = collections.defaultdict(collections.Counter)
    cw = collections.defaultdict(collections.Counter)
    cs = collections.defaultdict(collections.Counter)
    errors = []
    for k in keys:
        plat = k[0]
        out = run[k]['output'] if run[k]['ok'] else None
        if not run[k]['ok']:
            c['anomaly'][plat] += 1
        g_items, p_items = items(gold[k]), items(out)
        pairs = match(g_items, p_items)
        loose = match(g_items, p_items, use_drug=False)
        for kind in ('purpose', 'side_effect'):
            for scope, ps in (('', pairs), ('nodrug_', loose)):
                tp = sum(1 for i, j in ps if g_items[i]['kind'] == kind)
                fn = sum(1 for g in g_items if g['kind'] == kind) - tp
                fp = sum(1 for p in p_items if p['kind'] == kind) - tp
                c[scope + kind][(plat, 'tp')] += tp
                c[scope + kind][(plat, 'fn')] += fn
                c[scope + kind][(plat, 'fp')] += fp
                if not scope:
                    for x, v in (('tp', tp), ('fp', fp), ('fn', fn)):
                        cw[kind][(plat, x)] += v * W[k]; cs[(ST[k], kind)][x] += v
        mg, mp = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, g in enumerate(g_items):
            if i not in mg:
                errors.append({'record': f'{plat}:{k[1]}', 'error': 'FN', 'kind': g['kind'], 'drug': g['drug'], 'text': g['text'], 'ev': g['ev']})
        for j, p in enumerate(p_items):
            if j not in mp:
                errors.append({'record': f'{plat}:{k[1]}', 'error': 'FP', 'kind': p['kind'], 'drug': p['drug'], 'text': p['text'], 'ev': p['ev']})
        for i, j in pairs:
            g, p = g_items[i], p_items[j]
            if g['kind'] != 'side_effect':
                continue
            for d in DETAILS:
                gv, pv = g[d], p[d]
                g_set, p_set = gv not in (None, []), pv not in (None, [])
                if not g_set and not p_set:
                    continue
                if g_set and p_set and detail_equal(d, gv, pv):
                    c[d]['tp'] += 1
                else:
                    c[d]['fp'] += p_set
                    c[d]['fn'] += g_set
                    errors.append({'record': f'{plat}:{k[1]}', 'error': 'detail', 'field': d, 'effect': g['text'], 'gold': gv, 'pred': pv})
        g_drugs = {(d['brand'], d['ingredient']) for d in gold[k].get('drugs', [])}
        p_drugs = {(d['brand'], d['ingredient']) for d in (out or {}).get('drugs', [])}
        c['drug'][(plat, 'tp')] += len(g_drugs & p_drugs)
        c['drug'][(plat, 'fp')] += len(p_drugs - g_drugs)
        c['drug'][(plat, 'fn')] += len(g_drugs - p_drugs)
        for x, v in (('tp', len(g_drugs & p_drugs)), ('fp', len(p_drugs - g_drugs)), ('fn', len(g_drugs - p_drugs))):
            cw['drug'][(plat, x)] += v * W[k]; cs[(ST[k], 'drug')][x] += v
        for gd in g_drugs - p_drugs:
            errors.append({'record': f'{plat}:{k[1]}', 'error': 'drug_FN', 'drug': gd})
        for pd in p_drugs - g_drugs:
            errors.append({'record': f'{plat}:{k[1]}', 'error': 'drug_FP', 'drug': pd})
        if plat == 'reddit':
            gw, pw = gold[k]['writer'], (out or {}).get('writer') or {'age_group': None, 'sex': None}
            for f in ('age_group', 'sex'):
                gv, pv = gw[f], pw.get(f)
                if gv is None and pv is None:
                    continue
                if gv == pv:
                    c[f]['tp'] += 1; cw[f]['tp'] += W[k]; cs[(ST[k], f)]['tp'] += 1
                else:
                    c[f]['fp'] += pv is not None
                    c[f]['fn'] += gv is not None
                    cw[f]['fp'] += (pv is not None) * W[k]; cw[f]['fn'] += (gv is not None) * W[k]
                    cs[(ST[k], f)]['fp'] += pv is not None; cs[(ST[k], f)]['fn'] += gv is not None
                    errors.append({'record': f'{plat}:{k[1]}', 'error': 'writer', 'field': f, 'gold': gv, 'pred': pv})

    def by_plat(name):
        res = {}
        for plat in ('reddit', 'webmd', 'all'):
            plats = ('reddit', 'webmd') if plat == 'all' else (plat,)
            res[plat] = prf(*(sum(c[name][(p, x)] for p in plats) for x in ('tp', 'fp', 'fn')))
        return res

    metrics = {'records_scored': len(keys), 'anomalies': dict(c['anomaly']),
               'primary': {'drug': by_plat('drug'), 'purpose': by_plat('purpose'), 'side_effect': by_plat('side_effect'),
                           'age_group': prf(c['age_group']['tp'], c['age_group']['fp'], c['age_group']['fn']),
                           'sex': prf(c['sex']['tp'], c['sex']['fp'], c['sex']['fn'])},
               'diagnostic_items_ignoring_drug': {'purpose': by_plat('nodrug_purpose'), 'side_effect': by_plat('nodrug_side_effect')},
               'secondary_details_on_matched_side_effects': {d: prf(c[d]['tp'], c[d]['fp'], c[d]['fn']) for d in DETAILS}}

    if any(w != 1.0 for w in W.values()):
        def wprf(name, plat=None):
            g = cw[name]
            vals = [g[(plat, x)] for x in ('tp', 'fp', 'fn')] if plat else [g[x] for x in ('tp', 'fp', 'fn')]
            tp, fp, fn = vals
            return {'precision': round(tp / (tp + fp), 3) if tp + fp else None, 'recall': round(tp / (tp + fn), 3) if tp + fn else None}
        metrics['reddit_weighted'] = {n: wprf(n, 'reddit') for n in ('drug', 'purpose', 'side_effect')}
        metrics['reddit_weighted'].update({n: wprf(n) for n in ('age_group', 'sex')})
        metrics['per_stratum'] = {f'{s}/{n}': prf(v['tp'], v['fp'], v['fn']) for (s, n), v in sorted(cs.items())}

    # Full-corpus estimate from this run's measured usage (all attempts, i.e. including retries).
    tok = collections.defaultdict(lambda: collections.Counter())
    secs, n_rec = [], collections.Counter()
    for rk, r in run.items():
        wt = W.get(rk, 1.0)
        n_rec[r['platform']] += wt
        for t in r['attempts']:
            u = t.get('usage') or {}
            d = u.get('input_tokens_details') or {}
            cached, write = d.get('cached_tokens') or 0, d.get('cache_write_tokens') or 0
            tok[r['platform']]['plain'] += wt * max((u.get('input_tokens') or 0) - cached - write, 0)
            tok[r['platform']]['cached'] += wt * cached
            tok[r['platform']]['cache_write'] += wt * write
            tok[r['platform']]['output'] += wt * (u.get('output_tokens') or 0)
            tok[r['platform']]['reasoning'] += wt * ((u.get('output_tokens_details') or {}).get('reasoning_tokens') or 0)
            tok[r['platform']]['requests'] += wt
    est = {'per_record_mean_tokens': {}, 'full_records': FULL}
    usd_std = 0.0
    out_per_req, reqs = 0.0, 0
    for plat, n in FULL.items():
        if not n_rec[plat]:
            continue
        per = {k: v / n_rec[plat] for k, v in tok[plat].items()}
        est['per_record_mean_tokens'][plat] = {k: round(v, 1) for k, v in per.items()}
        usd_std += n * (per['plain'] * PRICE['input'] + per['cached'] * PRICE['cached'] + per['cache_write'] * PRICE['cache_write']
                        + per['output'] * PRICE['output']) / 1e6
        out_per_req += n * per['output']
        reqs += n * per['requests']
    out_per_req /= reqs
    manifest = json.load(open(os.path.join(RUNS, a.run, 'manifest.json')))
    est.update({'batch_usd': round(usd_std / 2, 2), 'standard_sync_usd': round(usd_std, 2),
                'batch_hours_estimate': round(BATCH_REF['hours'] * out_per_req / BATCH_REF['output_tokens_per_response'], 1),
                'batch_hours_basis': 'prior full Batch first pass (2026-10-01): 256,280 responses in 14.4 h, 764 output tokens each; scaled by output tokens per request',
                'sync_hours_at_run_workers': round(sum(FULL.values()) / (manifest['records'] / manifest['wall_seconds']) / 3600, 1),
                'sync_workers': manifest['workers'], 'run_spent_usd_standard': manifest['spent_usd_standard']})
    metrics['full_corpus_estimate'] = est
    out_dir = a.out or os.path.join(ROOT, 'results/reextraction-20261016/scores', a.run)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(metrics, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=1)
    priv = os.path.join(RUNS, a.run, 'errors.jsonl')
    with open(priv, 'w') as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(json.dumps(metrics, indent=1))


if __name__ == '__main__':
    main()
