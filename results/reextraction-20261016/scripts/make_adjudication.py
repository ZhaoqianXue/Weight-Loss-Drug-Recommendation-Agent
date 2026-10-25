"""Build blind adjudication packets from the disagreements of one or more runs with the current dev gold.

Usage: python make_adjudication.py <round-name> <run-id> [<run-id> ...]
Model-only items already judged "exclude" in earlier rounds and gold-only items already judged "include" are skipped.
Writes annotation/adjudication-<round>/packet-*.jsonl and annotation/adjudication-<round>-key.json (sources, private).
"""
import glob, json, os, random, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import items, match, text_match
from weightloss.extraction import reextract as rx
A = os.path.join(ROOT, '.cache/reextraction-20261016/annotation')
DET = ('dose', 'duration', 'intensity', 'impact', 'action_taken')


def full(out):
    res = []
    for d in (out or {}).get('drugs', []):
        for p in d['purposes']:
            res.append({'kind': 'purpose', 'brand': d['brand'], 'ingredient': d['ingredient'], 'text': p['purpose'], 'evidence': p['evidence']})
        for s in d['side_effects']:
            res.append({'kind': 'side_effect', 'brand': d['brand'], 'ingredient': d['ingredient'], 'text': s['effect'], **{k: s[k] for k in DET}, 'evidence': s['evidence']})
    return res


def judged():
    """(record -> list of (decision, item)) from every earlier round."""
    out = {}
    for keyf in glob.glob(os.path.join(A, 'adjudication-*-key.json')):
        rnd = keyf.rsplit('adjudication-', 1)[1][:-len('-key.json')]
        key = {p['record']: p for p in json.load(open(keyf))}
        for f in glob.glob(os.path.join(A, f'adjudication-{rnd}', 'decisions-*.jsonl')):
            for line in open(f):
                d = json.loads(line)
                cands = {c['cid']: c for c in key[d['record']]['candidates']}
                for x in d['decisions']:
                    c = cands[x['cid']]
                    out.setdefault(d['record'], []).append((x['decision'], {**c, **(x.get('item') or {})}))
    return out


def same(a, b):
    return a['kind'] == b['kind'] and (a['brand'], a['ingredient']) == (b['brand'], b['ingredient']) and text_match(a['text'], b['text'], a['evidence'], b['evidence'])


def main(rnd, runs):
    S = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(ROOT, '.cache/reextraction-20261016/samples/dev500.jsonl')))}
    G = {(g['platform'], g['record_id']): g['gold'] for g in map(json.loads, open(os.path.join(A, 'gold-dev500.jsonl')))}
    prev = judged()
    packets, rng = [], random.Random(hash(rnd) % 1000)
    for k in S:
        rec = f'{k[0]}:{k[1]}'
        gf, gi = full(G[k]), items(G[k])
        cands = []
        for run in runs:
            out = {(r['platform'], r['record_id']): r for r in map(json.loads, open(os.path.join(ROOT, '.cache/reextraction-20261016/runs', run, 'responses.jsonl')))}[k]['output']
            pf, pi = full(out), items(out)
            pairs = match(gi, pi); mg = {i for i, _ in pairs}; mp = {j for _, j in pairs}
            for j, it in enumerate(pf):
                if j in mp or any(dec == 'exclude' and same(c, it) for dec, c in prev.get(rec, [])) or any(same(c, it) for s, c in cands):
                    continue
                cands.append(('model_only', it))
            for i, it in enumerate(gf):
                if i in mg or any(dec == 'include' and same(c, it) for dec, c in prev.get(rec, [])) or any(same(c, it) for s, c in cands):
                    continue
                cands.append(('gold_only', it))
        if not cands:
            continue
        rng.shuffle(cands)
        r = S[k]
        inp = rx.render('reddit', r['kind'], r['sentences'], community=r['community']) if k[0] == 'reddit' else r['model_input']
        packets.append({'record': rec, 'input': inp, 'candidates': [{'cid': f'c{n}', **it} for n, (s, it) in enumerate(cands)],
                        '_sources': {f'c{n}': s for n, (s, it) in enumerate(cands)}})
    D = os.path.join(A, f'adjudication-{rnd}'); os.makedirs(D, exist_ok=True)
    json.dump(packets, open(os.path.join(A, f'adjudication-{rnd}-key.json'), 'w'))
    nb = max(1, min(5, len(packets) // 15))
    for b in range(nb):
        with open(os.path.join(D, f'packet-{b}.jsonl'), 'w') as f:
            for p in packets[b::nb]:
                f.write(json.dumps({k: v for k, v in p.items() if k != '_sources'}, ensure_ascii=False) + '\n')
    n = sum(len(p['candidates']) for p in packets)
    print(f'{rnd}: {len(packets)} records, {n} candidates, {nb} packets')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
