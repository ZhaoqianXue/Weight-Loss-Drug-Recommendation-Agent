"""Apply blind adjudication decisions to the dev gold (writes a new gold file and a change log).

Usage: python apply_adjudication.py <gold_in> <key.json> <decisions_dir> <gold_out> <log_out>
Gold items that were not candidates stay unchanged; gold-sourced candidates are kept (as corrected) or removed;
model-sourced candidates judged "include" are added. Items are regrouped by (brand, ingredient).
"""
import glob, json, sys

DET = ('dose', 'duration', 'intensity', 'impact', 'action_taken')


def flat(gold):
    out = []
    for d in gold['drugs']:
        for p in d['purposes']:
            out.append({'kind': 'purpose', 'brand': d['brand'], 'ingredient': d['ingredient'], 'text': p['purpose'], 'evidence': p['evidence']})
        for s in d['side_effects']:
            out.append({'kind': 'side_effect', 'brand': d['brand'], 'ingredient': d['ingredient'], 'text': s['effect'],
                        **{k: s[k] for k in DET}, 'evidence': s['evidence']})
    return out


def regroup(items, writer):
    groups = {}
    for it in items:
        g = groups.setdefault((it['brand'], it['ingredient']), {'brand': it['brand'], 'ingredient': it['ingredient'], 'purposes': [], 'side_effects': []})
        if it['kind'] == 'purpose':
            g['purposes'].append({'purpose': it['text'], 'evidence': sorted(set(it['evidence']))})
        else:
            g['side_effects'].append({'effect': it['text'], **{k: it.get(k) if k != 'action_taken' else (it.get(k) or []) for k in DET},
                                      'evidence': sorted(set(it['evidence']))})
    out = {'drugs': list(groups.values())}
    if writer is not None:
        out = {'writer': writer, **out}
    return out


def main(gold_in, key_path, dec_dir, gold_out, log_out):
    gold = [json.loads(l) for l in open(gold_in)]
    key = {p['record']: p for p in json.load(open(key_path))}
    dec = {}
    for f in sorted(glob.glob(dec_dir + '/decisions-*.jsonl')):
        for line in open(f):
            d = json.loads(line); dec[d['record']] = {x['cid']: x for x in d['decisions']}
    log = []
    for g in gold:
        rec = f"{g['platform']}:{g['record_id']}"
        if rec not in key:
            continue
        cands, src, ds = key[rec]['candidates'], key[rec]['_sources'], dec[rec]
        items = flat(g['gold'])
        gold_cands = [c for c in cands if src[c['cid']] in ('gold_only', 'both')]
        keep = []
        for it in items:
            hit = next((c for c in gold_cands if {k: v for k, v in c.items() if k != 'cid'} == it), None)
            if hit is None:
                keep.append(it)
        for c in cands:
            d = ds[c['cid']]
            if d['decision'] == 'include':
                new = {**{k: v for k, v in c.items() if k != 'cid'}, **(d.get('item') or {})}
                new.pop('cid', None)
                keep.append(new)
                if src[c['cid']] == 'model_only' or new != {k: v for k, v in c.items() if k != 'cid'}:
                    log.append({'record': rec, 'source': src[c['cid']], 'change': 'added' if src[c['cid']] == 'model_only' else 'corrected', 'item': new, 'reason': d['reason']})
            elif src[c['cid']] != 'model_only':
                log.append({'record': rec, 'source': src[c['cid']], 'change': 'removed', 'item': {k: v for k, v in c.items() if k != 'cid'}, 'reason': d['reason']})
        g['gold'] = regroup(keep, g['gold'].get('writer'))
    with open(gold_out, 'w') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    json.dump(log, open(log_out, 'w'), indent=1, ensure_ascii=False)
    from collections import Counter
    print(Counter((e['source'], e['change']) for e in log))


if __name__ == '__main__':
    main(*sys.argv[1:6])
