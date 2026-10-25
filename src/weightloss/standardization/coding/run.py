"""CLI: mentions -> LLT index -> retrieval -> selection -> LLT/PT rows.

  python -m weightloss.standardization.coding.run mentions
  python -m weightloss.standardization.coding.run index
  python -m weightloss.standardization.coding.run sample --n 1000 --seed 20261015 --name subset1000
  python -m weightloss.standardization.coding.run retrieve --mentions subset1000
  python -m weightloss.standardization.coding.run select --mentions subset1000 --k 10 20 50 --effort medium --run-id ...
  python -m weightloss.standardization.coding.run summarize --run-id ...

Outputs that contain MedDRA names or codes stay under the git-ignored .cache directory.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from . import VERSION, index as idx, mentions as mn, select as sel

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / '.cache' / 'meddra-coding-20261015'
KG = ROOT / 'results/extraction-postjev-2026-09-29/kg-full-conversion-20261001-01a0f9aa/outputs/formal-full/kg-packages.jsonl'
MEDDRA_DIR = ROOT / '.cache/terminology/meddra/29.1/English/api-20261004'
PINS = ROOT / 'results/extraction-postjev-2026-09-29/kg-standardization-20261004-01a0fad5/resources/acquisition-20261004/meddra-pins.json'
K_MAX = 50


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def code_sha256():
    here = Path(__file__).resolve().parent
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(here.glob('*.py'))}


def dictionary():
    from weightloss.standardization.semantic.meddra import load_meddra
    return load_meddra(MEDDRA_DIR, encoding='cp1252', expected_version='29.1', pinned_sha256=json.loads(PINS.read_text()))


def mention_file(name):
    return WORK / ('mentions.jsonl' if name == 'all' else f'mentions-{name}.jsonl')


def cmd_mentions(args):
    n = mn.build(KG, mention_file('all'))
    print(f'{n} mentions -> {mention_file("all")}')


def cmd_index(args):
    d = dictionary()
    meta = idx.build_llt_index(d, idx.Encoder(device=args.device), WORK / 'index')
    print(json.dumps(meta, indent=1))


def cmd_sample(args):
    rows = mn.load(mention_file('all'))
    sample = random.Random(args.seed).sample(rows, args.n)
    with open(mention_file(args.name), 'w') as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(sample)} sampled (seed {args.seed}) -> {mention_file(args.name)}',
          dict(collections.Counter(r['role'] for r in sample)))


UMLS_MRCONSO = ROOT / '.cache/terminology/umls/2026AA/META/MRCONSO.RRF'


def cmd_synonyms(args):
    meta = idx.build_synonym_index(dictionary(), idx.Encoder(device=args.device), UMLS_MRCONSO, WORK / 'index')
    print(json.dumps(meta, indent=1))


def cmd_retrieve(args):
    """Top-K_MAX candidates for every distinct verbatim of the mention set (smaller k use the prefix)."""
    vectors, codes, meta = idx.load_llt_index(WORK / 'index')
    out = WORK / f'topk-{args.mentions}{"-syn" if args.synonyms else ""}.jsonl'
    rows = mn.load(mention_file(args.mentions))
    texts = sorted({r['verbatim'] for r in rows})
    encoder = idx.Encoder(device=args.device)
    if encoder.revision != meta['revision']:
        raise SystemExit('encoder revision differs from the index revision')
    q = encoder.encode(texts)
    if args.synonyms:
        syn_vectors, syn_rows = idx.load_synonym_index(WORK / 'index')
        results = idx.top_k_codes(q, vectors, codes, syn_vectors, syn_rows, K_MAX)
    else:
        ix, scores = idx.top_k(q, vectors, K_MAX)
        results = [([codes[j] for j in i], [float(x) for x in s]) for i, s in zip(ix, scores)]
    with open(out, 'w') as f:
        for t, (cs, ss) in zip(texts, results):
            f.write(json.dumps({'verbatim': t, 'llt_codes': cs, 'scores': [round(x, 5) for x in ss]}, ensure_ascii=False) + '\n')
    print(f'{len(texts)} distinct verbatims -> {out}')


def cmd_select(args):
    d = dictionary()
    key = sel.load_key(ROOT / '.env')
    meta = json.loads((WORK / 'index' / 'index-meta.json').read_text())
    topk = {}
    with open(WORK / f'topk-{args.topk or args.mentions}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            topk[r['verbatim']] = r
    rows = mn.load(mention_file(args.mentions))
    run_dir = WORK / 'runs' / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / 'results.jsonl'
    done = set()
    spent = 0.0
    if results_path.exists():
        for line in open(results_path):
            r = json.loads(line)
            if not (args.retry_anomalies and r.get('anomaly')):
                done.add((r['mention_id'], r['k']))
            spent += r.get('cost_usd') or 0
    manifest = {'version': VERSION, 'run_id': args.run_id, 'created_utc': now(), 'mentions': args.mentions,
                'mentions_sha256': hashlib.sha256(mention_file(args.mentions).read_bytes()).hexdigest(),
                'k': args.k, 'effort': args.effort, 'model': sel.MODEL, 'endpoint': sel.UA_BASE,
                'prompt_version': sel.PROMPT_VERSION, 'prompt_sha256': sel.prompt_sha256(), 'index': meta,
                'dictionary_snapshot_sha256': d.snapshot_sha256, 'concurrency': args.concurrency, 'budget_usd': args.budget,
                'code_sha256': code_sha256()}
    manifest['retry_anomalies'] = bool(args.retry_anomalies)
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():  # a resumed or retry pass keeps the first manifest and records its own
        manifest_path = run_dir / f"manifest-{manifest['created_utc'].replace(':', '')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    jobs = [(r, k) for k in args.k for r in rows if (r['mention_id'], k) not in done]
    print(f'{len(jobs)} calls to make ({len(done)} already done); spent so far ${spent:.4f}', flush=True)
    lock = threading.Lock()

    def candidates(r, k):
        t = topk[r['verbatim']]
        out = []
        for code, score in zip(t['llt_codes'][:k], t['scores'][:k]):
            llt = d.llts[code]
            pt = d.pt_of(llt)
            out.append({'llt_code': code, 'llt_name': llt.name, 'pt_code': pt.code, 'pt_name': pt.name, 'score': score})
        return out

    def work(job):
        r, k = job
        cands = candidates(r, k)
        res = sel.call(r, cands, args.effort, key)
        chosen = None
        if res.get('choice') and res['choice'] != 'none':
            chosen = cands[sel.candidate_ids(k).index(res['choice'])]
        status = 'anomaly' if res.get('anomaly') else ('none' if res.get('choice') == 'none' else 'coded')
        return {'mention_id': r['mention_id'], 'record_id': r['record_id'], 'role': r['role'], 'verbatim': r['verbatim'],
                'k': k, 'effort': args.effort, 'status': status, 'choice': res.get('choice'),
                'rank': (sel.candidate_ids(k).index(res['choice']) + 1) if chosen else None,
                'llt_code': chosen and chosen['llt_code'], 'llt_name': chosen and chosen['llt_name'],
                'pt_code': chosen and chosen['pt_code'], 'pt_name': chosen and chosen['pt_name'],
                'candidate_llt_codes': [c['llt_code'] for c in cands], **{k2: v for k2, v in res.items() if k2 != 'choice'},
                'at': now()}

    stopped = None
    with open(results_path, 'a') as out, ThreadPoolExecutor(args.concurrency) as pool:
        pending, it = set(), iter(jobs)
        n_done = 0
        while True:
            while len(pending) < args.concurrency * 2 and stopped is None:
                job = next(it, None)
                if job is None:
                    break
                pending.add(pool.submit(work, job))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    row = fut.result()
                except sel.RoutingError as e:
                    stopped = f'routing: {e}'
                    continue
                with lock:
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    out.flush()
                    spent += row.get('cost_usd') or 0
                    n_done += 1
                if spent > args.budget and stopped is None:
                    stopped = f'budget ${args.budget} reached'
                if n_done % 500 == 0:
                    print(f'{now()} {n_done}/{len(jobs)} spent ${spent:.4f}', flush=True)
    print(f'finished {n_done} calls, spent ${spent:.4f}' + (f'; STOPPED: {stopped}' if stopped else ''))
    if stopped:
        sys.exit(2)


def cmd_summarize(args):
    rows = [json.loads(l) for l in open(WORK / 'runs' / args.run_id / 'results.jsonl')]
    by_k = collections.defaultdict(list)
    for r in rows:
        by_k[r['k']].append(r)
    summary = {'run_id': args.run_id, 'by_k': {}, 'agreement_pt': {}}
    for k, rs in sorted(by_k.items()):
        n = len(rs)
        st = collections.Counter(r['status'] for r in rs)
        lat = sorted(r['latency_s'] for r in rs if r.get('latency_s'))
        summary['by_k'][k] = {
            'calls': n, 'coded': st['coded'], 'none': st['none'], 'anomaly': st['anomaly'],
            'anomaly_kinds': dict(collections.Counter(r['anomaly'] for r in rs if r.get('anomaly'))),
            'mean_cost_usd': round(sum(r.get('cost_usd') or 0 for r in rs) / n, 7),
            'mean_input_tokens': round(np.mean([r['input_tokens'] for r in rs if r.get('input_tokens')]), 1),
            'mean_output_tokens': round(np.mean([r['output_tokens'] for r in rs if r.get('output_tokens')]), 1),
            'mean_reasoning_tokens': round(np.mean([r['reasoning_tokens'] or 0 for r in rs if r.get('output_tokens')]), 1),
            'latency_mean_s': round(float(np.mean(lat)), 2), 'latency_p90_s': round(lat[int(.9 * len(lat))], 2),
            'chosen_rank_hist': dict(sorted(collections.Counter(min(r['rank'], 21) for r in rs if r.get('rank')).items())),
            'routes': sorted({r.get('route') for r in rs if r.get('route')})}
    pt = {k: {r['mention_id']: (r['pt_code'] if r['status'] == 'coded' else r['status']) for r in rs} for k, rs in by_k.items()}
    ks = sorted(pt)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            common = set(pt[a]) & set(pt[b])
            same = sum(pt[a][m] == pt[b][m] for m in common)
            summary['agreement_pt'][f'{a}_vs_{b}'] = {'n': len(common), 'same': same, 'rate': round(same / len(common), 4)}
    (WORK / 'runs' / args.run_id / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def cmd_export(args):
    """One row per mention for one k: the final mapping, keyed by KG edge and node id (licensed: stays in .cache)."""
    run_dir = WORK / 'runs' / args.run_id
    nodes = {r['mention_id']: r['node_id'] for r in mn.load(mention_file(args.mentions))}
    latest = {}
    for line in open(run_dir / 'results.jsonl'):
        r = json.loads(line)
        if r['k'] == args.k:
            latest[r['mention_id']] = r
    missing = set(nodes) - set(latest)
    out = run_dir / f'mapping-k{args.k}.jsonl'
    fields = ('mention_id', 'record_id', 'role', 'verbatim', 'status', 'anomaly', 'llt_code', 'llt_name', 'pt_code', 'pt_name', 'rank')
    with open(out, 'w') as f:
        for mid, r in sorted(latest.items()):
            f.write(json.dumps({'node_id': nodes.get(mid), **{k2: r.get(k2) for k2 in fields}}, ensure_ascii=False) + '\n')
    counts = collections.Counter(r['status'] for r in latest.values())
    summary = {'run_id': args.run_id, 'k': args.k, 'mentions': len(nodes), 'mapped_rows': len(latest), 'missing': len(missing),
               'status': dict(counts), 'by_role': {role: dict(collections.Counter(r['status'] for r in latest.values() if r['role'] == role))
                                                    for role in ('side_effect', 'purpose')},
               'distinct_pt': len({r['pt_code'] for r in latest.values() if r['pt_code']}),
               'mapping_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    (run_dir / f'mapping-k{args.k}.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('mentions')
    s = sub.add_parser('index'); s.add_argument('--device', default='cpu')
    s = sub.add_parser('sample'); s.add_argument('--n', type=int, required=True); s.add_argument('--seed', type=int, default=20261015); s.add_argument('--name', required=True)
    s = sub.add_parser('synonyms'); s.add_argument('--device', default='cpu')
    s = sub.add_parser('retrieve'); s.add_argument('--mentions', required=True); s.add_argument('--device', default='cpu')
    s.add_argument('--synonyms', action='store_true', help='also search UMLS synonyms of MedDRA concepts')
    s = sub.add_parser('select'); s.add_argument('--mentions', required=True); s.add_argument('--k', type=int, nargs='+', required=True)
    s.add_argument('--effort', default='medium'); s.add_argument('--run-id', required=True)
    s.add_argument('--concurrency', type=int, default=16); s.add_argument('--budget', type=float, required=True)
    s.add_argument('--topk', default=None, help='candidate file name (default: same as --mentions)')
    s.add_argument('--retry-anomalies', action='store_true', help='also redo calls whose recorded result is an anomaly')
    s = sub.add_parser('summarize'); s.add_argument('--run-id', required=True)
    s = sub.add_parser('export'); s.add_argument('--run-id', required=True); s.add_argument('--mentions', required=True); s.add_argument('--k', type=int, required=True)
    args = p.parse_args(argv)
    if args.cmd == 'select' and max(args.k) > K_MAX:
        p.error(f'k must be <= {K_MAX}')
    {'mentions': cmd_mentions, 'index': cmd_index, 'synonyms': cmd_synonyms, 'sample': cmd_sample, 'retrieve': cmd_retrieve,
     'select': cmd_select, 'summarize': cmd_summarize, 'export': cmd_export}[args.cmd](args)


if __name__ == '__main__':
    main()
