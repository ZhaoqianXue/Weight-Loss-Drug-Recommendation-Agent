"""Mentions for MedDRA coding from v9 extraction results, in the input format of weightloss.standardization.coding.

Usage: python meddra_mentions.py --results <responses.jsonl | final.jsonl> --name <mention-set name>
Writes .cache/meddra-coding-20261015/mentions-<name>.jsonl, which the unchanged coding CLI then reads:
  run retrieve --mentions <name>; run select --mentions <name> --k 10 --effort medium ...; run export ...

One mention per extracted purpose and side effect, with the cited sentences as its quotes (Section 3.5), and one purpose
mention per WebMD profile condition (patient or blank profile, condition not "Other"). mention_id is
<platform>:<record_id>:d<group>:p<item> / :s<item> / :profile, the key the graph builder joins on.
"""
import argparse, collections, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUTS = os.path.join(ROOT, '.cache/reextraction-20261016/full/inputs.jsonl')
WORK = os.path.join(ROOT, '.cache/meddra-coding-20261015')


def profile_purpose(profile):
    """The WebMD profile condition used as a purpose, or None (caregiver profile, blank or "Other")."""
    if profile['patient_type'] not in ('Patient', ''):
        return None
    condition = profile['condition'].strip()
    return condition if condition and condition != 'Other' else None


def mentions(results, inputs):
    for r in results:
        if not r.get('ok'):
            continue
        key = f"{r['platform']}:{r['record_id']}"
        rec = inputs[key]
        text = {s['id']: s['text'] for s in rec['sentences']}
        base = {'record_id': key, 'platform': r['platform'], 'node_id': None}
        for g, drug in enumerate(r['output']['drugs']):
            for i, p in enumerate(drug['purposes']):
                yield {**base, 'mention_id': f'{key}:d{g}:p{i}', 'role': 'purpose', 'verbatim': p['purpose'],
                       'quotes': [text[e] for e in p['evidence']]}
            for i, s in enumerate(drug['side_effects']):
                yield {**base, 'mention_id': f'{key}:d{g}:s{i}', 'role': 'side_effect', 'verbatim': s['effect'],
                       'quotes': [text[e] for e in s['evidence']]}
        if r['platform'] == 'webmd':
            condition = profile_purpose(rec['profile'])
            if condition:
                yield {**base, 'mention_id': f'{key}:profile', 'role': 'purpose', 'verbatim': condition, 'quotes': [condition]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    results = [json.loads(l) for l in open(a.results)]
    want = {f"{r['platform']}:{r['record_id']}" for r in results}
    inputs = {}
    for line in open(INPUTS):
        rec = json.loads(line)
        k = f"{rec['platform']}:{rec['record_id']}"
        if k in want:
            inputs[k] = rec
    out = os.path.join(WORK, f'mentions-{a.name}.jsonl')
    if os.path.exists(out):
        raise SystemExit(f'{out} exists; nothing written')
    rows = list(mentions(results, inputs))
    with open(out, 'w') as f:
        for m in rows:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(out, len(rows), dict(collections.Counter((m['platform'], m['role'], m['mention_id'].endswith(':profile')) for m in rows)),
          'distinct verbatim', len({m['verbatim'] for m in rows}))


if __name__ == '__main__':
    main()
