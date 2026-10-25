"""Draw the stratified test set v2 (replaces test200, which was never opened): 200 Reddit + 100 WebMD.

Reddit strata are defined by text patterns only (never by model output or gold): A = states an age or sex,
P = a reason pattern for drug use (and not A), N = the rest. Allocation A 55 / P 65 / N 80, random within strata;
population estimates use weights = stratum share of the eligible pool / stratum share of the sample. WebMD: 100 random.
The pool excludes every earlier development or validation record and the dev500 sample.
"""
import csv, json, os, random, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weightloss.extraction.reextract import split_sentences, render
from sample import AGE_SEX, INPUT, WEBMD, RAW, PRIV, PUB, sha

PURP = re.compile(r"\b(?:for|because(?: of)?|due to|since|to help(?: with)?|to (?:lose|control|lower|manage|reduce|treat|maintain|get))\b[^.!?\n]{0,40}?\b(?:my\s+)?(?:diabet\w*|t2d?\b|type ?(?:2|ii|two)|pcos|insulin resistan\w*|pre-?diabet\w*|a1c|blood sugars?|glucose|weight|obes\w*|sleep apnea|fatty liver|hypertension|blood pressure|maintenance|food noise|binge\w*|fat)\b"
                  r"|\b(?:prescribed|put (?:me )?on|started)\b[^.!?\n]{0,60}?\b(?:for|because|due to)\b", re.I)
ALLOC = {'A': 55, 'P': 65, 'N': 80}


def stratum(text):
    return 'A' if AGE_SEX.search(text) else ('P' if PURP.search(text) else 'N')


def main():
    excluded = set(json.load(open(os.path.join(PRIV, 'excluded-ids.json'))))
    dev = [json.loads(l) for l in open(os.path.join(PRIV, 'dev500.jsonl'))]
    excluded |= {r['record_id'] for r in dev if r['platform'] == 'reddit'}
    pool = {'A': [], 'P': [], 'N': []}
    for line in open(INPUT):
        r = json.loads(line)
        if r['record_id'].split(':', 1)[1] not in excluded:
            pool[stratum(r['fields']['text'])].append(r)
    rng = random.Random(20261017)
    picked = {h: rng.sample(pool[h], n) for h, n in ALLOC.items()}
    dev_w = {r['record_id'] for r in dev if r['platform'] == 'webmd'}
    rows = [r for r in csv.DictReader(open(WEBMD, encoding='utf-8')) if r['Textual Review'].strip() and r['Review ID'] not in dev_w]
    webmd = rng.sample(rows, 100)
    wanted = {r['record_id'].split(':', 1)[1] for h in picked for r in picked[h]}
    meta = {}
    for sub in os.listdir(RAW):
        for kind in ('posts', 'comments'):
            path = os.path.join(RAW, sub, kind + '.jsonl')
            if not os.path.exists(path):
                continue
            pre = 't3_' if kind == 'posts' else 't1_'
            for line in open(path, encoding='utf-8'):
                m = re.search(r'"id": "([a-z0-9]+)"', line)
                if m and pre + m.group(1) in wanted:
                    o = json.loads(line); meta[pre + o['id']] = (sub, o['created_utc'])
    total = sum(len(v) for v in pool.values())
    weights = {h: (len(pool[h]) / total) / (ALLOC[h] / sum(ALLOC.values())) for h in ALLOC}
    recs = []
    for h in ('A', 'P', 'N'):
        for r in picked[h]:
            sid = r['record_id'].split(':', 1)[1]; sents = split_sentences(r['fields']['text']); sub, utc = meta[sid]
            recs.append({'platform': 'reddit', 'record_id': sid, 'kind': r['kind'], 'stratum': h, 'weight': weights[h],
                         'community': sub, 'created_utc': utc, 'text': r['fields']['text'], 'sentences': sents,
                         'model_input': render('reddit', r['kind'], sents, community=sub)})
    for r in webmd:
        sents = split_sentences(r['Textual Review'])
        recs.append({'platform': 'webmd', 'record_id': r['Review ID'], 'kind': 'review', 'stratum': 'W', 'weight': 1.0,
                     'date': r['Date'], 'text': r['Textual Review'], 'sentences': sents,
                     'profile': {'brand': r['Brand Name'], 'ingredient': r['Drug Name'].lower(), 'age': r['Age'], 'gender': r['Gender'],
                                 'patient_type': r['Patient Type'], 'condition': r['Condition']},
                     'model_input': render('webmd', 'review', sents, (r['Brand Name'], r['Drug Name'].lower()))})
    out = os.path.join(PRIV, 'test300.jsonl')
    with open(out, 'w') as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    os.chmod(out, 0o444)
    man = json.load(open(os.path.join(PUB, 'manifest.json')))
    man['sets']['test200']['status'] = 'superseded on 2026-10-17 by test300 before anyone opened it (user decision: 200 Reddit + 100 WebMD, stratified)'
    man['sets']['test300'] = {'n': len(recs), 'sha256': sha(out), 'seed': 20261017, 'allocation': ALLOC,
                              'pool_sizes': {h: len(v) for h, v in pool.items()}, 'weights': weights,
                              'strata': 'A age/sex text pattern; P reason pattern (not A); N rest; text rules only',
                              'ids': [f"{r['platform']}:{r['record_id']}" for r in recs]}
    json.dump(man, open(os.path.join(PUB, 'manifest.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in man['sets']['test300'].items() if k != 'ids'}, indent=1))


if __name__ == '__main__':
    main()
