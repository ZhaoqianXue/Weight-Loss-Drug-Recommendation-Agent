"""Draw the frozen dev-500 and test-200 gold samples for the v9 re-extraction (run once).

Dev: 320 random + 80 age/sex-enriched Reddit records, 90 patient + 10 caregiver WebMD reviews.
Test: 120 random + 40 age/sex-enriched Reddit records, 40 random WebMD reviews. Dev and test are disjoint. Reddit
records named in any earlier development or validation artifact are excluded (excluded-ids.json).
"""
import csv, hashlib, json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from weightloss.extraction.reextract import split_sentences, render

INPUT = os.path.join(ROOT, '.cache/extraction-postjev-2026-09-29/full-batch-p7c1-luna-low-20261001-01a0f4c2/inputs/full-256273.jsonl')
WEBMD = os.path.join(ROOT, 'data/raw/webmd/2026-09-12/webmd_all_reviews.csv')
RAW = os.path.join(ROOT, '.cache/reddit-production/full-01')
PRIV = os.path.join(ROOT, '.cache/reextraction-20261016/samples')
PUB = os.path.join(ROOT, 'results/reextraction-20261016/samples')
AGE_SEX = re.compile(r"\bI'?m\s+(?:a\s+)?\d{2}\b(?!\s*(?:lbs?|pounds|kg|mg|weeks|days|months|%))|\b\d{2}\s*(?:yo|y/o|years?\s+old|yrs?\s+old)\b"
                     r"|(?:^|[\s(])\d{2}\s?[MF]\b|(?:^|[\s(])[MF]\s?\d{2}\b|\b(?:in\s+my|early|mid|late)[\s-]*(?:20|30|40|50|60|70)'?s\b", re.I)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    excluded = set(json.load(open(os.path.join(PRIV, 'excluded-ids.json'))))
    reddit = [json.loads(l) for l in open(INPUT)]
    eligible = [r for r in reddit if r['record_id'].split(':', 1)[1] not in excluded]
    rng = random.Random(20261016)
    enriched_pool = [r for r in eligible if AGE_SEX.search(r['fields']['text'])]
    enriched = rng.sample(enriched_pool, 120)
    taken = {r['record_id'] for r in enriched}
    rest = [r for r in eligible if r['record_id'] not in taken]
    plain = rng.sample(rest, 440)
    dev_r = [(r, 'random') for r in plain[:320]] + [(r, 'age_sex_enriched') for r in enriched[:80]]
    test_r = [(r, 'random') for r in plain[320:]] + [(r, 'age_sex_enriched') for r in enriched[80:]]

    rows = [r for r in csv.DictReader(open(WEBMD, encoding='utf-8')) if r['Textual Review'].strip()]
    care = [r for r in rows if r['Patient Type'] == 'Caregiver']
    others = [r for r in rows if r['Patient Type'] != 'Caregiver']
    care_dev = rng.sample(care, 10)
    other_dev = rng.sample(others, 90)
    used = {r['Review ID'] for r in care_dev + other_dev}
    test_w = rng.sample([r for r in rows if r['Review ID'] not in used], 40)
    dev_w = [(r, 'caregiver') for r in care_dev] + [(r, 'random') for r in other_dev]
    test_w = [(r, 'random') for r in test_w]

    wanted = {r['record_id'].split(':', 1)[1] for r, _ in dev_r + test_r}
    meta = {}
    for sub in os.listdir(RAW):
        for kind in ('posts', 'comments'):
            path = os.path.join(RAW, sub, kind + '.jsonl')
            if not os.path.exists(path):
                continue
            for line in open(path, encoding='utf-8'):
                m = re.search(r'"id": "([a-z0-9]+)"', line)
                if m and (('t3_' if kind == 'posts' else 't1_') + m.group(1)) in wanted:
                    o = json.loads(line)
                    meta[('t3_' if kind == 'posts' else 't1_') + o['id']] = (sub, o['created_utc'])

    def reddit_rec(r, stratum):
        sid = r['record_id'].split(':', 1)[1]
        sents = split_sentences(r['fields']['text'])
        sub, utc = meta[sid]
        return {'platform': 'reddit', 'record_id': sid, 'kind': r['kind'], 'stratum': stratum, 'community': sub,
                'created_utc': utc, 'text': r['fields']['text'], 'sentences': sents,
                'model_input': render('reddit', r['kind'], sents)}

    def webmd_rec(r, stratum):
        brand, ingredient = r['Brand Name'], r['Drug Name'].lower()
        sents = split_sentences(r['Textual Review'])
        return {'platform': 'webmd', 'record_id': r['Review ID'], 'kind': 'review', 'stratum': stratum,
                'date': r['Date'], 'text': r['Textual Review'], 'sentences': sents,
                'profile': {'brand': brand, 'ingredient': ingredient, 'age': r['Age'], 'gender': r['Gender'],
                            'patient_type': r['Patient Type'], 'condition': r['Condition']},
                'model_input': render('webmd', 'review', sents, (brand, ingredient))}

    os.makedirs(PUB, exist_ok=True)
    manifest = {'created': '2026-10-16', 'seed': 20261016, 'reddit_input_sha256': sha(INPUT), 'webmd_csv_sha256': sha(WEBMD),
                'excluded_reddit_ids': len(excluded), 'eligible_reddit': len(eligible), 'enriched_pool': len(enriched_pool), 'sets': {}}
    for name, recs in (('dev500', [reddit_rec(r, s) for r, s in dev_r] + [webmd_rec(r, s) for r, s in dev_w]),
                       ('test200', [reddit_rec(r, s) for r, s in test_r] + [webmd_rec(r, s) for r, s in test_w])):
        with open(os.path.join(PRIV, name + '.jsonl'), 'w') as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        manifest['sets'][name] = {'n': len(recs), 'sha256': sha(os.path.join(PRIV, name + '.jsonl')),
                                  'by_stratum': {f"{p}/{s}": sum(1 for r in recs if (r['platform'], r['stratum']) == (p, s))
                                                 for p, s in sorted({(r['platform'], r['stratum']) for r in recs})},
                                  'ids': [f"{r['platform']}:{r['record_id']}" for r in recs]}
    json.dump(manifest, open(os.path.join(PUB, 'manifest.json'), 'w'), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'ids'} for k, v in manifest['sets'].items()}, indent=1))
    print('eligible', len(eligible), 'enriched pool', len(enriched_pool))


if __name__ == '__main__':
    main()
