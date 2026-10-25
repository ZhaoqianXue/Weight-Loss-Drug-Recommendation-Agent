"""Build the frozen full-run input: all 256,273 screened Reddit records and all 2,681 WebMD reviews with text.

Each line: platform, record_id, kind, community (Reddit), created_utc or date, sentences, model_input (header with the
community type only), and for WebMD the profile fields used directly in the graph. Writes
.cache/reextraction-20261016/full/inputs.jsonl (read-only) and results/reextraction-20261016/full-inputs-manifest.json.
"""
import csv, hashlib, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weightloss.extraction.reextract import split_sentences, render
from sample import INPUT, WEBMD, RAW, sha

OUT = os.path.join(ROOT, '.cache/reextraction-20261016/full/inputs.jsonl')


def main():
    reddit = [json.loads(l) for l in open(INPUT)]
    want = {r['record_id'].split(':', 1)[1] for r in reddit}
    meta = {}
    for sub in sorted(os.listdir(RAW)):
        for kind in ('posts', 'comments'):
            path = os.path.join(RAW, sub, kind + '.jsonl')
            if not os.path.exists(path):
                continue
            pre = 't3_' if kind == 'posts' else 't1_'
            for line in open(path, encoding='utf-8'):
                m = re.search(r'"id": "([a-z0-9]+)"', line)
                if m and pre + m.group(1) in want and pre + m.group(1) not in meta:
                    o = json.loads(line)
                    meta[pre + o['id']] = (sub, o['created_utc'])
    missing = [rid for rid in want if rid not in meta]
    if missing:
        raise SystemExit(f'{len(missing)} Reddit records have no raw metadata; nothing written')
    rows = []
    for r in reddit:
        sid = r['record_id'].split(':', 1)[1]; sents = split_sentences(r['fields']['text']); sub, utc = meta[sid]
        rows.append({'platform': 'reddit', 'record_id': sid, 'kind': r['kind'], 'community': sub, 'created_utc': utc,
                     'sentences': sents, 'model_input': render('reddit', r['kind'], sents, community=sub)})
    for w in csv.DictReader(open(WEBMD, encoding='utf-8')):
        if not w['Textual Review'].strip():
            continue
        sents = split_sentences(w['Textual Review']); brand, ing = w['Brand Name'], w['Drug Name'].lower()
        rows.append({'platform': 'webmd', 'record_id': w['Review ID'], 'kind': 'review', 'date': w['Date'], 'sentences': sents,
                     'profile': {'brand': brand, 'ingredient': ing, 'age': w['Age'], 'gender': w['Gender'],
                                 'patient_type': w['Patient Type'], 'condition': w['Condition'], 'medication_duration': w['Medication Duration']},
                     'model_input': render('webmd', 'review', sents, (brand, ing))})
    if os.path.exists(OUT):
        os.chmod(OUT, 0o644)
    with open(OUT, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.chmod(OUT, 0o444)
    man = {'inputs': OUT.replace(ROOT + '/', ''), 'sha256': sha(OUT), 'reddit': sum(r['platform'] == 'reddit' for r in rows),
           'webmd': sum(r['platform'] == 'webmd' for r in rows), 'source_reddit_sha256': sha(INPUT), 'source_webmd_sha256': sha(WEBMD),
           'empty_sentence_records': sum(1 for r in rows if not r['sentences'])}
    json.dump(man, open(os.path.join(ROOT, 'results/reextraction-20261016/full-inputs-manifest.json'), 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
