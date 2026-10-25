"""Replay the frozen holdout; export aggregate threshold curves and source audit.

No raw corpus changes, model calls, or production policy changes. Private text
for boundary inspection stays under .cache. Words are whitespace-separated
items in cleaned own-record segments, excluding removed placeholders; context
is never counted toward the minimum.
"""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sqlite3

from weightloss.cleaning import reddit_basic as basic, reddit_noise as noise

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/reddit-threshold-audit-2026-09-25'
CACHE = ROOT / '.cache/reddit-cleaning/basic-literature-audit-2026-09-25'


def ro(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*((p*(1-p)/n + z*z/(4*n*n))**.5)/(1+z*z/n)
    return 100*(c-h), 100*(c+h)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    spec = json.loads((CACHE / 'run-spec.json').read_text())
    assert spec['code_sha256'] == basic.file_digest(basic.__file__)
    assert spec['supporting_code_sha256'][str(Path(noise.__file__).resolve())] == basic.file_digest(noise.__file__)
    old = ro(ROOT / '.cache/reddit-cleaning/basic-audit-2026-09-25/candidates.sqlite')
    old_ids = {r[0] for r in old.execute('SELECT source_rowid FROM decisions')}
    source = ro(ROOT / '.cache/reddit-cleaning/v2-strict/cleaned.sqlite')
    saved = ro(CACHE / 'candidates.sqlite')
    records = []; candidates = []; reasons = Counter(); diagnostics = Counter()
    for rid, decision, reason, sha in saved.execute('SELECT source_rowid,decision,reason,source_feature_sha256 FROM decisions ORDER BY source_rowid'):
        if rid in old_ids:
            continue
        payload = source.execute('SELECT features_json FROM records WHERE rowid=?', (rid,)).fetchone()[0]
        assert basic.digest(payload) == sha
        f = json.loads(payload)
        p = noise.prepare(f, spec['policy'])
        assert (p['decision'], p['reason']) == (decision, reason)
        segs = [s for i, s in enumerate(p['segments']) if i not in p.get('noise_placeholder_segments', [])]
        text = '\n\n'.join(s['text'] for s in segs)
        item = dict(fullname=f['fullname'], kind=f['kind'], subreddit=f['subreddit'],
                    source_rowid=rid, words=len(text.split()), characters=len(text),
                    candidate=decision == 'candidate', text=text,
                    context_references=len(p['context_references']), context_issues=p['context_issues'],
                    flags=p['flags'])
        records.append(item); reasons[reason] += 1
        if item['candidate']:
            candidates.append(item)
            diagnostics['no_unquoted_prose'] += not any(s['role'] == 'prose' and basic.readable(s['text']) for s in segs)
            diagnostics['alpha_ratio_below_half'] += sum(c.isalpha() for c in text)/max(1,len(text)) < .5
            diagnostics['context_unusable_flag'] += 'context_unusable_body_retained' in p['flags']
            diagnostics['promotion_marker_flag'] += any('promo' in flag for flag in p['flags'])
            diagnostics['crosspost_flag'] += 'crosspost_retained' in p['flags']
            diagnostics['comments_no_context_reference'] += f['kind'] == 'comments' and not p['context_references']
            diagnostics['posts_body_unavailable'] += f['kind'] == 'posts' and 'body_unavailable' in p['flags']
            diagnostics['posts_title_only'] += f['kind'] == 'posts' and bool(segs) and all(s.get('field') == 'title' for s in segs)
            diagnostics['min10_and_alpha_below_half'] += item['words'] >= 10 and sum(c.isalpha() for c in text)/max(1,len(text)) < .5
            diagnostics['min10_no_context_comments'] += item['words'] >= 10 and f['kind'] == 'comments' and not p['context_references']
            diagnostics['min10_title_only_posts'] += item['words'] >= 10 and f['kind'] == 'posts' and bool(segs) and all(s.get('field') == 'title' for s in segs)
    assert len(records) == 9992 and len(candidates) == 9407
    all_chars = sum(r['characters'] for r in candidates)
    curve = []
    for x in range(101):
        retained = [r for r in candidates if r['words'] >= x]
        lo, hi = wilson(len(retained), len(records))
        row = dict(min_words=x, retained=len(retained), total=len(records),
                   retention_pct=100*len(retained)/len(records),
                   retention_ci_low=lo, retention_ci_high=hi,
                   candidate_records_removed_pct=100*(1-len(retained)/len(candidates)),
                   candidate_chars_retained_pct=100*sum(r['characters'] for r in retained)/all_chars)
        for kind in ('posts','comments'):
            n = sum(r['kind'] == kind for r in records)
            k = sum(r['kind'] == kind for r in retained)
            row[kind+'_n'] = n; row[kind+'_retained'] = k
            row[kind+'_retention_pct'] = 100*k/n
        curve.append(row)
    assert all(a['retained'] >= b['retained'] for a,b in zip(curve,curve[1:]))
    with (OUT/'word-threshold-curve.csv').open('w') as h:
        w = csv.DictWriter(h, fieldnames=list(curve[0])); w.writeheader(); w.writerows(curve)
    rng = random.Random(20260927)
    examples = []
    bands = {}
    for name, low, high in [('1-4',1,4),('5-9',5,9),('10-19',10,19),('20+',20,10**9)]:
        subset = [r for r in candidates if low <= r['words'] <= high]
        bands[name] = dict(n=len(subset), posts=sum(r['kind']=='posts' for r in subset),
                           comments=sum(r['kind']=='comments' for r in subset),
                           characters=sum(r['characters'] for r in subset))
        examples.extend(dict(band=name, **r) for r in rng.sample(subset, min(40,len(subset))))
    (CACHE/'threshold-inspection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in examples))
    acquisition = json.loads((ROOT/'results/reddit-production-full-01/acquisition-completion.json').read_text())
    v1 = json.loads((ROOT/'results/reddit-cleaning-v1/manifest.json').read_text())
    production = ROOT/'.cache/reddit-production/full-01'
    communities = []; years = Counter(); annual_kinds = Counter()
    for c in acquisition['communities']:
        m = json.loads((production/c['subreddit']/'manifest.json').read_text())
        total = c['posts'] + c['comments']
        assert total == sum(v['count'] for v in m['counts'].values())
        communities.append(dict(subreddit=c['subreddit'], posts=c['posts'], comments=c['comments'],
            total=total, pct=100*total/acquisition['total_rows'], start_utc=m['start_utc'],
            earliest_actual_utc=min(v['earliest_utc'] for v in m['counts'].values() if v['count']),
            latest_actual_utc=max(v['latest_utc'] for v in m['counts'].values() if v['count']),
            source=m['spec']['source'], verification=c['verification']))
        for month in m['local_month_counts']:
            year = month['month_utc'][:4]
            years[year] += month['count']; annual_kinds[(year,month['kind'])] += month['count']
    assert sum(years.values()) == acquisition['total_rows'] == v1['unique_records']
    communities.sort(key=lambda r:r['total'],reverse=True)
    with (OUT/'source-communities.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(communities[0]));w.writeheader();w.writerows(communities)
    annual=[dict(year=y,posts=annual_kinds[(y,'posts')],comments=annual_kinds[(y,'comments')],total=years[y]) for y in sorted(years)]
    result = dict(sample_n=len(records), enhanced_candidates=len(candidates), reasons=reasons,
                  candidate_diagnostics=diagnostics, word_bands=bands,
                  selected_thresholds=[curve[x] for x in [0,3,5,8,10,15,20,30,50,100]],
                  source_database_sha256=spec['source_database_sha256'],
                  source_row_hashes_verified=len(records), policy_revision=spec['policy']['revision'],
                  source_total=acquisition['total_rows'], source_posts=acquisition['total_posts'],
                  source_comments=acquisition['total_comments'], source_communities=communities, annual_counts=annual,
                  duplicate_id_rows=v1['duplicate_id_rows'], repeated_text_groups=v1['same_text_multiple_ids'],
                  limitations=['Fixed holdout now used for threshold exploration, not final validation.',
                               'Wilson intervals quantify record-level sample variation only; not semantic accuracy.',
                               'Character volume excludes context, prompts and outputs; not billed tokens.',
                               'No new whole-database digest; replay verifies every selected source feature hash.',
                               'No human gold-standard labels or optimal-threshold claim.'])
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_communities','limitations')},indent=2))


if __name__ == '__main__':
    main()
