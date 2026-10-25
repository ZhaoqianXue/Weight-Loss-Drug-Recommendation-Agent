"""Final product for the ~1,000,000-record strict subset (lossy selection on top of x=20); text stays in .cache.

export:  writes an independent, readable copy of the frozen strict version to a NEW private directory:
         records/part-*.parquet (selected by the strict version's final `decision`, never by the inherited
         `version_decision`), representatives.parquet (removed copies/duplicates -> kept representative, with a
         flag for whether the representative is in the final set), manifest.json (counts, sha256, schema, source
         manifests, verification) and README.md. Every exported text is re-read and compared with the base text.
validate: independent cross-check of the export (exported set == final decision set, no strict removal exported),
         review-sample fidelity (text, digits and negations identical), which reviewed items the final version removed,
         named examples, and a census of residual structures (information only, never used for selection). Writes
         a private JSON with IDs to the iterations directory.
summary: aggregate tables for results/: per-layer counts and reductions, strict reasons and overlaps, timings,
         review rounds, audits and the validation counts. No text leaves .cache.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import polars as pl

CACHE = Path('.cache/reddit-cleaning')
BASE = CACHE / 'denoise-x10-2026-09-25'
R6 = CACHE / 'denoise-x20-r6-2026-09-25'
A2 = CACHE / 'denoise-x20-a2-2026-09-26'
ITER = CACHE / 'denoise-iterations-2026-09-25'
COLUMNS = ['rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'title', 'text', 'quoted_text', 'context_parent',
           'lexical_words']
NEGATION = r"(?i)\b(?:not|no|never|none|nothing|nobody|neither|nor|without|can't|cannot|won't|don't|doesn't|didn't|isn't|wasn't|aren't|haven't|hasn't|n't)\b"
TOKEN = r"(?-i:[A-Z][A-Z0-9]{3,}|[A-Z0-9]*[0-9][A-Z0-9]*[A-Z][A-Z0-9]*)"
# Information only: structures behind the final review's J items and the residual census, counted in the final set.
# Every pattern of an entry must match. None of these is a selection rule.
RESIDUAL_PROXIES = {
    'code_colon_token': (rf"(?i)\b(?:promo|discount|coupon|referral|offer|affiliate|my)?\s*codes?\s*(?:is|:|=|-|—)\s*:?\s*[\"“'‘]?{TOKEN}\b",),
    'use_code_token': (rf"(?i)\b(?:use|enter|apply|try)\s+(?:my\s+)?(?:promo\s+|discount\s+|coupon\s+|referral\s+)?code\s*:?\s*[\"“'‘]?{TOKEN}\b",),
    'app_builder_recruiting': (r"(?i)\b(?:i|we)(?:'ve| have|'m| am)? (?:just |recently )?(?:built|created|developed|made|building|creating|developing|launched|launching) (?:a|an|my|our|this)\b[^.!?\n]{0,80}\bapp\b",
                               r"(?i)\b(?:beta|testers?|early access|waitlist|wait list|share access|sign ?up|try it out|free access|premium)\b"),
    'disclosed_ai_paste': (r"(?i)\b(?:i )?asked (?:chat ?gpt|chatgpt|gpt|claude|gemini|copilot|grok|perplexity)\b|\b(?:chat ?gpt|chatgpt|gemini|grok) (?:said|says|told me|tells me|gave me|suggested)\b|\baccording to (?:chat ?gpt|chatgpt|gemini)\b",),
    'not_affiliated_disclaimer': (r"(?i)\bnot (?:affiliated|sponsored)\b|\bno affiliation\b|\b(?:this is )?not an ad\b",),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def representatives(final_ids):
    """Removed copies/duplicates from r6 and a2 with their kept representative (structure only)."""
    parts = [
        pl.read_parquet(R6 / 'body_groups.parquet').filter(pl.col('rowid') != pl.col('representative_rowid'))
        .select(pl.col('rowid').cast(pl.Int64), layer=pl.lit('r6'), relation=pl.lit('duplicate_body_text'),
                representative=pl.col('representative_rowid').cast(pl.Int64)),
        pl.read_parquet(R6 / 'paragraph_copies.parquet')
        .select(pl.col('rowid').cast(pl.Int64), layer=pl.lit('r6'), relation=pl.lit('contained_paragraph_copy'),
                representative=pl.col('host_rowid').cast(pl.Int64)),
        pl.read_parquet(A2 / 'near_duplicates.parquet')
        .select(pl.col('rowid').cast(pl.Int64), layer=pl.lit('a2'), relation=pl.lit('near_duplicate'),
                representative=pl.col('rep_rowid').cast(pl.Int64)),
        pl.read_parquet(A2 / 'copy_owners.parquet').explode('owners')
        .select(pl.col('rowid').cast(pl.Int64), layer=pl.lit('a2'), relation=pl.lit('sentence_copy'),
                representative=pl.col('owners').cast(pl.Int64)),
    ]
    rep = pl.concat(parts).unique()
    return rep.with_columns(representative_in_final=pl.col('representative').is_in(final_ids)).sort('rowid', 'representative')


def export(version, out):
    if out.exists():
        raise SystemExit(f'{out} exists; the export never overwrites an earlier product')
    manifest_path = version / 'manifest.json'
    man = json.loads(manifest_path.read_text())
    (out / 'records').mkdir(parents=True)
    totals = dict(records=0, characters=0, posts=0, comments=0, with_digits=0, with_negation=0, text_mismatches=0)
    files, final_ids = [], []
    for shard in sorted((version / 'decisions').glob('part-*.parquet')):
        dec = pl.read_parquet(shard, columns=['rowid', 'decision', 'version_decision', 'characters'])
        keep = dec.filter(pl.col('decision') == 'candidate').select('rowid', 'characters')
        base = pl.read_parquet(BASE / 'candidates' / shard.name, columns=COLUMNS)
        rec = base.join(keep, on='rowid', how='inner').sort('rowid')
        if rec.height != keep.height:
            raise SystemExit(f'{shard.name}: {keep.height - rec.height} selected rowids missing from the base candidates')
        path = out / 'records' / shard.name
        rec.write_parquet(path, compression='zstd')
        back = pl.read_parquet(path, columns=['rowid', 'text']).join(base.select('rowid', base_text='text'), on='rowid')
        totals['text_mismatches'] += int((back['text'] != back['base_text']).sum())
        totals['records'] += rec.height
        totals['characters'] += int(rec['characters'].sum() or 0)
        totals['posts'] += int((rec['kind'] == 'posts').sum())
        totals['comments'] += int((rec['kind'] == 'comments').sum())
        totals['with_digits'] += int(rec['text'].str.contains(r'\d').sum())
        totals['with_negation'] += int(rec['text'].str.contains(NEGATION).sum())
        final_ids.append(rec['rowid'])
        files.append(dict(path=f'records/{shard.name}', rows=rec.height, sha256=digest(path)))
    final_ids = pl.concat(final_ids)
    rep = representatives(final_ids)
    rep.write_parquet(out / 'representatives.parquet', compression='zstd')
    files.append(dict(path='representatives.parquet', rows=rep.height, sha256=digest(out / 'representatives.parquet')))
    expected = man['counts']['candidates']
    checks = dict(count_matches_version_manifest=totals['records'] == expected,
                  characters_match_version_manifest=totals['characters'] == man['counts']['characters'],
                  unique_rowids=final_ids.n_unique() == final_ids.len(),
                  texts_identical_to_base=totals['text_mismatches'] == 0)
    if not all(checks.values()):
        raise SystemExit(f'verification failed: {checks}')
    schema = {k: str(v) for k, v in pl.read_parquet(out / files[0]['path']).schema.items()}
    info = dict(
        product='Reddit GLP-1 strict subset (~1,000,000 records)', created_at=datetime.now(timezone.utc).isoformat(),
        selection=f"{version}/decisions: decision == 'candidate' (the strict layer's final decision; version_decision is the inherited a2 decision and is not used)",
        layers='base x10 candidates -> r6 (x=20 frozen baseline) -> a1 -> a2 (structure/redundancy) -> strict subset (lossy selection toward ~1,000,000; not denoising)',
        revision=man.get('revision'), extra_thresholds=man['extra_thresholds'], precedence=man['precedence'],
        counts=dict(totals, token_proxy=round(totals['characters'] / 4)), files=files, schema=schema,
        representative_relations=rep.group_by('layer', 'relation').agg(pl.len().alias('rows'),
                                                                        pl.col('representative_in_final').sum().alias('representative_in_final'))
        .sort('layer', 'relation').to_dicts(),
        sources={'base_candidates': str(BASE / 'candidates'), 'r6_manifest_sha256': digest(R6 / 'manifest.json'),
                 'a2_manifest_sha256': digest(A2 / 'manifest.json'), 'strict_manifest_sha256': digest(manifest_path),
                 'code_sha256': digest(Path(__file__))},
        verification=checks,
        caveats=['Lossy selection: most removed records are normal discussion, not noise.',
                 "'N' in reviews means readable normal content, not research relevance or highest quality.",
                 'Texts are copied unchanged from the base candidates (numbers, units and negations intact); nothing is rewritten.',
                 'No drug, symptom, first-person, topic or subreddit admission conditions; no random trimming; no paid APIs.'])
    (out / 'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=1) + '\n')
    (out / 'README.md').write_text('\n'.join([
        '# Reddit GLP-1 strict subset (~1,000,000 records)', '',
        f"Records: {totals['records']:,}; characters: {totals['characters']:,} (token proxy ~{round(totals['characters'] / 4):,}); "
        f"posts {totals['posts']:,}, comments {totals['comments']:,}.", '',
        '- `records/part-*.parquet`: one row per kept record, text unchanged from the base candidates. Columns: ' + ', '.join(schema) + '.',
        '- `representatives.parquet`: removed copies/duplicates (r6 body duplicates and contained paragraphs; a2 near-duplicates and '
        'sentence copies) with their kept representative and whether that representative is in this set.',
        '- `manifest.json`: selection rule, thresholds, counts, sha256 of every file, source manifests and verification.', '',
        "Selected by the strict version's final `decision == 'candidate'`. This is a lossy selection on top of the x=20 baseline: "
        'most removed records are normal discussion. Review label N means readable normal content, not highest quality.',
        'Load: `polars.scan_parquet("records/*.parquet")`.', '']))
    return info


def facts(text):
    return re.findall(r'\d+(?:[.,]\d+)?', text), len(re.findall(NEGATION, text)), len(text)


def review_labels(rnd):
    return {r['id']: r for f in sorted((ITER / f'round-{rnd}').glob('labels-*.jsonl')) for r in map(json.loads, f.read_text().splitlines())}


def round_items(rowids):
    """'第 N 轮 #id label' for rowids that were drawn in an earlier independent review round (read there in full)."""
    out = []
    for d in sorted(ITER.glob('round-*')):
        sample = {r['rowid']: r['id'] for r in map(json.loads, (d / 'sample.jsonl').read_text().splitlines())}
        hit = sorted(sample[r] for r in rowids if r in sample)
        if hit:
            labels = review_labels(d.name.split('-')[1])
            out += [f"第 {int(d.name.split('-')[1])} 轮 #{i} {labels[i]['label']}" for i in hit]
    return out


def census_text(key, c, brief=False):
    """One census entry: for versions whose whole net removal was read, new reads + earlier census examples + earlier
    review-round items and their union; otherwise the read count and whether it was every new hit or a sample."""
    labels = '，'.join(f'{lab} {v}' for lab, v in c['labels'].items())
    name = key.split('_')[0]
    if 'net_removed' not in c:
        return f"{name} {c['records']} 条 {c['labels']}" if brief else f"{name} {'全部' if c['scope'] == 'full' else '抽读（非全量）'} {c['records']} 条：{labels}"
    parts = [f"新审 {c['new_reads']} 条"] + ([f"普查已读例证 {c['census_examples']} 条"] if c['census_examples'] else [])
    text = f"{name} 净减 {c['net_removed']} 条全部读过：" + ' + '.join(parts) + f"（{labels}）"
    if c['seen_in_rounds']:
        text += f" + 此前审查轮已读 {len(c['seen_in_rounds'])} 条（" + '、'.join(c['seen_in_rounds']) + '）'
    return text + f"，并集 {c['union']} 条"


def validate(version, export_dir, rounds=('08', '09'), examples=('t3_1ti5a7e',)):
    dec = pl.scan_parquet(version / 'decisions' / 'part-*.parquet').select('rowid', 'decision', 'version_decision', 'reason').collect()
    recs = pl.scan_parquet(export_dir / 'records' / 'part-*.parquet')
    ids = recs.select('rowid').collect()
    chosen = dec.filter(pl.col('decision') == 'candidate').select('rowid')
    removed = dec.filter((pl.col('version_decision') == 'candidate') & (pl.col('decision') != 'candidate')).select('rowid')
    out = dict(exported=ids.height, decision_candidates=chosen.height,
               version_decision_candidates=int((dec['version_decision'] == 'candidate').sum()),
               export_equals_decision_set=ids['rowid'].sort().equals(chosen['rowid'].sort()),
               strict_removed_in_export=ids.join(removed, on='rowid', how='semi').height, samples={})
    for rnd in rounds:
        sample = [json.loads(line) for line in (ITER / f'round-{rnd}' / 'sample.jsonl').read_text().splitlines()]
        got = dict(recs.filter(pl.col('rowid').is_in([r['rowid'] for r in sample])).select('rowid', 'text').collect().iter_rows())
        labels, gone = review_labels(rnd), [r for r in sample if r['rowid'] not in got]
        why = dict(dec.filter(pl.col('rowid').is_in([r['rowid'] for r in gone])).select('rowid', 'reason').iter_rows())
        out['samples'][rnd] = dict(
            sample=len(sample), in_export=len(got), text_identical=sum(got.get(r['rowid']) == r['text'] for r in sample),
            digits_negations_identical=sum(r['rowid'] in got and facts(got[r['rowid']]) == facts(r['text']) for r in sample),
            labels=dict(Counter(labels[r['id']]['label'] for r in sample)),
            removed_by_final=[dict(id=r['id'], fullname=r['fullname'], label=labels[r['id']]['label'], reason=why.get(r['rowid'])) for r in gone])
    # J classes come from the reviewer's written reasons (not a model), in this order: a named rule gap (a structure a generic
    # rule could catch but the current rules miss), a stealth vendor/product/app/program plug (also when written in AI
    # style), research or product-research recruitment, AI-style or pasted AI text, or another semantic call.
    j = [r for r in review_labels(rounds[-1]).values() if r['label'] == 'J']
    gap = [r['id'] for r in j if 'rule gap' in r['reason']]
    plug = [r['id'] for r in j if r['id'] not in gap and re.search(r'\bplug\b', r['reason'])]
    recruit = [r['id'] for r in j if r['id'] not in gap + plug and 'recruit' in r['reason']]
    ai = [r['id'] for r in j if r['id'] not in gap + plug + recruit and re.search(r'\bAI\b|ChatGPT', r['reason'])]
    out['final_review_j'] = dict(structural_rule_gap=len(gap), stealth_plug=len(plug), recruitment=len(recruit), ai_generated_or_pasted=len(ai),
                                 other_semantic=len(j) - len(gap) - len(plug) - len(recruit) - len(ai),
                                 ids=sorted(r['id'] for r in j), rule_gap_ids=sorted(gap))
    # Semi-join, not is_in: Polars 1.32.3 row-group pruning drops matches for a multi-value String is_in on scan_parquet.
    ex = pl.scan_parquet(BASE / 'candidates' / 'part-*.parquet').select('rowid', 'fullname') \
        .join(pl.LazyFrame({'fullname': list(examples)}), on='fullname', how='semi').collect()
    out['examples'] = ex.join(dec, on='rowid', how='left').select('fullname', 'decision', 'reason').to_dicts()
    flags = {k: pl.all_horizontal([pl.col('text').str.contains(p) for p in pats]) for k, pats in RESIDUAL_PROXIES.items()}
    hits = recs.select('rowid', 'fullname', 'text').with_columns(**flags).filter(pl.any_horizontal(list(flags))).drop('text').collect()
    hits.write_parquet(ITER / f'{version.name}-residual-proxy-hits.parquet')
    out['residual_proxies'] = dict({k: int(hits[k].sum()) for k in flags}, any=hits.height)
    out['generated_at'] = datetime.now(timezone.utc).isoformat()
    (ITER / f'{version.name}-validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    return out


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), c + h


def strict_overlaps(version):
    """Standalone and pairwise overlaps of strict hits among the records the strict layer saw."""
    frames = [pl.read_parquet(p).filter(pl.col('version_decision') == 'candidate') for p in sorted((version / 'decisions').glob('part-*.parquet'))]
    dec = pl.concat([f.select('rowid', 'characters', *[c for c in f.columns if c.startswith('hit_')]) for f in frames])
    hits = [c for c in dec.columns if c.startswith('hit_')]
    dec = dec.with_columns([pl.col(h).fill_null(False) for h in hits]).with_columns(n=pl.sum_horizontal(hits))
    pairs = {f'{a[4:]}&{b[4:]}': int((dec[a] & dec[b]).sum()) for i, a in enumerate(hits) for b in hits[i + 1:]}
    return dict(seen=dec.height, hit_by_any=int((dec['n'] > 0).sum()), hit_by_two_or_more=int((dec['n'] > 1).sum()), pairs=pairs)


STRICT_VERSIONS = ('denoise-x20-s2-2026-09-26', 'denoise-x20-s3-2026-09-26', 'denoise-x20-s4-2026-09-26',
                   'denoise-x20-s5-2026-09-26', 'denoise-x20-s6-2026-09-26', 'denoise-x20-s7-2026-09-26',
                   'denoise-x20-s8-2026-09-26', 'denoise-x20-s9-2026-09-26', 'denoise-x20-s10-2026-09-26')  # frozen strict versions, in order
# Full-text reads of each version's new promotion hits in the previous version's kept set: (labels file, filter field,
# 'full' when every new hit was read, 'sample' when only a sample was[, net-removed rowids file]). With the net-removed file
# the summary separates new reads, census examples read earlier and review-round items read earlier, and checks the union.
CENSUS = {'denoise-x20-s3-2026-09-26': ('s3-promotion-dev/s3-final-hit-labels.jsonl', None, 'full'),
          'denoise-x20-s4-2026-09-26': ('s4-promotion-dev/s4-hit-labels.jsonl', 'in_final_draft', 'full'),
          'denoise-x20-s5-2026-09-26': ('s5-promotion-dev/s5-hit-labels.jsonl', 'in_s5_removal', 'sample'),
          'denoise-x20-s6-2026-09-26': ('s6-promotion-dev/s6-hit-labels.jsonl', 'in_s6_removal', 'sample'),
          'denoise-x20-s7-2026-09-26': ('s7-promotion-dev/s7-hit-labels.jsonl', 'in_s7_removal', 'sample'),
          'denoise-x20-s8-2026-09-26': ('s8-promotion-dev/s8-hit-labels.jsonl', 'in_s8_removal', 'full', 's8-promotion-dev/s8-net-removed-parts.parquet'),
          'denoise-x20-s9-2026-09-26': ('s9-promotion-dev/s9-hit-labels.jsonl', 'in_s9_removal', 'full', 's9-promotion-dev/s9-net-removed-parts.parquet'),
          'denoise-x20-s10-2026-09-26': ('s10-referral-patch/patch-hit-labels.jsonl', None, 'full', 's10-referral-patch/s10-net-removed-parts.parquet')}
CENSUS_LEGEND = 'P solicitation, R research recruitment (raffle or registration code), B borderline, D ordinary discussion'
RESIDUAL_NOTE = {
    'denoise-x20-s3-2026-09-26': '抽读：码 token 两类几乎全是推荐码/分享码招揽（含少量研究招募码），app_builder_recruiting 抽 8 条均为开发者自荐；disclosed_ai_paste 约一半含本人经历，'
                                 'not_affiliated_disclaimer 多为正常产品经验，精度都不足以作规则。可规则化的已知残留约 50–80 条（<0.01%），按 Codex 07:39Z 不为低收益规则再开版本，记为残留代价。',
    'denoise-x20-s4-2026-09-26': '码 token 与应用招测试者的高精度形式已并入 s4（code_offer、app_builder）；剩下的 3 条码类命中逐条读过，都是非推广码（诊断码 E11.9、医院 340B 定价码、防骗帖引用的银行 Swift 码），按设计保留；'
                                 'disclosed_ai_paste 与 not_affiliated_disclaimer 只能语义判断，按 Codex 09:04Z 列为语义残留，不扩大筛选范围。',
    'denoise-x20-s7-2026-09-26': '第 11 轮结构漏项在 s6 保留集一次普查后，高精度形式已并入 s7（care team 话术的副词/人称变体、自有站“… is mine / my site + 外链”、'
                                 '“disclosure: it\'s my site / I built / I made …”、c0de 类混淆码、Use<TOKEN> 粘连码、“initial consultations are free”、“I have / here\'s a referral code”、'
                                 '“link/code … in my bio/profile”）；实测精度不足、未作规则的有：泛“free consultation”（9/15）、affiliate 链接/计划（约 1/15，多为批评讨论）、'
                                 'referral program 泛提及（约 7/15）、“no bloodwork/zoom”（约 1/12）、“use TOKEN at checkout”（2/3，仅 3 条）。第 11 轮 #490 这类只靠 affiliate/referral program '
                                 '字样的推广因此保留为残留；AI 风格通用回答、隐性品牌植入、引流与 SEO 种草只能语义判断。',
    'denoise-x20-s8-2026-09-26': '第 12 轮唯一结构漏项（#170 独立开发者 app 自荐）经 s7 保留集普查确认为可泛化结构，已并入 s8（开发者身份、“working on an app/tool”、'
                                 'app 分发话术+外链、TestFlight/beta testers+邀请）；实测不作规则的有：“get it (for) free”（0/20，均为费用讨论）、单数“a beta tester”（比喻）、'
                                 '无邀请的 TestFlight/beta 提及（3 条均为普通讨论）。第 12 轮其余 J 为 AI 风格通用回答与隐性品牌/产品/服务植入，只能语义判断。',
    'denoise-x20-s9-2026-09-26': '第 13 轮结构漏项与写法候选经 s8 保留集普查，高精度形式已并入 s9（linktr.ee/stan.store/payhip 等创作者变现链接、自家商品+Etsy 链接、'
                                 '“拍照算卡路里”的 app 推广模板、try the beta / why I built this）；实测不作规则的有：8 个以上 hashtag（6/10，含研究招募与普通帖）、3 条以上 YouTube 链接（2/15）、'
                                 '空格或拼写的域名（1/15）、“found a great tool”+链接（1/3）、无推销提示词的“拍照…卡路里”（普通讨论 2/15）。零宽字符串是聊天助手粘贴痕迹'
                                 '（抽读约 13/15 为 AI 生成），但不是广告，未作规则、交 Codex 决定；AI 口吻互动回复、隐性供应商/诊所/app 植入、“not affiliated / not an ad”式产品推荐只能语义判断。'
                                 '第 14 轮（s9 独立 1000）16 条 J 全为语义判断（AI 口吻通用回答或粘贴 10、隐性供应商/产品/app/项目植入 5、产品研发调研招募 1），由其反推的写法普查见上条；'
                                 '当时拟以 s9 为最终版，Codex 13:04Z 纠正推荐码口径后开 s10（见 s10）。',
    'denoise-x20-s10-2026-09-26': '第 14 轮 J 反推的 7 种价格/月费/无会员费/无额外费用/分期写法在 s9 保留集普查，精度 5%–45%，按 Codex 12:34Z 属正常讨论、不作规则；'
                                  '普查中读到的 7 条自有推荐码/推荐链接漏网按 Codex 13:04Z 做了一次合并的 referral 健壮性修补（匹配副本归一化与粘连拆分、save/saves、'
                                  '兑换动作+码 token+折扣表达、code 与冒号间短连接语、/ref/ 路径、明确折扣链接邀请+非 Reddit 外链），7 条全部由通用部件命中，'
                                  's9 保留集一次扫描另得 18 条（招揽 17、边界 1：广告点击得来的补剂公开折扣链接，非本人推荐链接，随修补删除、记为代价）。'
                                  '不作规则的：/s/ 等分享链接本身、价格本身、厂商储蓄卡/项目链接（以 card/program/savings 结尾的匹配剔除）、只报告用过的码（“I used code X … saved me $20”）。'
                                  '第 15 轮（s10 独立 1000）20 条 J 全为语义判断：隐性供应商/产品/app/服务植入 10（含名称被不可见字符 U+2064 拆开的 app 推荐 #421，只报告、未作规则；'
                                  '与第 14 轮 #604 同一供应商的 AI 口吻植入 #98；多条记录同句、作者未验证的小写 telehealth 推销模板 #129、#587）、AI 口吻通用回答或粘贴 9、'
                                  '无注册码/抽奖的临床试验招募通知 1（#242；严格层只删带注册码或抽奖的研究招募，“enrolling / clinical trial”字样也出现在本人经历中，如同轮 #960 家长讲孩子入组，标 N）。'
                                  's10 为最终版，余下语义内容交后续 Jev。',
}
# Census of structures behind the final review's rule-gap J items, over the final kept set (counting only; nothing filtered):
# (census file, reading of the result).
GAP_CENSUS = {'denoise-x20-s4-2026-09-26': (
    's4-gap-census/census.json',
    '只有自制产品/应用自荐达到可规则化精度；研究招募宽结构与诊所电话广告精度不足。可规则化合计约 150–220 条（约 0.02%），远低于第 9 轮样本对这类内容的比例估计'
    '（6/1000，Wilson 0.3–1.3%），残留为长尾，多数仍需语义判断；已作为合并反馈报告，未改规则、未开新版本。'),
    'denoise-x20-s9-2026-09-26': (
    's10-promotion-dev/census.json',
    '7 类写法实测精度都不足（5%–45%），未作规则。按 Codex 12:34Z，价格、月费、无会员费、无额外费用、透明收费、分期付款本身是正常讨论，单独或组合都不等同广告；'
    '多条记录同句（作者未验证）的小写供应商推销模板（“… starting at $99/mo with no membership fees”）属于模板复用，但核心句与真实用户的价格描述无法用结构区分、误删代价说不清，保留为 J 边界。'
    '普查中另读到 7 条自有推荐码/推荐链接漏网（“code to share: TOKEN”、“Usecode NAME”、“My code NAME50 saves”、/ref/ 与 /s/ 分享链接、“use TOKEN when ordering ($100 off)”、'
    '“link to the discount”），逐条回放 s9 无部件命中。当时按“近似变体、只能逐条加短语”列为残留，Codex 13:04Z 纠正：这些是可泛化的既有代码/招揽结构，已在 s10 做一次合并的 referral 规则健壮性修补（见 s10）；价格/会员费类写法仍不作规则。')}


NOISE_X10 = ('link_redirect_only', 'survey_link_only', 'social_template_only', 'moderation_or_bot_template', 'bot_command_only',
             'courtesy_template_only', 'media_without_text', 'link_or_mention_only', 'expanded_deleted_placeholder',
             'social_phrase_combination', 'no_readable_text', 'automation_template_only', 'terminal_automation_disclosure',
             'duplicate_long_text', 'duplicate_quick_repeat')
SCOPE = ('outside_english_scope', 'non_latin_script_scope', 'low_english_function_words', 'strict_non_english')


def removal_groups(base, r6, a2, final, audits):
    """Removed records split into noise/redundancy, mixed, lossy (mostly normal content) and scope, with the evidence for
    each call; the length floors and the strict information floor are lossy by design."""
    x10, beyond, a2n, st = base['reasons'], r6['net_beyond_floor'], a2['net_by_reason'], final['removed_by_reason']
    rec = lambda d, keys: sum((d[k]['records'] if isinstance(d[k], dict) else d[k]) for k in keys if k in d)  # noqa: E731
    floor20 = base['candidates'] - r6['candidates'] - sum(v['records'] for v in beyond.values())
    promo = '；'.join(census_text(k, c, brief=True) for k, c in audits.items() if k.endswith('_new_promotion_census'))
    return [
        dict(group='x10 机械噪声/机器人/模板/空文本/重复', nature='噪声', records=rec(x10, NOISE_X10), evidence='denoise-coverage-1 规则类别（已验收）'),
        dict(group='r6 复制段落/同文/机械与社交模板', nature='冗余/噪声', records=rec(beyond, [k for k in beyond if k not in SCOPE]), evidence='r6 净增规则'),
        dict(group='严格层推广/招揽', nature='噪声（招揽）', records=rec(st, ['strict_promotion']),
             evidence=f"s2 推广 30 条审计 {audits['s2_promotion_30']}；各版新增命中全文阅读：{promo}"),
        dict(group='严格层借用为主（复制句占多数）', nature='冗余', records=rec(st, ['strict_mostly_borrowed']), evidence='复制/模板句占原文多数'),
        dict(group='a1/a2 结构/冗余层', nature='混合', records=rec(a2n, [k for k in a2n if k not in SCOPE]),
             evidence=f"a2 新增命中 180 条审计 {audits['a2_new_hits_180']}"),
        dict(group='长度门槛 x10（<10 实词）', nature='有损（短内容，含社交语与正常短回复）', records=x10['below_min_lexical_words'], evidence='阈值删除；本汇总不含删除侧抽样'),
        dict(group='长度门槛 x20（10–19 实词）', nature='有损（短内容）', records=floor20, evidence='阈值删除（用户设定 x=20）；本汇总不含删除侧抽样'),
        dict(group='严格层原创信息不足（原创不同实词 <41）', nature='有损（多为正常讨论）', records=rec(st, ['strict_info_below']),
             evidence='随机 30 条删除样例 30/30 为正常内容（约 1/3 信息较弱）'),
        dict(group='非英语/范围外', nature='范围', evidence='语言/文字规则',
             records=rec(x10, ['outside_english_scope']) + rec(beyond, ['non_latin_script_scope']) + rec(a2n, ['low_english_function_words'])
             + rec(st, ['strict_non_english']))]


def summary(version, export_dir, rounds_status, out_dir):
    cache = lambda d: json.loads((CACHE / d / 'manifest.json').read_text())  # noqa: E731
    short = lambda name: name.split('-')[2]  # noqa: E731  (denoise-x20-s3-2026-09-26 -> s3)
    base, r6, a1, a2 = cache('denoise-x10-2026-09-25'), cache('denoise-x20-r6-2026-09-25'), cache('denoise-x20-a1-2026-09-26'), cache('denoise-x20-a2-2026-09-26')
    final = json.loads((version / 'manifest.json').read_text())
    superseded = [(name, cache(name)) for name in STRICT_VERSIONS[:STRICT_VERSIONS.index(version.name)]]
    prev_name, prev = superseded[-1]
    exp = json.loads((export_dir / 'manifest.json').read_text())
    layers = [dict(layer='source records', records=base['source_records'], characters=None),
              dict(layer='x=10 candidates', records=base['candidates'], characters=None),
              dict(layer='r6 (x=20 frozen baseline)', records=r6['candidates'], characters=r6['candidate_characters']),
              dict(layer='a1 (structure/redundancy)', records=a1['candidates'], characters=a1['candidate_characters']),
              dict(layer='a2 (structure/redundancy)', records=a2['candidates'], characters=a2['candidate_characters']),
              *[dict(layer=f'{short(name)} (strict, superseded)', records=m['counts']['candidates'], characters=m['counts']['characters'])
                for name, m in superseded],
              dict(layer=f"{final.get('revision')} (strict, final)", records=final['counts']['candidates'], characters=final['counts']['characters'])]
    for row in layers:
        row['token_proxy'] = round(row['characters'] / 4) if row['characters'] else None
        row['pct_of_r6_records'] = round(100 * row['records'] / r6['candidates'], 2)
        row['pct_of_r6_characters'] = round(100 * row['characters'] / r6['candidate_characters'], 2) if row['characters'] else None
    status = json.loads(Path(rounds_status).read_text())
    iter_dir = ITER
    count = lambda rows, key: dict(sorted(Counter(r[key] for r in rows).items()))  # noqa: E731
    censuses = {}
    for name in STRICT_VERSIONS[:STRICT_VERSIONS.index(version.name) + 1]:
        if name in CENSUS:
            path, field, scope, *net_path = CENSUS[name]
            rows = [r for r in map(json.loads, (iter_dir / path).read_text().splitlines()) if field is None or r[field]]
            entry = dict(records=len(rows), labels=count(rows, 'label'), scope=scope)
            if net_path:
                net, read = set(pl.read_parquet(iter_dir / net_path[0], columns=['rowid'])['rowid'].to_list()), {r['rowid'] for r in rows}
                examples = sum(1 for r in rows if r.get('known_example'))
                earlier = round_items(net - read)
                entry.update(net_removed=len(net), new_reads=len(rows) - examples, census_examples=examples,
                             seen_in_rounds=earlier, union=len(read) + len(earlier))
            censuses[f'{short(name)}_new_promotion_census'] = entry
    val = json.loads((iter_dir / f'{version.name}-validation.json').read_text())
    last = status['rounds'][-1]
    k, n = last['label_counts'], last['reviewed']
    final_validation = dict(
        round=last['round'], version=last['version'], seed=last['seed'], labels=k,
        j_rate_wilson95=[round(v, 4) for v in wilson(k['J'], n)], m_plus_s_upper95=round(wilson(k['M'] + k['S'], n)[1], 4),
        j_classes={c: val['final_review_j'].get(c, 0) for c in ('structural_rule_gap', 'stealth_plug', 'recruitment', 'ai_generated_or_pasted',
                                                                 'other_semantic', 'rule_gap_ids')},
        gap_census=(json.loads((iter_dir / GAP_CENSUS[version.name][0]).read_text())['classes'] if version.name in GAP_CENSUS else None),
        export_checks={c: val[c] for c in ('exported', 'decision_candidates', 'version_decision_candidates',
                                           'export_equals_decision_set', 'strict_removed_in_export')},
        samples={rnd: dict({c: s[c] for c in ('sample', 'in_export', 'text_identical', 'digits_negations_identical', 'labels')},
                           removed_by_final=[dict(id=r['id'], label=r['label'], reason=r['reason']) for r in s['removed_by_final']])
                 for rnd, s in val['samples'].items()},
        examples=val['examples'], residual_proxies=val['residual_proxies'])
    audits = {
        'a2_new_hits_180': count([json.loads(l) for l in (iter_dir / 'a2-new-hit-audit' / 'labels.jsonl').read_text().splitlines()], 'audit_class'),
        's1_hits_90': count([json.loads(l) for l in (iter_dir / 's1-hit-audit' / 'labels.jsonl').read_text().splitlines()], 'audit_class'),
        's2_promotion_30': count([json.loads(l) for l in (iter_dir / 'strict-comparison-audit' / 's2-promotion-labels.jsonl').read_text().splitlines()], 'audit_class'),
        **censuses,
    }
    delta_key = f'{short(prev_name)}_to_{short(version.name)}'
    data = dict(generated_at=datetime.now(timezone.utc).isoformat(), final_version=str(version), final_revision=final.get('revision'),
                superseded=[name for name, _ in superseded],
                layers=layers, a2_net_by_reason=a2['net_by_reason'], r6_net_beyond_floor=r6['net_beyond_floor'],
                strict_removed_by_reason=final['removed_by_reason'], strict_standalone_hits=final['standalone_hits'],
                strict_promotion_part_hits=final.get('promotion_part_hits'), strict_overlaps=strict_overlaps(version),
                **{delta_key: dict(records=prev['counts']['candidates'] - final['counts']['candidates'],
                                   characters=prev['counts']['characters'] - final['counts']['characters'])},
                export=dict(dir=str(export_dir), counts=exp['counts'], verification=exp['verification'],
                            representative_relations=exp['representative_relations']),
                timings=dict(r6=r6['metrics'], a1=a1['metrics'], a2=a2['metrics'], strict_final=final['metrics']),
                rounds=[dict(round=r['round'], version=r['version'], seed=r['seed'], reviewed=r['reviewed'], labels=r['label_counts'],
                             retained=r.get('retained')) for r in status['rounds']],
                audits=audits, final_validation=final_validation, removal_groups=removal_groups(base, r6, a2, final, audits),
                residual_examples=[dict(id=r['id'], label=r['label'], reason=r['reason']) for r in sorted(review_labels(f"{last['round']:02d}").values(),
                                                                                                       key=lambda r: r['id']) if r['label'] in ('J', 'S', 'U', 'M')],
                wording=['The strict layer is lossy selection on top of x=20, not denoising; most removed records are normal discussion.',
                         "Review label N means readable normal content, not research relevance or highest quality; zero findings in a sample do not prove the whole set is noise-free.",
                         'Vocabulary counts (distinct original content words) are a proxy for information, not a measure of relevance.'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'strict-final-summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    fmt = lambda v: '' if v is None else f'{v:,}' if isinstance(v, int) else str(v)  # noqa: E731
    lines = ['# 约 100 万条严格子集：最终汇总', '', f"生成：{data['generated_at']}。最终版本 `{version}`（{final.get('revision')}）；独立导出 `{export_dir}`（仅本机私有）。"
             + f"被取代的严格版本：{'、'.join(short(n) for n, _ in superseded)}（目录均保留、未覆盖）。", '',
             '## 各层计数', '', '| 层 | 记录 | 字符 | token 代理 | 占 r6 记录 % | 占 r6 字符 % |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['layer']} | {fmt(r['records'])} | {fmt(r['characters'])} | {fmt(r['token_proxy'])} | {r['pct_of_r6_records']} | {fmt(r['pct_of_r6_characters'])} |" for r in layers]
    lines += ['', '## 严格层删除原因（优先级顺序，互斥计数）', '', '| 原因 | 记录 | 字符 |', '|---|---:|---:|']
    lines += [f"| {k} | {v['records']:,} | {v['characters']:,} |" for k, v in final['removed_by_reason'].items()]
    ov = data['strict_overlaps']
    lines += ['', f"单规则命中（a2 保留集 {ov['seen']:,} 条内）：" + '，'.join(f'{k} {v:,}' for k, v in final['standalone_hits'].items())
              + f"；被 ≥2 条严格规则同时命中 {ov['hit_by_two_or_more']:,} 条；两两重叠：" + '，'.join(f'{k} {v:,}' for k, v in ov['pairs'].items() if v) + '。',
              f"推广结构分项命中：" + '，'.join(f'{k} {v:,}' for k, v in (final.get('promotion_part_hits') or {}).items()) + '。',
              f"{short(prev_name)} → {short(version.name)} 净减 {data[delta_key]['records']:,} 条、{data[delta_key]['characters']:,} 字符。各版新推广命中的全文阅读（{CENSUS_LEGEND}）："
              + '；'.join(census_text(k, c) for k, c in censuses.items()) + '。', '',
              '## 独立导出与保真', '',
              f"- 记录 {exp['counts']['records']:,}，字符 {exp['counts']['characters']:,}（token 代理约 {exp['counts']['token_proxy']:,}），帖子 {exp['counts']['posts']:,}、评论 {exp['counts']['comments']:,}。",
              f"- 选择依据：最终 `decision == 'candidate'`（不用继承的 `version_decision`）；校验：" + '，'.join(f'{k}={v}' for k, v in exp['verification'].items()) + '。',
              f"- 正文逐条回读与基础库比对，不一致 {exp['counts']['text_mismatches']} 条；含数字 {exp['counts']['with_digits']:,} 条、含否定词 {exp['counts']['with_negation']:,} 条，均为原文未改写。",
              '- 来源/代表映射（被删副本 → 保留代表，是否仍在最终集）：' + '；'.join(f"{r['layer']} {r['relation']} {r['rows']:,} 行，代表在最终集 {r['representative_in_final']:,}" for r in exp['representative_relations'])
              + '。代表不在最终集的副本，其内容在最终集中不再有保留代表（有损选择的代价之一）。', '',
              '## 逐条审查与审计', '', '| 轮 | 版本 | seed | 已审 | M | S | N | J | U |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['round']} | `{Path(r['version']).name}` | {r['seed']} | {r['reviewed']} | " + ' | '.join(str(r['labels'][k]) for k in 'MSNJU') + ' |' for r in data['rounds']]
    fv, ec = final_validation, final_validation['export_checks']
    jc = fv['j_classes']
    pct = lambda v: f'{100 * v:.1f}%'  # noqa: E731
    sample_line = lambda rnd, s: (f"第 {int(rnd)} 轮样本 {s['sample']} 条中 {s['in_export']} 条在导出中，正文逐字一致 {s['text_identical']} 条、数字与否定词序列一致 {s['digits_negations_identical']} 条"  # noqa: E731
                                  + (f"；被最终版删除 {len(s['removed_by_final'])} 条：" + '，'.join(f"#{r['id']}（{r['label']}，{r['reason']}）" for r in s['removed_by_final'])
                                     if s['removed_by_final'] else ''))
    lines += ['', '审计：' + '；'.join(f'{k} {v}' for k, v in audits.items() if k not in censuses) + '。', '',
              f"## 最终独立验收（第 {fv['round']} 轮）与导出核验", '',
              f"- 第 {fv['round']} 轮从最终版独立抽 1000 条（seed {fv['seed']}，排除全部已见 ID）逐条读全文：" + '，'.join(f'{c} {fv["labels"][c]}' for c in 'MSNJU')
              + f"；J 率 {pct(fv['labels']['J'] / 1000)}（Wilson 95% {pct(fv['j_rate_wilson95'][0])}–{pct(fv['j_rate_wilson95'][1])}），M+S 95% 上界 {pct(fv['m_plus_s_upper95'])}。"
              + (f"J 中审查时记为结构漏项 {jc['structural_rule_gap']} 条（" + '、'.join(f'#{i}' for i in jc['rule_gap_ids']) + '，普查见下条）；' if jc['structural_rule_gap'] else 'J 全为语义判断：')
              + f"AI 口吻通用回答或粘贴 AI 文本 {jc['ai_generated_or_pasted']} 条，隐性供应商/产品/app/项目植入 {jc['stealth_plug']} 条，"
              + f"调研/研究招募 {jc['recruitment']} 条，其他语义判断 {jc['other_semantic']} 条。",
              *([f"- 结构漏项普查（最终保留集内只计数、不筛选；结构从第 {fv['round']} 轮 J 反推，样本内命中是构造使然，精度只在未见命中上抽读估计；P 招揽、R 研究招募、B 边界、D 普通讨论）："
                 + '；'.join(f"{k} 命中 {v['hits']:,} 条，抽读 {v['read']} 条 " + '/'.join(f'{x}{n}' for x, n in v['labels'].items() if n)
                             + f"，精度 {pct(v['precision'])}（Wilson {pct(v['precision_wilson95'][0])}–{pct(v['precision_wilson95'][1])}），估计约 {v['est_true']} 条"
                             for k, v in fv['gap_census'].items())
                 + '。' + GAP_CENSUS[version.name][1]] if fv['gap_census'] else []),
              f"- 导出集合与最终 `decision == 'candidate'` 集合一致：{ec['export_equals_decision_set']}（{ec['exported']:,} 条；继承的 `version_decision` 为 {ec['version_decision_candidates']:,} 条，未使用）；被严格层删除却进入导出 {ec['strict_removed_in_export']} 条。",
              *[f'- {sample_line(rnd, s)}。' for rnd, s in sorted(fv['samples'].items())],
              '- 点名样例：' + '，'.join(f"{e['fullname']} → {e['decision']}（{e['reason']}）" for e in fv['examples']) + '。',
              f'- 残留结构普查（只计数、不参与筛选；命中 ID 在私有 {version.name}-residual-proxy-hits.parquet）：' + '，'.join(f'{c} {v:,}' for c, v in fv['residual_proxies'].items())
              + '。' + RESIDUAL_NOTE.get(version.name, ''), '',
              '## 噪声与有损删除（分开计）', '', '| 类别 | 性质 | 删除记录 | 依据 |', '|---|---|---:|---|',
              *[f"| {g['group']} | {g['nature']} | {g['records']:,} | {g['evidence']} |" for g in data['removal_groups']], '',
              '“有损”指删除的多为可读的正常内容（门槛选择的代价），不是噪声；“混合”层按审计既有噪声/冗余也有正常信息损失。', '',
              f"## 规则无法区分的残留示例（第 {fv['round']} 轮逐条理由）", '',
              *[f"- #{r['id']}（{r['label']}）：{r['reason']}" for r in data['residual_examples']], '',
              '## 方案选择说明', '',
              '- 选用“原创不同实词 ≥41”的依据是构念（扣除重复、复制句、模板句、社交套语与功能词后的不重复实词）与规则简单可审计，不是实测优越性；与纯长度、实词数、不同实词数、原创实词数各门槛的对照，以及 36 条方案差异小样本（3 组×12），见 `strict-subset-comparison.md`，都没有显示明显优势。',
              f"- 不同实词数只是信息量代理；N 只表示可读的正常内容，泛化建议、制度争论、购物推荐等仍会被标 N，不等于高信息或适合关系抽取；第 6–{fv['round']} 轮的高 N 比例不能证明这组选择最优。", '',
              '## 耗时与内存', '', '- ' + '；'.join(f'{k} {json.dumps(v, ensure_ascii=False)}' for k, v in data['timings'].items()), '',
              '## 说明', ''] + [f'- {w}' for w in data['wording']]
    (out_dir / 'strict-final-summary.md').write_text('\n'.join(lines) + '\n')
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='step', required=True)
    e = sub.add_parser('export')
    e.add_argument('--version', type=Path, required=True)
    e.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('validate')
    v.add_argument('--version', type=Path, required=True)
    v.add_argument('--export', type=Path, required=True)
    v.add_argument('--rounds', nargs='+', default=['08', '09'], help='review rounds whose samples are checked (the last one is the final review)')
    v.add_argument('--examples', nargs='+', default=['t3_1ti5a7e'], help='fullnames whose final decision is reported')
    s = sub.add_parser('summary')
    s.add_argument('--version', type=Path, required=True)
    s.add_argument('--export', type=Path, required=True)
    s.add_argument('--status', type=Path, default=Path('results/reddit-denoise-iterations-2026-09-25/status.json'))
    s.add_argument('--out', type=Path, default=Path('results/reddit-denoise-iterations-2026-09-25'))
    args = ap.parse_args()
    if args.step == 'export':
        info = export(args.version, args.out)
        print(json.dumps(dict(counts=info['counts'], verification=info['verification']), indent=1))
    elif args.step == 'validate':
        out = validate(args.version, args.export, tuple(args.rounds), tuple(args.examples))
        print(json.dumps({c: out[c] for c in ('export_equals_decision_set', 'strict_removed_in_export', 'final_review_j', 'examples', 'residual_proxies')}, indent=1))
    else:
        data = summary(args.version, args.export, args.status, args.out)
        print(json.dumps(dict(layers=data['layers'][-1], superseded=data['superseded']), indent=1))


if __name__ == '__main__':
    main()
