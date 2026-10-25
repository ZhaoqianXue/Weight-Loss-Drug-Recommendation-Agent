"""Round bookkeeping for iterative denoising: independent 1000-record samples and status files.

sample: seeded equal-probability draw from a refine version's retained set, excluding every
        rowid already reviewed (earlier rounds and the retained-corpus QA samples). Full text
        goes to a private .cache round directory; only IDs and aggregates leave it.
status: validates and counts labels per round from private labels-*.jsonl files and writes
        results/reddit-denoise-iterations-2026-09-25/status.json plus the Chinese status page.
        Any malformed, duplicate, foreign or mismatched label, a sample that disagrees with its
        ID file, or overlap between rounds is an error, never a silent overwrite.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path('.cache/reddit-cleaning/denoise-iterations-2026-09-25')
BASE = Path('.cache/reddit-cleaning/denoise-x10-2026-09-25')
QA = Path('.cache/reddit-cleaning/denoise-x10-qa-2026-09-25/review/samples.jsonl')
RESULTS = Path('results/reddit-denoise-iterations-2026-09-25')
DOC = Path('docs/reddit-denoise-iterations-2026-09-25.md')
LABELS = {'M': '明确机械噪声', 'S': '纯社交低信息', 'N': '正常可读内容', 'J': '需语义判断/上下文不足', 'U': '不确定'}
LABEL_FIELDS = ('id', 'rowid', 'fullname', 'label', 'reason')
NOTE_FIELDS = {'phase', 'new_rules', 'net_change', 'risks', 'next', 'boundary'}  # everything else is computed


def seen_rowids():
    seen = set()
    if QA.exists():
        seen |= {json.loads(l)['rowid'] for l in QA.read_text().splitlines()}
    for ids in [*ROOT.glob('round-*/sample-ids.json'), *ROOT.glob('void-*/sample-ids.json')]:  # voided draws stay excluded
        seen |= set(json.loads(ids.read_text())['rowids'])
    for audit in ROOT.rglob('*.jsonl'):  # hit audits, example views and external reviews were read too
        seen |= {row['rowid'] for row in map(json.loads, audit.read_text().splitlines()) if isinstance(row.get('rowid'), int)}
    return seen


def sample(round_no, version, seed, size):
    out = ROOT / f'round-{round_no:02d}'
    if (out / 'sample-ids.json').exists():
        raise ValueError('Round already sampled; samples are never redrawn')
    seen = seen_rowids()
    keep = (pl.scan_parquet(str(Path(version) / 'decisions' / '*.parquet')).filter(pl.col('decision') == 'candidate')
            .select('rowid').collect())
    pool = keep.filter(~pl.col('rowid').is_in(list(seen)))
    if pool.height < size:
        raise ValueError('Retained pool smaller than the sample size')
    picked = pool.sample(size, seed=seed).sort('rowid')
    if picked['rowid'].n_unique() != size or set(picked['rowid'].to_list()) & seen:
        raise ValueError('Sample must be unique and disjoint from every earlier sample')
    text = (pl.scan_parquet(str(BASE / 'candidates' / '*.parquet')).join(picked.lazy(), on='rowid')
            .select('rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'lexical_words', 'text', 'quoted_text', 'title',
                    'context_parent').collect())
    parents = (pl.scan_parquet(str(BASE / 'records' / '*.parquet')).select(pl.col('fullname').alias('context_parent'),
               pl.col('text').str.slice(0, 500).alias('parent_text'))
               .join(text.select('context_parent').unique().lazy(), on='context_parent').collect())
    rows = text.join(parents, on='context_parent', how='left').sort('rowid').with_row_index('id').to_dicts()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sample.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    ids = dict(round=round_no, version=str(version), seed=seed, size=len(rows), pool=pool.height, excluded_seen=len(seen),
               rowids=[r['rowid'] for r in rows], drawn_at=datetime.now(timezone.utc).isoformat(),
               version_manifest_sha256=hashlib.sha256((Path(version) / 'manifest.json').read_bytes()).hexdigest())
    (out / 'sample-ids.json').write_text(json.dumps(ids, indent=1))
    return ids


def read_sample(round_dir):
    """Sample rows and ID file, checked against each other."""
    ids = json.loads((round_dir / 'sample-ids.json').read_text())
    rows = [json.loads(line) for line in (round_dir / 'sample.jsonl').read_text().splitlines()]
    rowids = [r['rowid'] for r in rows]
    if [r['id'] for r in rows] != list(range(len(rows))):
        raise ValueError(f'{round_dir}: sample ids must run 0..n-1')
    if len(rows) != ids['size'] or rowids != ids['rowids']:
        raise ValueError(f'{round_dir}: sample.jsonl disagrees with sample-ids.json')
    if len(set(rowids)) != len(rows) or len({r['fullname'] for r in rows}) != len(rows):
        raise ValueError(f'{round_dir}: duplicate records in sample')
    return ids, rows


def labels(round_dir, rows):
    """Validated labels by sample id; duplicates, foreign ids and mismatches raise."""
    by_id = {r['id']: r for r in rows}
    found, where = {}, {}
    for path in sorted(round_dir.glob('labels-*.jsonl')):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            item, at = json.loads(line), f'{path.name}:{n}'
            missing = [k for k in LABEL_FIELDS if k not in item]
            if missing:
                raise ValueError(f'{at}: missing {missing}')
            ref = by_id.get(item['id'])
            if ref is None:
                raise ValueError(f"{at}: id {item['id']} is not in the sample")
            if (item['rowid'], item['fullname']) != (ref['rowid'], ref['fullname']):
                raise ValueError(f"{at}: rowid/fullname do not match sample id {item['id']}")
            if item['label'] not in LABELS:
                raise ValueError(f"{at}: invalid label {item['label']!r}")
            if not isinstance(item['reason'], str) or not item['reason'].strip():
                raise ValueError(f'{at}: empty reason')
            if item['id'] in found:
                raise ValueError(f"{at}: duplicate label for id {item['id']} (also {where[item['id']]})")
            found[item['id']], where[item['id']] = item, at
    return found


def status(notes_path):
    notes = json.loads(Path(notes_path).read_text()) if Path(notes_path).exists() else {}
    rounds, drawn = [], {}
    qa = {json.loads(line)['rowid'] for line in QA.read_text().splitlines()} if QA.exists() else set()
    voided = []
    for void_dir in sorted(ROOT.glob('void-*')):
        ids = json.loads((void_dir / 'sample-ids.json').read_text())
        drawn[void_dir.name] = set(ids['rowids'])
        voided.append(dict(name=void_dir.name, version=ids['version'], seed=ids['seed'], size=ids['size'],
                           reason=json.loads((void_dir / 'void.json').read_text())['reason']))
    for round_dir in sorted(ROOT.glob('round-*')):
        ids, rows = read_sample(round_dir)
        mine = set(ids['rowids'])
        for other, theirs in [('QA', qa), *drawn.items()]:
            if mine & theirs:
                raise ValueError(f'{round_dir.name} overlaps {other} sample')
        drawn[round_dir.name] = mine
        found = labels(round_dir, rows)
        complete = set(found) == {r['id'] for r in rows}
        counts = {k: sum(1 for v in found.values() if v['label'] == k) for k in LABELS}
        manifest = json.loads((Path(ids['version']) / 'manifest.json').read_text())
        spec = json.loads((Path(ids['version']) / 'run-spec.json').read_text())
        entry = dict(round=ids['round'], state='审查完成' if complete else '审查中', sample_path=str(round_dir / 'sample.jsonl'),
                     seed=ids['seed'], sample_size=ids['size'], reviewed=len(found), review_complete=complete, label_counts=counts,
                     version=ids['version'], policy_revision=manifest.get('policy_revision', manifest.get('revision')),
                     policy_sha256=hashlib.sha256(json.dumps(spec['policy'], sort_keys=True).encode()).hexdigest(),
                     code_sha256=spec['code_sha256'],
                     base_manifest_sha256=spec.get('base_manifest_sha256', spec.get('frozen_manifest_sha256', spec.get('version_manifest_sha256'))),
                     retained=manifest.get('candidates', manifest.get('counts', {}).get('candidates')),
                     retained_characters=manifest.get('candidate_characters', manifest.get('counts', {}).get('characters')),
                     token_proxy=manifest.get('candidate_token_proxy', round(manifest.get('counts', {}).get('characters', 0) / 4)),
                     run_seconds=manifest['metrics']['seconds'],
                     peak_tree_rss_mib=manifest['metrics'].get('peak_tree_rss_mib', manifest['metrics'].get('peak_rss_mib')))
        extra = notes.get(str(ids['round']), {})
        if set(extra) - NOTE_FIELDS:
            raise ValueError(f"notes for round {ids['round']} may not set {sorted(set(extra) - NOTE_FIELDS)}; those fields are computed")
        entry.update(extra)
        rounds.append(entry)
    state = dict(updated_at=datetime.now(timezone.utc).isoformat(), formal_min_lexical_words=20,
                 source='v2-strict cleaned.sqlite sha256 904267bb… via accepted run denoise-x10-2026-09-25 (no source rescan)',
                 label_scheme=LABELS, overall=notes.get('overall', {}), rounds=rounds, voided_samples=voided)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Reddit 基础去噪迭代状态（x=20）', '', f"更新：{state['updated_at']}。机器可读版：[status.json](../{RESULTS}/status.json)。",
             '逐条标签和全文只在 `.cache/reddit-cleaning/denoise-iterations-2026-09-25/` 私有目录；标签为代理（Claude）逐条阅读结果，不是人工金标准。', '',
             '## 标签定义（固定）', '',
             '- **M 明确机械噪声**：不理解讨论主题也能判定的噪声——机器人输出、版务/自动审核整段模板、逐字或近逐字群发的推广/推荐码/招募模板、占位/媒体空壳、编码乱码、逐字重复灌水、版务定期帖。记 M 必须有可复现的模板重复或结构证据；措辞各异的疑似软广或刷量活动记 J，不记 M。',
             '- **S 纯社交低信息**：整条只有致谢、祝贺、称赞、祝福、问候、同情、客服寒暄或鼓励，没有事实内容。属于轻度语义，只能用审阅过的整条模板（全文由模板单元组成）处理；以客套开头但后面有叙述的不算。',
             '- **N 正常可读内容**：可以独立理解、有内容的人类文本（经历、提问、建议、观点、信息、保险/药房/饮食/健身/衣着/闲聊等）。是否与研究相关不在这里判断。',
             '- **J 需语义判断/上下文不足**：可读，但离开父帖/线程或语义理解就无法判断是否有有效信息（依赖父帖的片段式回复、以引用他人为主、疑似软广与真实经历难以区分、混合语言等）。',
             '- **U 不确定**：怀疑是机械噪声但无法仅凭本条确认（如疑似模板或刷屏），必须用频次等证据复核，不能直接记为 N。',
             '- 研究无关、非第一人称、没有药名或症状，以及保险/健身/饮食等主题，本身都不构成 M。', '',
             '## 验收口径', '',
             '- 迭代对象：M，以及能用合规、可泛化的整条社交/客服模板处理的 S。新抽 1000 条里两者都为 0、且没有未解决的 U，才冻结规则，再用另一组独立 1000 条做最终验证。',
             '- 不能泛化、必须语义判断的 S、J、U 分别计数并附边界证据，不为凑零改标签；“规则可处理噪声为 0”不等于“语料质量问题为 0”。',
             '- 规则只能是整条覆盖的模板（起止锚点加有界变量槽）或结构规则；不设药名、症状、身份、第一人称或主题门槛。',
             '- 每轮版本是对已验收 x=10 候选集的决策覆盖层（decisions），不是独立文本库；正文通过 rowid 连接 denoise-x10 候选分片读取，最终冻结后再导出候选 Parquet。“字符数/4”只是粗略 token 代理，不是真实分词计数。', '',
             '| 轮次 | 状态 | 版本 | 保留 | 审查 | M 机械噪声 | S 纯社交 | N 正常 | J 需语义 | U 不确定 |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rounds:
        c = r['label_counts']
        lines.append(f"| {r['round']} | {r['state']} | {r['policy_revision']} | "
                     f"{r['retained']:,} | {r['reviewed']}/{r['sample_size']} | {c['M']} | {c['S']} | {c['N']} | {c['J']} | {c['U']} |")
    for r in rounds:
        lines += ['', f"## 第 {r['round']} 轮", '',
                  f"- 样本：seed {r['seed']}，{r['sample_size']} 条，已审查 {r['reviewed']} 条" + ('' if r['review_complete'] else '（未完成，计数不是最终结果）'),
                  f"- 版本：`{r['policy_revision']}`，保留 {r['retained']:,} 条，正文 {r['retained_characters']:,} 字符（约 {r['token_proxy'] / 1e6:.0f}M token 代理），运行 {r['run_seconds']:.0f} 秒，峰值 RSS {r['peak_tree_rss_mib']:.0f} MiB"]
        for key, label in (('phase', '进展'), ('new_rules', '规则'), ('net_change', '净减少'), ('boundary', '边界证据'), ('risks', '反例/风险'), ('next', '下一步')):
            if r.get(key):
                lines.append(f'- {label}：{r[key]}')
    if voided:
        lines += ['', '## 作废的抽样', ''] + [f"- `{v['name']}`（{v['version']}，seed {v['seed']}，{v['size']} 条）：{v['reason']}；这些 ID 仍计入已见样本，以后不会再抽到"
                                            for v in voided]
    if state['overall']:
        lines += ['', '## 总体', ''] + [f'- {k}：{v}' for k, v in state['overall'].items()]
    DOC.write_text('\n'.join(lines) + '\n')
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--round', type=int, required=True)
    s.add_argument('--version', type=Path, required=True)
    s.add_argument('--seed', type=int, required=True)
    s.add_argument('--size', type=int, default=1000)
    t = sub.add_parser('status')
    t.add_argument('--notes', type=Path, default=ROOT / 'notes.json')
    args = ap.parse_args()
    if args.command == 'sample':
        ids = sample(args.round, args.version, args.seed, args.size)
        print(json.dumps({k: v for k, v in ids.items() if k != 'rowids'}, indent=2))
    else:
        state = status(args.notes)
        print(json.dumps([{k: r[k] for k in ('round', 'reviewed', 'label_counts', 'retained')} for r in state['rounds']], ensure_ascii=False))


if __name__ == '__main__':
    main()
