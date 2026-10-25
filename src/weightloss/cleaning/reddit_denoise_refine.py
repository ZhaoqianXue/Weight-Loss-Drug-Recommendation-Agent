"""Iterative refinement over an accepted reddit_denoise run, without rescanning the source.

Reads the base run's candidate shards (text already extracted once), applies the
length floor and reviewed structural rules with vectorized Polars expressions, and
writes one compact decision shard per base shard. Rows excluded by the base run keep
their base decision; text is never copied except by an explicit final export.
Records whose own body (posts without their title) is identical to another retained
body after only Unicode NFC and whitespace normalization are grouped; one representative
is kept (a post before a comment, then the earliest), comment copies are excluded, and a
post copy is excluded only if its title is equal under the same normalization; any other
title difference keeps the post. Every group member and its title are written to
body_groups.parquet. A comment whose paragraph list (same normalization, in order, repeats
counted) is an ordered subsequence of one other retained record's paragraph list, linked
by a shared long paragraph, is a lossless copy and is excluded; posts are never removed
this way (paragraph_copies.parquet maps each copy to its host). Templates are complete-message rules: one fires only when its pattern, or a sequence of
its units, covers the whole normalized message, so any extra original content keeps the
record. No drug, symptom, identity, first-person or topic-relevance rule.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import unicodedata

import polars as pl

from . import reddit_basic as basic, reddit_denoise as denoise

READY = 'ready_for_semantic_screen'
URL = r'(?:https?://|www\.)[^\s<>]+'
LEXICAL = r"[\p{L}\p{N}]+(?:['’.][\p{L}\p{N}]+)*"
TEMPLATE_REASONS = {'mechanical': 'mechanical_template', 'social': 'social_template'}
UNBOUNDED = re.compile(r'\.[*+]|\{\d+,\}|\)[*+]')  # open wildcards or repeated groups would allow extra content


def validate_policy(policy):
    r = policy.get('refine')
    if policy.get('schema_version') != 1 or not isinstance(r, dict):
        raise ValueError('Invalid refine policy')
    if type(r['min_lexical_words']) is not int or r['min_lexical_words'] < 1:
        raise ValueError('Invalid length floor')
    ids = [t['id'] for t in r['templates']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate template ids')
    dup = r.get('body_duplicates')
    if dup is not None and (type(dup.get('min_body_words')) is not int or dup['min_body_words'] < 15):
        raise ValueError('Invalid body duplicate rule')
    par = r.get('paragraph_containment')
    if par is not None and (type(par.get('min_paragraph_words')) is not int or par['min_paragraph_words'] < 20):
        raise ValueError('Invalid paragraph containment rule')
    for t in r['templates']:
        parts = t['units'] if 'units' in t else [t.get('pattern')]
        if t.get('class') not in TEMPLATE_REASONS or ('units' in t) == ('pattern' in t) or not all(parts):
            raise ValueError(f"Invalid template {t['id']}")
        if any(UNBOUNDED.search(part) for part in parts):
            raise ValueError(f"Template {t['id']} has an unbounded span")


def template_regex(template):
    """Whole-surface regex: the pattern, or one or more units, from start to end."""
    if 'units' in template:
        unit = '|'.join(f'(?:{u})' for u in template['units'])
        return f'^(?:{unit})(?: (?:{unit}))*$'
    return f"^(?:{template['pattern']})$"


def surface(text):
    """Anchor surface: NFKC, case-folded words; URLs and punctuation dropped."""
    return (text.str.normalize('NFKC').str.to_lowercase().str.replace_all('’', "'")
            .str.replace_all(URL, ' ').str.replace_all(r"[^\w']+", ' ').str.replace_all(r'\s+', ' ').str.strip_chars())


GROUP_SCHEMA = dict(group=pl.UInt32, rowid=pl.Int64, fullname=pl.String, kind=pl.String, subreddit=pl.String,
                    created_utc=pl.Int64, title=pl.String, role=pl.String, representative_rowid=pl.Int64,
                    representative_fullname=pl.String, representative_kind=pl.String, representative_subreddit=pl.String,
                    copies=pl.UInt32)


COPY_SCHEMA = dict(rowid=pl.Int64, fullname=pl.String, kind=pl.String, subreddit=pl.String, created_utc=pl.Int64,
                   paragraphs=pl.UInt32, host_rowid=pl.Int64, host_fullname=pl.String, host_kind=pl.String,
                   host_subreddit=pl.String, component=pl.UInt32)
WORD = re.compile(r"[^\W_]+(?:['’.][^\W_]+)*")
URL_RE = re.compile(URL)


def own_body(text, title, kind):
    return text[len(title):] if kind == 'posts' and title and text.startswith(title) else text


def exact_key(body):
    """Equality key: Unicode NFC and whitespace only, so case, digits, decimal points,
    hyphens, units and negations still separate records (1.5 mg != 1-5 mg != 15 mg)."""
    return ' '.join(unicodedata.normalize('NFC', body).split())


def paragraph_list(text):
    """Blank-line paragraphs under exact_key, in order and with repeats; empty ones dropped."""
    return [k for k in (exact_key(p) for p in text.split('\n\n')) if k]


def ordered_within(needle, hay):
    """True when needle is an ordered subsequence of hay: each hay paragraph matches at most once."""
    rest = iter(hay)
    return all(any(p == h for h in rest) for p in needle)


def body_features(frame, min_paragraph_words=None):
    """Own body (posts lose their leading title), its lexical word count and prefilter hashes.

    Hashes drop all whitespace, which is coarser than exact_key, so equal texts always share
    them; they only select rows for the exact comparison. long_paragraphs holds one hash per
    blank-line paragraph with at least min_paragraph_words words."""
    title = pl.col('title').fill_null('')
    body = (pl.when((pl.col('kind') == 'posts') & pl.col('text').str.starts_with(title))
            .then(pl.col('text').str.slice(title.str.len_chars())).otherwise(pl.col('text')))
    key = body.str.normalize('NFC').str.replace_all(r'[\s\x1c-\x1f]+', '')
    post_body = pl.when(pl.col('kind') == 'posts').then(body).otherwise(pl.lit(''))  # comments reuse the base word count
    count = pl.when(pl.col('kind') == 'posts').then(post_body.str.replace_all(URL, ' ').str.count_matches(LEXICAL)).otherwise(pl.col('lexical_words'))
    frame = frame.with_columns(body_words=count.cast(pl.Int32), body_hash=key.hash(11, 13, 17, 19))
    if not min_paragraph_words:
        return frame.with_columns(long_paragraphs=pl.lit([], dtype=pl.List(pl.UInt64)))
    paragraphs = pl.col('text').str.split('\n\n').list.eval(pl.element().str.strip_chars())
    return frame.with_columns(long_paragraphs=paragraphs.list.eval(
        pl.element().filter(pl.element().str.replace_all(URL, ' ').str.count_matches(LEXICAL) >= min_paragraph_words)
        .str.normalize('NFC').str.replace_all(r'[\s\x1c-\x1f]+', '').hash(23, 29, 31, 37)))


def body_groups(stage, base_dir, min_words):
    """Groups of retained rows with identical bodies, every member with its role.

    Representative: a post (its title adds context) before a comment, then the earliest.
    Comment copies are removed. A post copy is removed only when its title is also equal
    under exact_key; any other title difference keeps it (counted, never judged equivalent)."""
    pool = stage.filter((pl.col('decision') == 'candidate') & (pl.col('body_words') >= min_words))
    hot = pool.group_by('body_hash').len().filter(pl.col('len') > 1).select('body_hash')
    rows = pool.join(hot, on='body_hash', how='semi').select('rowid').collect()
    if not rows.height:
        return pl.DataFrame(schema=GROUP_SCHEMA)
    texts = (pl.scan_parquet(str(base_dir / 'candidates' / '*.parquet')).select('rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'title', 'text')
             .join(rows.lazy(), on='rowid').collect().sort('rowid'))
    members = {}
    for r in texts.iter_rows(named=True):
        r['body'] = own_body(r['text'], r['title'], r['kind'])
        members.setdefault(exact_key(r['body']), []).append(r)
    out = []
    for number, group in enumerate(g for _, g in sorted(members.items()) if len(g) > 1):
        group.sort(key=lambda r: (r['kind'] != 'posts', r['created_utc'], r['rowid']))
        rep = group[0]
        for r in group:
            differs = r['kind'] == 'posts' and exact_key(r['title'] or '') != exact_key(rep['title'] or '')
            role = 'representative' if r is rep else 'kept_title_differs' if differs else 'removed'
            out.append(dict(group=number, rowid=r['rowid'], fullname=r['fullname'], kind=r['kind'], subreddit=r['subreddit'],
                            created_utc=r['created_utc'], title=r['title'], role=role, representative_rowid=rep['rowid'],
                            representative_fullname=rep['fullname'], representative_kind=rep['kind'],
                            representative_subreddit=rep['subreddit'], copies=len(group)))
    return pl.DataFrame(out, schema=GROUP_SCHEMA) if out else pl.DataFrame(schema=GROUP_SCHEMA)


def paragraph_copies(pool, base_dir, min_words):
    """Comments whose paragraph list is an ordered subsequence of one other retained record.

    Records are linked by shared long paragraphs (hashes and paragraph sets only preselect);
    within each component the longest paragraph lists are kept first, so a comment is removed
    only if one kept record holds all of its paragraphs verbatim (NFC + whitespace) in the same
    order and at least as many times. Posts always stay."""
    exploded = pool.select('rowid', 'long_paragraphs').explode('long_paragraphs').drop_nulls()
    shared = exploded.group_by('long_paragraphs').agg(pl.col('rowid').n_unique().alias('n')).filter(pl.col('n') > 1)
    rows = exploded.join(shared.select('long_paragraphs'), on='long_paragraphs', how='semi').select('rowid').unique().collect()
    if not rows.height:
        return pl.DataFrame(schema=COPY_SCHEMA)
    texts = (pl.scan_parquet(str(base_dir / 'candidates' / '*.parquet')).select('rowid', 'fullname', 'kind', 'subreddit', 'created_utc', 'text')
             .join(rows.lazy(), on='rowid').collect().sort('rowid'))
    records, by_paragraph = {}, {}
    for r in texts.iter_rows(named=True):
        paragraphs = paragraph_list(r['text'])
        keys = set(paragraphs)
        records[r['rowid']] = dict(r, paragraphs=paragraphs, keys=keys)
        for k in keys:
            if len(WORD.findall(URL_RE.sub(' ', k))) >= min_words:
                by_paragraph.setdefault(k, []).append(r['rowid'])
    parent = {i: i for i in records}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for ids in by_paragraph.values():
        for j in ids[1:]:
            parent[find(j)] = find(ids[0])
    components = {}
    for i in records:
        components.setdefault(find(i), []).append(i)
    out = []
    for number, ids in enumerate(sorted(components.values(), key=min)):
        kept = []
        for i in sorted(ids, key=lambda i: (-len(records[i]['paragraphs']), records[i]['kind'] != 'posts', records[i]['created_utc'], i)):
            r = records[i]
            host = next((records[k] for k in kept if r['keys'] <= records[k]['keys']
                         and ordered_within(r['paragraphs'], records[k]['paragraphs'])), None) if r['kind'] == 'comments' else None
            if host is None:
                kept.append(i)
                continue
            out.append(dict(rowid=i, fullname=r['fullname'], kind=r['kind'], subreddit=r['subreddit'], created_utc=r['created_utc'],
                            paragraphs=len(r['paragraphs']), host_rowid=host['rowid'], host_fullname=host['fullname'],
                            host_kind=host['kind'], host_subreddit=host['subreddit'], component=number))
    return pl.DataFrame(out, schema=COPY_SCHEMA).sort('rowid') if out else pl.DataFrame(schema=COPY_SCHEMA)


def rules(frame, policy):
    """Vectorized rule flags and first-applying reason for base candidates."""
    r = policy['refine']
    text = pl.col('text')
    flags = [f'_template_{i}' for i in range(len(r['templates']))]
    hits = []
    for flag, t in zip(flags, r['templates']):
        hit = pl.col('_surface').str.contains(template_regex(t)) & pl.col('kind').is_in(t.get('kinds', ['comments']))
        if t.get('subreddits'):
            hit = hit & pl.col('subreddit').str.to_lowercase().is_in(t['subreddits'])
        hits.append(hit.alias(flag))
    frame = frame.with_columns(_surface=surface(text)).with_columns(hits).drop('_surface')  # surface computed once
    template = kind = pl.lit(None, dtype=pl.String)
    for flag, t in reversed(list(zip(flags, r['templates']))):  # first listed template wins
        template = pl.when(pl.col(flag)).then(pl.lit(t['id'])).otherwise(template)
        kind = pl.when(pl.col(flag)).then(pl.lit(TEMPLATE_REASONS[t['class']])).otherwise(kind)
    normalized = text.str.normalize('NFKC')
    letters = normalized.str.count_matches(r'\p{L}')
    latin = normalized.str.count_matches(r'\p{Latin}')
    script = (letters >= r['script']['min_letters']) & ((letters - latin) / letters >= r['script']['non_latin_share'])
    tokens = text.str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(LEXICAL)
    repetition = (tokens.list.len() >= r['repetition']['min_tokens']) & (tokens.list.n_unique() <= r['repetition']['max_unique'])
    floor = pl.col('lexical_words') < r['min_lexical_words']
    out = frame.with_columns(hit_template=template, template_reason=kind, hit_script=script, hit_repetition=repetition,
                             hit_floor=floor, characters=text.str.len_chars())
    reason = (pl.when(pl.col('hit_template').is_not_null()).then(pl.col('template_reason'))
              .when(pl.col('hit_script')).then(pl.lit('non_latin_script_scope'))
              .when(pl.col('hit_repetition')).then(pl.lit('token_repetition_spam'))
              .when(pl.col('hit_floor')).then(pl.lit('below_min_lexical_words'))
              .otherwise(pl.lit(READY)))
    detail = (pl.when(pl.col('hit_template').is_not_null()).then(pl.col('hit_template'))
              .when(pl.col('hit_floor') & ~pl.col('hit_script') & ~pl.col('hit_repetition'))
              .then(pl.lit(json.dumps(dict(min_lexical_words=r['min_lexical_words']))))
              .otherwise(None))
    return out.with_columns(reason=reason, detail=detail).with_columns(
        decision=pl.when(pl.col('reason') == READY).then(pl.lit('candidate')).otherwise(pl.lit('exclude'))).drop(flags)


def run(base_dir, output_dir, policy, export_candidates=False):
    validate_policy(policy)
    started = time.perf_counter()
    base_dir, output_dir = Path(base_dir).resolve(), Path(output_dir).resolve()
    base = json.loads((base_dir / 'manifest.json').read_text())
    if base['status'] != 'full_denoise_complete':
        raise ValueError('Base run incomplete')
    if output_dir == base_dir or output_dir.is_relative_to(base_dir):
        raise ValueError('Refinement must not write into the base run')
    code = {Path(p).name: basic.file_digest(p) for p in (__file__, denoise.__file__)}
    spec = dict(schema_version=1, base_run=str(base_dir), base_manifest_sha256=basic.file_digest(base_dir / 'manifest.json'),
                base_run_spec_sha256=basic.file_digest(base_dir / 'run-spec.json'), policy=policy, code_sha256=code)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'run-spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Policy/code/base changed; use a new version directory')
    if not path.exists():
        basic.write_json(path, spec)
    (output_dir / 'source').mkdir(exist_ok=True)
    for p in (__file__, denoise.__file__):
        shutil.copyfile(p, output_dir / 'source' / Path(p).name)
    shards = sorted((base_dir / 'candidates').glob('part-*.parquet'))
    stage, target = output_dir / 'stage', output_dir / 'decisions'
    stage.mkdir(exist_ok=True)
    target.mkdir(exist_ok=True)
    written = 0
    columns = ['rowid', 'decision', 'reason', 'detail', 'lexical_words', 'characters', 'kind',
               'hit_template', 'hit_script', 'hit_repetition', 'hit_floor']
    min_body = policy['refine'].get('body_duplicates', {}).get('min_body_words')
    min_paragraph = policy['refine'].get('paragraph_containment', {}).get('min_paragraph_words')
    with denoise._TreeMemory() as memory:
        for shard in shards:  # pass 1: per-record rules plus body keys
            out = stage / shard.name
            if out.exists():
                continue
            frame = pl.read_parquet(shard, columns=['rowid', 'kind', 'subreddit', 'created_utc', 'lexical_words', 'text', 'title'])
            decided = body_features(rules(frame, policy), min_paragraph).select(*columns, 'created_utc', 'body_words', 'body_hash',
                                                                              'long_paragraphs')
            temp = out.with_suffix('.tmp')
            decided.write_parquet(temp, compression='zstd')
            temp.replace(out)
            written += 1
        group_path = output_dir / 'body_groups.parquet'
        if not group_path.exists():  # pass 2: corpus-wide identical bodies among retained rows
            groups = (body_groups(pl.scan_parquet(str(stage / '*.parquet')), base_dir, min_body) if min_body
                      else pl.DataFrame(schema=GROUP_SCHEMA))
            groups.write_parquet(group_path.with_suffix('.tmp'), compression='zstd')
            group_path.with_suffix('.tmp').replace(group_path)
        dups = (pl.read_parquet(group_path).filter(pl.col('role') == 'removed')
                .select('rowid', pl.col('representative_rowid').alias('duplicate_of_rowid')))
        copy_path = output_dir / 'paragraph_copies.parquet'
        if not copy_path.exists():  # pass 2b: lossless paragraph copies among what remains
            pool = (pl.scan_parquet(str(stage / '*.parquet')).filter(pl.col('decision') == 'candidate')
                    .join(dups.lazy().select('rowid'), on='rowid', how='anti'))
            copies = paragraph_copies(pool, base_dir, min_paragraph) if min_paragraph else pl.DataFrame(schema=COPY_SCHEMA)
            copies.write_parquet(copy_path.with_suffix('.tmp'), compression='zstd')
            copy_path.with_suffix('.tmp').replace(copy_path)
        copies = pl.read_parquet(copy_path, columns=['rowid', 'host_rowid'])
        for shard in shards:  # pass 3: final decisions
            out = target / shard.name
            if out.exists():
                continue
            decided = (pl.read_parquet(stage / shard.name).join(dups, on='rowid', how='left').join(copies, on='rowid', how='left')
                       .with_columns(hit_body_duplicate=pl.col('duplicate_of_rowid').is_not_null(),
                                     hit_paragraph_copy=pl.col('host_rowid').is_not_null()))
            decided = decided.with_columns(
                reason=pl.when(pl.col('hit_body_duplicate')).then(pl.lit('duplicate_body_text'))
                .when(pl.col('hit_paragraph_copy')).then(pl.lit('contained_paragraph_copy')).otherwise(pl.col('reason')),
                detail=pl.when(pl.col('hit_body_duplicate')).then(pl.format('{{"representative_rowid": {}}}', pl.col('duplicate_of_rowid')))
                .when(pl.col('hit_paragraph_copy')).then(pl.format('{{"host_rowid": {}}}', pl.col('host_rowid')))
                .otherwise(pl.col('detail')),
                decision=pl.when(pl.col('hit_body_duplicate') | pl.col('hit_paragraph_copy')).then(pl.lit('exclude'))
                .otherwise(pl.col('decision')))
            temp = out.with_suffix('.tmp')
            decided.select(*columns, 'hit_body_duplicate', 'hit_paragraph_copy').write_parquet(temp, compression='zstd')
            temp.replace(out)
        decisions = pl.scan_parquet(str(target / '*.parquet'))
        counts = dict(decisions.group_by('reason').len().collect().iter_rows())
        kinds = dict(decisions.filter(pl.col('decision') == 'candidate').group_by('kind').len().collect().iter_rows())
        chars = decisions.filter(pl.col('decision') == 'candidate').select(pl.col('characters').cast(pl.Int64).sum()).collect().item()
        flags = decisions.select(*(pl.col(c).is_not_null().sum() if c == 'hit_template' else pl.col(c).sum()
                                   for c in ('hit_template', 'hit_script', 'hit_repetition', 'hit_floor',
                                             'hit_body_duplicate', 'hit_paragraph_copy'))).collect().to_dicts()[0]
        groups = pl.read_parquet(group_path)
        removed = groups.filter(pl.col('role') == 'removed')
        duplicate_summary = dict(min_body_words=min_body, equality='Unicode NFC + whitespace collapse; case, digits, punctuation kept',
                                 groups=groups['group'].n_unique(), members=groups.height, removed=removed.height,
                                 kept_title_differs=groups.filter(pl.col('role') == 'kept_title_differs').height,
                                 removed_by_kind=dict(removed.group_by('kind').len().sort('kind').iter_rows()),
                                 cross_subreddit=removed.filter(pl.col('subreddit') != pl.col('representative_subreddit')).height,
                                 cross_kind=removed.filter(pl.col('kind') != pl.col('representative_kind')).height)
        copies = pl.read_parquet(copy_path)
        copy_summary = dict(min_paragraph_words=min_paragraph, equality='ordered subsequence of one kept host paragraph list after NFC + whitespace collapse; '
                            'order and repeat counts kept',
                            removed=copies.height, hosts=copies['host_rowid'].n_unique(), components=copies['component'].n_unique(),
                            removed_by_kind=dict(copies.group_by('kind').len().sort('kind').iter_rows()),
                            cross_subreddit=copies.filter(pl.col('subreddit') != pl.col('host_subreddit')).height,
                            host_is_post=copies.filter(pl.col('host_kind') == 'posts').height)
        beyond = decisions.filter((pl.col('decision') == 'exclude') & ~pl.col('hit_floor'))
        net = {k: dict(records=n, characters=c) for k, n, c in beyond.group_by('reason').agg(
            pl.len(), pl.col('characters').cast(pl.Int64).sum()).sort('reason').collect().iter_rows()}
        templates = {k: dict(hits=n, net_beyond_floor=b) for k, n, b in decisions.filter(pl.col('hit_template').is_not_null())
                     .group_by('hit_template').agg(pl.len(), (~pl.col('hit_floor')).sum()).sort('hit_template').collect().iter_rows()}
        candidates_path = None
        if export_candidates:
            export = output_dir / 'candidates'
            export.mkdir(exist_ok=True)
            for shard in shards:
                keep = pl.read_parquet(target / shard.name, columns=['rowid', 'decision']).filter(pl.col('decision') == 'candidate')
                pl.read_parquet(shard).join(keep.select('rowid'), on='rowid').write_parquet(export / shard.name, compression='zstd')
            candidates_path = 'candidates/part-*.parquet'
    reasons = {k: v for k, v in base['reasons'].items() if k != READY}
    for k, v in counts.items():
        if k != READY:
            reasons[k] = reasons.get(k, 0) + v
    candidates = counts.get(READY, 0)
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(), status='refine_complete',
                  policy_revision=policy['revision'], min_lexical_words=policy['refine']['min_lexical_words'],
                  base_run=str(base_dir), base_policy_revision=base['policy_revision'], base_min_lexical_words=base['min_lexical_words'],
                  source_records=base['source_records'], candidates=candidates, excluded=base['source_records'] - candidates,
                  retention_pct=100 * candidates / base['source_records'], candidate_characters=chars,
                  candidate_token_proxy=chars / 4, token_proxy_note='characters / 4; rough proxy, not a tokenizer count',
                  candidates_by_kind=kinds, reasons=reasons, standalone_rule_hits_on_base_candidates=flags,
                  net_beyond_floor=net, template_hits=templates, body_duplicates=duplicate_summary,
                  paragraph_containment=copy_summary,
                  outputs=dict(decisions='decisions/part-*.parquet', body_groups='body_groups.parquet',
                               paragraph_copies='paragraph_copies.parquet', candidates=candidates_path,
                               text_source=f'{base_dir}/candidates/part-*.parquet (join on rowid)',
                               note='decisions are an overlay on the base candidates, not a separate text corpus; '
                                    'read retained text with scan_candidates() or export with --export-candidates'),
                  metrics=dict(seconds=time.perf_counter() - started, shards_written=written,
                               peak_tree_rss_mib=memory.peak_mib))
    basic.write_json(output_dir / 'manifest.json', report)
    return report


def scan_candidates(version_dir):
    """Lazy retained records with text: the exported shards, else the decision overlay joined to base text."""
    version_dir = Path(version_dir)
    manifest = json.loads((version_dir / 'manifest.json').read_text())
    if manifest['outputs'].get('candidates'):
        return pl.scan_parquet(str(version_dir / manifest['outputs']['candidates']))
    keep = pl.scan_parquet(str(version_dir / 'decisions' / '*.parquet')).filter(pl.col('decision') == 'candidate')
    return pl.scan_parquet(str(Path(manifest['base_run']) / 'candidates' / '*.parquet')).join(keep.select('rowid'), on='rowid', how='semi')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path('.cache/reddit-cleaning/denoise-x10-2026-09-25'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-denoise-refine.json'))
    parser.add_argument('--export-candidates', action='store_true')
    args = parser.parse_args()
    report = run(args.base_dir, args.output_dir, json.loads(args.policy.read_text()), args.export_candidates)
    print(json.dumps({k: v for k, v in report.items() if k not in ('reasons',)}, indent=2))


if __name__ == '__main__':
    main()
