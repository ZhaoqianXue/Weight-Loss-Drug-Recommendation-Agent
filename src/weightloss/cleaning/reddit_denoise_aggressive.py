"""Aggressive structural compression on top of a frozen refine version, without rescanning the source.

Only records the frozen version retained can be newly excluded; its own exclusions keep their
reasons. Rules, in precedence order (thresholds come from the policy, chosen by a corpus-wide
yield scan):
  near_duplicate            MinHash word-trigram similarity to its LSH group's earliest retained
                            record >= threshold (the earliest record stays; mapping written).
  copy_compression          share of words in >=5-word sentences that already appeared in an
                            earlier retained record >= threshold (earliest owners stay).
  boilerplate_template      words in sentences found in >= K retained records cover >= share of the
                            record and the other words are <= a residual limit.
  title_only_post           a post whose own body (title and URLs removed) has < N words.
  emoji_dominant            at least 3 emoji and emoji per word >= threshold.
  low_english_function_words  >= 20 words and English function-word share < threshold
                            (non-English prose, numeric tables, keyword lists).
  non_english_prose         >= 20 words, English function-word share < threshold and Spanish/
                            Portuguese/French/Italian/German function-word share >= threshold.
  title_dominant_post       a post whose own body (title and URLs removed) has < N words.
  social_dominant           generalized social expressions (congratulation, praise, thanks, wishes,
                            encouragement, sympathy, greeting) make up >= share of the non-function
                            words and <= N other content words remain (light semantic judgement).
  low_content_words         <= N content words after removing English function words and social
                            expressions: a residual content-word threshold reported on its own; it
                            removes mostly ordinary short discussion, not noise.
near_duplicate additionally requires the removed record to have exactly the representative's set of
number tokens and negation words, so dose or negation variants are never merged.
Rules absent from the policy are disabled; title_only_post, emoji_dominant and
low_english_function_words were set aside after hit review (tiny yield, real content lost).
The first three are lossy redundancy compression and the third also a light semantic judgement
by repeated structure; none is pure noise removal. Sentence equality is lowercase, whitespace and
trimmed outer punctuation only, so digits, decimal points and negations still separate sentences.
No drug, symptom, first-person, topic or subreddit condition; retained text is never modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import time

import polars as pl

from . import reddit_basic as basic
from .reddit_denoise_refine import LEXICAL as LEX, READY, URL

SENTENCE = r'(?:[^.!?\n]|\.\d)+[.!?]*'  # a decimal point does not end a sentence
FUNCTION = ('the|and|to|of|a|i|is|it|you|that|in|my|for|on|this|was|with|but|have|be|not|are|so|just|me|at|do|if|or|what|'
            'can|about|all|get|like|had|they|we|one|as|when|from|your|an|he|she|will|would|there|been|has|out|up|no|how|'
            'more|now|am|did|because|then|still|than|them|some|any|also|much|only|really|very|im|its|dont')
FOREIGN = ('de|que|el|la|los|las|y|en|un|una|por|para|con|es|lo|se|mi|pero|como|más|muy|está|tengo|estoy|com|não|uma|os|'
           'da|dos|das|eu|isso|mais|muito|também|je|et|le|les|des|est|pas|pour|avec|une|il|che|di|sono|und|der|das|ich|'
           'nicht|ist|mit|ein|eine')
NEGATION = "not|no|never|nothing|none|without|cannot|can't|cant|don't|dont|didn't|didnt|doesn't|doesnt|won't|wont|isn't|isnt|wasn't|wasnt|haven't|havent|hasn't|hasnt|aren't|arent|nor"
SOCIAL_PHRASES = ('keep it up|keep up the (?:good|great|amazing|awesome|hard) work|keep (?:going|pushing)|hang in there|'
                  "you(?:'ve| have)? got this|you can do (?:it|this)|don'?t give up|stay (?:strong|positive)|one day at a time|"
                  'proud of (?:you|yourself)|(?:so )?happy for you|well done|(?:great|good|nice|amazing|awesome) (?:job|work)|way to go|'
                  'you look (?:so )?(?:amazing|great|fantastic|incredible|beautiful|gorgeous|stunning|awesome|good|wonderful|fabulous|'
                  'healthy|happy|younger)|looking (?:good|great|amazing|fantastic)|good luck|best of luck|best wishes|all the best|'
                  'sending (?:hugs|love|prayers|good vibes)|thank you so much|thanks so much|thanks? (?:you )?for sharing|thank you|'
                  'congrats on|congratulations on|you should be (?:so |very )?proud|you deserve (?:it|this)|love (?:this|that|it)|'
                  "that'?s (?:amazing|awesome|great|incredible|wonderful|fantastic)|this is (?:amazing|awesome|great|incredible)|"
                  "you'?re (?:doing )?(?:great|amazing|awesome|so well|killing it|crushing it|rocking it)|doing (?:great|amazing|awesome)|"
                  "good for you|go you|let'?s go|hell yeah|get it girl|yes girl|so inspiring|what an? (?:amazing|awesome|incredible|great) "
                  '(?:achievement|accomplishment|transformation|progress|job|journey)')
SOCIAL_WORDS = ('congrats|congratulations|congratz|grats|thanks|thank|thx|ty|tysm|appreciate|appreciated|amazing|awesome|fantastic|'
                'incredible|wonderful|gorgeous|stunning|beautiful|proud|inspiring|inspiration|inspirational|inspired|impressive|'
                'brilliant|fabulous|wow|yay|woohoo|woo|omg|lol|lmao|haha|hahaha|bravo|kudos|hugs|prayers|blessed|bless|welcome|'
                'sorry|luck|cheers|rooting|rockstar|glowing|stoked|thrilled|yass|yasss|yesss|wooo')
MINHASH, BANDS = 20, 5
REASONS = ('near_duplicate', 'copy_compression', 'boilerplate_template', 'title_only_post', 'title_dominant_post', 'emoji_dominant',
           'low_english_function_words', 'non_english_prose', 'social_dominant', 'low_content_words')
KEYS = {'near_duplicate': ('min_similarity',), 'copy_compression': ('min_share', 'min_sentence_words'),
        'boilerplate_template': ('min_records', 'min_share', 'max_residual_words'), 'title_only_post': ('max_body_words',),
        'emoji_dominant': ('min_emoji', 'min_per_word'), 'low_english_function_words': ('min_words', 'max_share'),
        'non_english_prose': ('min_words', 'max_english_share', 'min_foreign_share'), 'title_dominant_post': ('max_body_words',),
        'social_dominant': ('min_share', 'max_content_words'), 'low_content_words': ('max_content_words',)}


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def validate_policy(policy):
    a = policy.get('aggressive')
    if not isinstance(a, dict) or not a or set(a) - set(KEYS):
        raise ValueError('Invalid aggressive policy')
    for rule, rule_policy in a.items():
        if set(rule_policy) != set(KEYS[rule]) or not all(isinstance(rule_policy[k], (int, float)) for k in KEYS[rule]):
            raise ValueError(f'Invalid {rule} rule')
    if not 0.5 <= a.get('near_duplicate', {}).get('min_similarity', 1) <= 1 or not 0.3 <= a.get('copy_compression', {}).get('min_share', 1) <= 1:
        raise ValueError('Similarity/share thresholds out of range')


def retained(base_dir, frozen_dir, shard):
    keep = (pl.read_parquet(frozen_dir / 'decisions' / shard.name, columns=['rowid', 'decision'])
            .filter(pl.col('decision') == 'candidate'))
    return (pl.read_parquet(shard, columns=['rowid', 'kind', 'created_utc', 'lexical_words', 'text', 'title'])
            .join(keep.select('rowid'), on='rowid', how='semi'))


def tokens():
    return pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(LEX)


def features(frame):
    text, title = pl.col('text'), pl.col('title').fill_null('')
    body = (pl.when((pl.col('kind') == 'posts') & text.str.starts_with(title)).then(text.str.slice(title.str.len_chars()))
            .otherwise(text))
    return frame.select(
        pl.col('rowid').cast(pl.UInt32), 'kind', chars=text.str.len_chars(),
        body_words=pl.when(pl.col('kind') == 'posts').then(body.str.replace_all(URL, ' ').str.count_matches(LEX))
        .otherwise(pl.col('lexical_words')).cast(pl.Int32),
        n_tok=tokens().list.len().cast(pl.Int32),
        fn_n=text.str.to_lowercase().str.count_matches(rf'\b(?:{FUNCTION})\b').cast(pl.Int32),
        foreign_n=text.str.to_lowercase().str.replace_all(URL, ' ').str.count_matches(rf'\b(?:{FOREIGN})\b').cast(pl.Int32),
        **social_counts(),
        emoji_n=text.str.count_matches(r'\p{Extended_Pictographic}').cast(pl.Int32))


def social_counts():
    """Words in generalized social expressions and the content words left after them and function words."""
    low = pl.col('text').str.to_lowercase().str.replace_all(URL, ' ')
    rest = low.str.replace_all(rf'\b(?:{SOCIAL_PHRASES})\b', ' ')
    n_all, n_rest = low.str.count_matches(LEX), rest.str.count_matches(LEX)
    soc_rest = rest.str.count_matches(rf'\b(?:{SOCIAL_WORDS})\b')
    return dict(social_n=((n_all - n_rest) + soc_rest).cast(pl.Int32),
                content_n=(n_rest - rest.str.count_matches(rf'\b(?:{FUNCTION})\b') - soc_rest).cast(pl.Int32))


def order_key():
    return pl.col('created_utc').cast(pl.UInt64) * (1 << 24) + pl.col('rowid').cast(pl.UInt64)


def sentences(frame):
    s = (frame.select(pl.col('rowid').cast(pl.UInt32), key=order_key(),
                      piece=pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(SENTENCE))
         .explode('piece')
         .with_columns(norm=pl.col('piece').str.replace_all(r'^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$', '').str.replace_all(r'\s+', ' '))
         .with_columns(sw=pl.col('norm').str.count_matches(LEX).cast(pl.UInt16))
         .filter(pl.col('sw') > 0))
    return (s.select('rowid', 'key', shash=pl.col('norm').hash(41, 43, 47, 53), sw='sw')
            .unique(subset=['rowid', 'shash'], keep='first', maintain_order=True))


def signatures(frame):
    tri = tokens().list.eval(pl.concat_str([pl.element(), pl.element().shift(-1), pl.element().shift(-2)], separator=' ').drop_nulls())
    sh = (frame.select(pl.col('rowid').cast(pl.UInt32), key=order_key(), tri=tri)
          .explode('tri').drop_nulls('tri').with_columns(h=pl.col('tri').hash(7, 11, 13, 17)).drop('tri'))
    sig = sh.group_by('rowid').agg(pl.col('key').first(),
                                   **{f'm{i}': (pl.col('h').hash(1000 + i) % 4294967296).min().cast(pl.UInt32) for i in range(MINHASH)})
    marks = frame.select(pl.col('rowid').cast(pl.UInt32),  # number and negation sets must match for a near-duplicate
                         numbers=tokens().list.eval(pl.element().filter(pl.element().str.contains(r'\d'))).list.unique().list.sort()
                         .list.join(' ').hash(5),
                         negations=pl.col('text').str.to_lowercase().str.extract_all(rf'\b(?:{NEGATION})\b').list.unique().list.sort()
                         .list.join(' ').hash(6))
    sig = sig.join(marks, on='rowid', how='left')
    rows = MINHASH // BANDS
    return sig.with_columns(**{f'b{b}': pl.struct([f'm{b * rows + j}' for j in range(rows)]).hash(99) for b in range(BANDS)})


def near_duplicates(work):
    """Each LSH group member is compared with the group's earliest record; best match per member."""
    scan = pl.scan_parquet(str(work / 'minhash' / '*.parquet'))
    pairs = pl.concat([
        scan.select('rowid', 'key', band=pl.col(f'b{b}')).group_by('band')
        .agg(n=pl.len(), rep=pl.col('rowid').sort_by('key').first(), members=pl.col('rowid'))
        .filter(pl.col('n') > 1).select('rep', 'members').explode('members')
        .filter(pl.col('members') != pl.col('rep')).rename({'members': 'rowid'}).collect(engine='streaming')
        for b in range(BANDS)]).unique()
    if not pairs.height:
        return pl.DataFrame(schema=dict(rowid=pl.UInt32, sim=pl.Float64, rep_rowid=pl.UInt32))
    ids = pl.concat([pairs['rowid'], pairs['rep']]).unique().to_frame('rowid')
    sig = (scan.join(ids.lazy(), on='rowid', how='semi')
           .select('rowid', 'numbers', 'negations', *[f'm{i}' for i in range(MINHASH)]).collect())
    both = (pairs.join(sig, on='rowid').join(sig, left_on='rep', right_on='rowid', suffix='_rep')
            .filter((pl.col('numbers') == pl.col('numbers_rep')) & (pl.col('negations') == pl.col('negations_rep'))))
    sim = pl.sum_horizontal([(pl.col(f'm{i}') == pl.col(f'm{i}_rep')).cast(pl.UInt8) for i in range(MINHASH)]) / MINHASH
    return (both.select('rowid', 'rep', sim=sim).group_by('rowid')
            .agg(pl.col('sim').max(), rep_rowid=pl.col('rep').sort_by('sim', descending=True).first()).sort('rowid'))


def docstats(work, shard, stats, min_records, min_sentence_words):
    s = pl.read_parquet(work / 'sentences' / shard.name).join(stats, on='shash', how='left').with_columns(
        docs=pl.col('docs').fill_null(1), copied=pl.col('owner').is_not_null() & (pl.col('owner') != pl.col('key'))
        & (pl.col('sw') >= min_sentence_words))
    w = pl.col('sw').cast(pl.Int32)
    owners = (s.filter(pl.col('copied')).group_by('rowid', 'owner').agg(words=w.sum())
              .sort(['rowid', 'words', 'owner'], descending=[False, True, False]).group_by('rowid')
              .agg(owners=(pl.col('owner') % (1 << 24)).cast(pl.UInt32).head(3)))  # main sources first
    return s.group_by('rowid').agg(
        total=w.sum(), copied=w.filter(pl.col('copied')).sum(), frequent=w.filter(pl.col('docs') >= min_records).sum()
    ).join(owners, on='rowid', how='left')


def flags(frame, a):
    c = pl.col
    share = lambda x, y: c(x).cast(pl.Float64) / pl.max_horizontal(c(y), pl.lit(1)).cast(pl.Float64)  # noqa: E731
    make = {
        'near_duplicate': lambda r: c('sim').fill_null(0) >= r['min_similarity'],
        'copy_compression': lambda r: share('copied', 'total') >= r['min_share'],
        'boilerplate_template': lambda r: (share('frequent', 'total') >= r['min_share']) & ((c('total') - c('frequent')) <= r['max_residual_words']),
        'title_only_post': lambda r: (c('kind') == 'posts') & (c('body_words') < r['max_body_words']),
        'emoji_dominant': lambda r: (c('emoji_n') >= r['min_emoji']) & (share('emoji_n', 'n_tok') >= r['min_per_word']),
        'low_english_function_words': lambda r: (c('n_tok') >= r['min_words']) & (share('fn_n', 'n_tok') < r['max_share']),
        'non_english_prose': lambda r: (c('n_tok') >= r['min_words']) & (share('fn_n', 'n_tok') < r['max_english_share'])
        & (share('foreign_n', 'n_tok') >= r['min_foreign_share']),
        'title_dominant_post': lambda r: (c('kind') == 'posts') & (c('body_words') < r['max_body_words']),
        'social_dominant': lambda r: (c('social_n').cast(pl.Float64) / pl.max_horizontal(c('social_n') + c('content_n'), pl.lit(1))
                                      >= r['min_share']) & (c('content_n') <= r['max_content_words']),
        'low_content_words': lambda r: c('content_n') <= r['max_content_words']}
    return {name: make[name](a[name]) for name in REASONS if name in a}


def run(base_dir, frozen_dir, output_dir, policy):
    validate_policy(policy)
    started = time.perf_counter()
    base_dir, frozen_dir, output_dir = Path(base_dir).resolve(), Path(frozen_dir).resolve(), Path(output_dir).resolve()
    frozen = json.loads((frozen_dir / 'manifest.json').read_text())
    if frozen.get('status') != 'refine_complete':
        raise ValueError('Frozen version incomplete')
    if output_dir in (base_dir, frozen_dir) or output_dir.is_relative_to(base_dir) or output_dir.is_relative_to(frozen_dir):
        raise ValueError('Aggressive version must be a new directory outside the base run and the frozen version')
    spec = dict(schema_version=1, base_run=str(base_dir), frozen_version=str(frozen_dir),
                frozen_manifest_sha256=basic.file_digest(frozen_dir / 'manifest.json'), policy=policy,
                code_sha256={Path(__file__).name: basic.file_digest(__file__)})
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'run-spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Policy/code/frozen version changed; use a new version directory')
    if not path.exists():
        basic.write_json(path, spec)
    (output_dir / 'source').mkdir(exist_ok=True)
    shutil.copyfile(__file__, output_dir / 'source' / Path(__file__).name)
    work, target = output_dir / 'work', output_dir / 'decisions'
    for d in ('features', 'sentences', 'minhash'):
        (work / d).mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=True)
    a = policy['aggressive']
    shards = sorted((base_dir / 'candidates').glob('part-*.parquet'))
    timings = {}
    t = time.perf_counter()
    for shard in shards:  # pass 1: numeric features, sentence hashes, MinHash signatures (no text copied)
        outs = [work / d / shard.name for d in ('features', 'sentences', 'minhash')]
        if all(o.exists() for o in outs):
            continue
        frame = retained(base_dir, frozen_dir, shard)
        for o, fn in zip(outs, (features, sentences, signatures)):
            fn(frame).write_parquet(o.with_suffix('.tmp'), compression='zstd')
            o.with_suffix('.tmp').replace(o)
    timings['features_sentences_minhash'] = time.perf_counter() - t
    t = time.perf_counter()
    stats_path, near_path = work / 'sentence_stats.parquet', output_dir / 'near_duplicates.parquet'
    if not stats_path.exists():  # pass 2: corpus-wide sentence document counts and earliest owners
        (pl.scan_parquet(str(work / 'sentences' / '*.parquet')).group_by('shash')
         .agg(docs=pl.len().cast(pl.UInt32), owner=pl.col('key').min()).filter(pl.col('docs') > 1)
         .collect(engine='streaming').write_parquet(stats_path, compression='zstd'))
    if not near_path.exists():  # pass 3: LSH near-duplicate candidates verified by signature agreement
        near_duplicates(work).write_parquet(near_path, compression='zstd')
    timings['sentence_stats_near_duplicates'] = time.perf_counter() - t
    t = time.perf_counter()
    stats, near = pl.read_parquet(stats_path), pl.read_parquet(near_path)
    for shard in shards:  # pass 4: decisions overlay on the frozen version
        out = target / shard.name
        if out.exists():
            continue
        feats = (pl.read_parquet(work / 'features' / shard.name)
                 .join(docstats(work, shard, stats, a.get('boilerplate_template', {}).get('min_records', 3),
                                a.get('copy_compression', {}).get('min_sentence_words', 5)),
                       on='rowid', how='left')
                 .join(near, on='rowid', how='left')
                 .with_columns(pl.col(['total', 'copied', 'frequent']).fill_null(0)))
        f = flags(feats, a)
        reason, detail = pl.lit(None, dtype=pl.String), pl.lit(None, dtype=pl.String)
        for name in reversed([n for n in REASONS if n in f]):  # first listed reason wins
            reason = pl.when(f[name]).then(pl.lit(name)).otherwise(reason)
        detail = (pl.when(reason == 'near_duplicate').then(pl.format('{{"rep_rowid": {}, "similarity": {}}}', 'rep_rowid', 'sim'))
                  .when(reason == 'copy_compression').then(pl.format('{{"copied_words": {}, "words": {}}}', 'copied', 'total'))
                  .when(reason == 'boilerplate_template').then(pl.format('{{"template_words": {}, "words": {}}}', 'frequent', 'total'))
                  .otherwise(None))
        new = feats.select(pl.col('rowid').cast(pl.Int64), new_reason=reason, new_detail=detail,
                           **{f'hit_{n}': e for n, e in f.items()})
        frozen_rows = pl.read_parquet(frozen_dir / 'decisions' / shard.name,
                                      columns=['rowid', 'decision', 'reason', 'detail', 'characters', 'kind', 'lexical_words'])
        decided = frozen_rows.join(new, on='rowid', how='left').with_columns(
            decision=pl.when(pl.col('new_reason').is_not_null()).then(pl.lit('exclude')).otherwise(pl.col('decision')),
            reason=pl.coalesce('new_reason', 'reason'), detail=pl.coalesce('new_detail', 'detail'),
            frozen_decision=pl.col('decision')).drop('new_reason', 'new_detail')
        (work / 'copy_owners').mkdir(exist_ok=True)
        feats.filter(f.get('copy_compression', pl.lit(False))).select(pl.col('rowid').cast(pl.Int64), 'copied', 'total', 'owners').write_parquet(
            work / 'copy_owners' / shard.name, compression='zstd')
        decided.write_parquet(out.with_suffix('.tmp'), compression='zstd')
        out.with_suffix('.tmp').replace(out)
    timings['decisions'] = time.perf_counter() - t
    pl.scan_parquet(str(work / 'copy_owners' / '*.parquet')).collect().write_parquet(output_dir / 'copy_owners.parquet', compression='zstd')
    decisions = pl.scan_parquet(str(target / '*.parquet'))
    keep = decisions.filter(pl.col('decision') == 'candidate')
    candidates, chars = keep.select(pl.len(), pl.col('characters').cast(pl.Int64).sum()).collect().row(0)
    newly = decisions.filter((pl.col('frozen_decision') == 'candidate') & (pl.col('decision') == 'exclude'))
    net = {k: dict(records=n, characters=c) for k, n, c in newly.group_by('reason').agg(
        pl.len(), pl.col('characters').cast(pl.Int64).sum()).sort('reason').collect().iter_rows()}
    standalone = decisions.filter(pl.col('frozen_decision') == 'candidate').select(
        *[pl.col(f'hit_{n}').sum().alias(n) for n in REASONS if n in a]).collect().row(0, named=True)
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(), status='aggressive_complete',
                  policy_revision=policy['revision'], enabled_rules=[n for n in REASONS if n in a], frozen_version=str(frozen_dir), frozen_policy_revision=frozen['policy_revision'],
                  min_lexical_words=frozen['min_lexical_words'], source_records=frozen['source_records'],
                  frozen_candidates=frozen['candidates'], frozen_characters=frozen['candidate_characters'],
                  candidates=candidates, candidate_characters=chars, candidate_token_proxy=chars / 4,
                  token_proxy_note='characters / 4; rough proxy, not a tokenizer count',
                  candidates_by_kind=dict(keep.group_by('kind').len().collect().iter_rows()),
                  removed_beyond_frozen=dict(records=frozen['candidates'] - candidates, characters=frozen['candidate_characters'] - chars),
                  net_by_reason=net, standalone_hits_on_frozen_candidates=standalone,
                  near_duplicate_pairs_verified=near.height,
                  outputs=dict(decisions='decisions/part-*.parquet', near_duplicates='near_duplicates.parquet', copy_owners='copy_owners.parquet',
                               text_source=f'{base_dir}/candidates/part-*.parquet (join on rowid)',
                               note='decisions overlay the frozen version; retained text is never modified or copied'),
                  metrics=dict(seconds=time.perf_counter() - started, **{f'{k}_seconds': round(v, 1) for k, v in timings.items()},
                               peak_rss_mib=round(peak_mib())))
    basic.write_json(output_dir / 'manifest.json', report)
    return report


def scan_candidates(version_dir):
    version_dir = Path(version_dir)
    m = json.loads((version_dir / 'manifest.json').read_text())
    keep = pl.scan_parquet(str(version_dir / 'decisions' / '*.parquet')).filter(pl.col('decision') == 'candidate')
    base = json.loads((version_dir / 'run-spec.json').read_text())['base_run']
    return pl.scan_parquet(str(Path(base) / 'candidates' / '*.parquet')).join(keep.select('rowid'), on='rowid', how='semi')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=Path('.cache/reddit-cleaning/denoise-x10-2026-09-25'))
    parser.add_argument('--frozen-dir', type=Path, default=Path('.cache/reddit-cleaning/denoise-x20-r6-2026-09-25'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-denoise-aggressive.json'))
    args = parser.parse_args()
    report = run(args.base_dir, args.frozen_dir, args.output_dir, json.loads(args.policy.read_text()))
    print(json.dumps({k: v for k, v in report.items() if k != 'outputs'}, indent=2))


if __name__ == '__main__':
    main()
