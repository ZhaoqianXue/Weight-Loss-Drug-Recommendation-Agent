"""Corpus-wide yield scan for aggressive structural rules on top of the frozen r6 retained set.

features:  per base shard, numeric structure features and normalized-sentence hashes for r6
           retained records only (no text is copied; resumable per shard).
sentences: corpus-wide sentence statistics: number of records containing each sentence and
           its earliest owner (created_utc, rowid).
docstats:  per record, words in sentences also present in an earlier record (copied), and in
           high-frequency sentences (boilerplate) at several document-frequency cut-offs.
yield:     threshold curves: hits among r6 retained records, characters removed, overlaps and the
           union, written to results/reddit-denoise-iterations-2026-09-25/aggressive-rule-yield.*
Sentence equality is lowercase + whitespace + trimmed outer punctuation only; digits, decimal
points and negations stay, so "1.5 mg" and "2.5 mg" sentences never match.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from weightloss.cleaning import reddit_denoise_refine as refine  # noqa: E402

BASE = Path('.cache/reddit-cleaning/denoise-x10-2026-09-25')
R6 = Path('.cache/reddit-cleaning/denoise-x20-r6-2026-09-25')
WORK = Path('.cache/reddit-cleaning/aggressive-yield-2026-09-26')
RESULTS = Path('results/reddit-denoise-iterations-2026-09-25')
URL, LEX = refine.URL, refine.LEXICAL
SENTENCE = r'(?:[^.!?\n]|\.\d)+[.!?]*'  # a decimal point does not end a sentence
FUNCTION = ('the|and|to|of|a|i|is|it|you|that|in|my|for|on|this|was|with|but|have|be|not|are|so|just|me|at|do|if|or|what|'
            'can|about|all|get|like|had|they|we|one|as|when|from|your|an|he|she|will|would|there|been|has|out|up|no|how|'
            'more|now|am|did|because|then|still|than|them|some|any|also|much|only|really|very|im|its|dont')
RUNS = '|'.join([r'!{4,}', r'\?{4,}', r'\.{5,}', r'\*{4,}', '~{3,}', '={4,}', '_{4,}', '-{5,}', r'\p{Extended_Pictographic}{4,}']
                + [f'{c}{{5,}}' for c in 'abcdefghijklmnopqrstuvwxyz'])
FREQ = (2, 3, 5, 10, 50, 200)


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def retained(shard):
    keep = pl.read_parquet(R6 / 'decisions' / shard.name, columns=['rowid', 'decision']).filter(pl.col('decision') == 'candidate')
    cols = ['rowid', 'kind', 'created_utc', 'lexical_words', 'text', 'quoted_text', 'title']
    return pl.read_parquet(shard, columns=cols).join(keep.select('rowid'), on='rowid', how='semi')


def features(frame):
    text = pl.col('text')
    low = text.str.to_lowercase()
    title = pl.col('title').fill_null('')
    body = (pl.when((pl.col('kind') == 'posts') & text.str.starts_with(title)).then(text.str.slice(title.str.len_chars()))
            .otherwise(text))
    tok = low.str.replace_all(URL, ' ').str.extract_all(LEX)
    tri = tok.list.eval(pl.concat_str([pl.element(), pl.element().shift(-1), pl.element().shift(-2)], separator=' ').drop_nulls())
    return frame.select(
        pl.col('rowid').cast(pl.UInt32), 'kind', 'lexical_words',
        chars=text.str.len_chars(),
        body_words=pl.when(pl.col('kind') == 'posts').then(body.str.replace_all(URL, ' ').str.count_matches(LEX))
        .otherwise(pl.col('lexical_words')).cast(pl.Int32),
        url_n=text.str.count_matches(URL).cast(pl.Int32),
        url_chars=text.str.extract_all(URL).list.eval(pl.element().str.len_chars()).list.sum().fill_null(0).cast(pl.Int32),
        quote_chars=pl.col('quoted_text').fill_null('').str.len_chars().cast(pl.Int32),
        n_tok=tok.list.len().cast(pl.Int32), n_uniq=tok.list.n_unique().cast(pl.Int32),
        top1=tok.list.eval(pl.element().unique_counts().max()).list.first().fill_null(0).cast(pl.Int32),
        tri_n=tri.list.len().cast(pl.Int32), tri_uniq=tri.list.n_unique().cast(pl.Int32),
        fn_n=low.str.count_matches(rf'\b(?:{FUNCTION})\b').cast(pl.Int32),
        sym_n=text.str.count_matches(r'[\p{P}\p{S}]').cast(pl.Int32),
        emoji_n=text.str.count_matches(r'\p{Extended_Pictographic}').cast(pl.Int32),
        letters=text.str.count_matches(r'\p{L}').cast(pl.Int32),
        space_n=text.str.count_matches(r'\s').cast(pl.Int32),
        nonascii=text.str.count_matches(r'[^\x00-\x7F]').cast(pl.Int32),
        moji=text.str.count_matches('Ã[\u0080-¿]|â€|ï¿½|�').cast(pl.Int32),
        run_n=low.str.count_matches(RUNS).cast(pl.Int32),
        lines=(text.str.count_matches('\n') + 1).cast(pl.Int32),
        list_lines=text.str.count_matches(r'(?m)^[ \t]*(?:[-*•+]|\d+[.)])[ \t]').cast(pl.Int32),
        table_lines=text.str.count_matches(r'(?m)^[ \t]*\|.*\|[ \t]*$').cast(pl.Int32))


def sentences(frame):
    key = (pl.col('created_utc').cast(pl.UInt64) * (1 << 24) + pl.col('rowid').cast(pl.UInt64)).alias('key')
    s = (frame.select(pl.col('rowid').cast(pl.UInt32), key,
                      piece=pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(SENTENCE))
         .explode('piece')
         .with_columns(norm=pl.col('piece').str.replace_all(r'^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$', '').str.replace_all(r'\s+', ' '))
         .with_columns(sw=pl.col('norm').str.count_matches(LEX).cast(pl.UInt16))
         .filter(pl.col('sw') > 0))
    return (s.select('rowid', 'key', shash=pl.col('norm').hash(41, 43, 47, 53), sw='sw')
            .unique(subset=['rowid', 'shash'], keep='first', maintain_order=True))


def run_features():
    (WORK / 'features').mkdir(parents=True, exist_ok=True)
    (WORK / 'sentences').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for shard in sorted((BASE / 'candidates').glob('part-*.parquet')):
        f_out, s_out = WORK / 'features' / shard.name, WORK / 'sentences' / shard.name
        if f_out.exists() and s_out.exists():
            continue
        frame = retained(shard)
        features(frame).write_parquet(f_out.with_suffix('.tmp'), compression='zstd')
        sentences(frame).write_parquet(s_out.with_suffix('.tmp'), compression='zstd')
        f_out.with_suffix('.tmp').replace(f_out)
        s_out.with_suffix('.tmp').replace(s_out)
        print(shard.name, frame.height, f'{time.perf_counter() - started:.0f}s', f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def run_sentences():
    started = time.perf_counter()
    stats = (pl.scan_parquet(str(WORK / 'sentences' / '*.parquet')).group_by('shash')
             .agg(docs=pl.len().cast(pl.UInt32), owner=pl.col('key').min()).filter(pl.col('docs') > 1)
             .collect(engine='streaming'))
    stats.write_parquet(WORK / 'sentence_stats.parquet', compression='zstd')
    print('shared sentences', stats.height, f'{time.perf_counter() - started:.0f}s', f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def run_docstats():
    (WORK / 'docstats').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stats = pl.read_parquet(WORK / 'sentence_stats.parquet')
    for shard in sorted((WORK / 'sentences').glob('part-*.parquet')):
        out = WORK / 'docstats' / shard.name
        if out.exists():
            continue
        s = pl.read_parquet(shard).join(stats, on='shash', how='left').with_columns(
            docs=pl.col('docs').fill_null(1), copied=pl.col('owner').is_not_null() & (pl.col('owner') != pl.col('key')))
        w = pl.col('sw').cast(pl.Int32)
        agg = s.group_by('rowid').agg(
            total=w.sum(), copied_all=w.filter(pl.col('copied')).sum(),
            copied_5=w.filter(pl.col('copied') & (pl.col('sw') >= 5)).sum(),
            **{f'freq{k}': w.filter(pl.col('docs') >= k).sum() for k in FREQ})
        agg.write_parquet(out.with_suffix('.tmp'), compression='zstd')
        out.with_suffix('.tmp').replace(out)
    print('docstats', f'{time.perf_counter() - started:.0f}s', f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def rules():
    """(family, rule, threshold label, boolean expression) over features joined with docstats."""
    c = pl.col
    share = lambda a, b: c(a).cast(pl.Float64) / pl.max_horizontal(c(b), pl.lit(1)).cast(pl.Float64)  # noqa: E731
    out = []
    for t in (0.6, 0.75, 0.9):
        out.append(('F1 lossy copy compression', 'copied_share_all', f'>={t}', share('copied_all', 'total') >= t))
        out.append(('F1 lossy copy compression', 'copied_share_5w', f'>={t}', share('copied_5', 'total') >= t))
    for r in (10, 20):
        for t in (0.4, 0.5):
            out.append(('F1 lossy copy compression', 'copied_share_all_small_residual', f'>={t} & residual<={r}',
                        (share('copied_all', 'total') >= t) & ((c('total') - c('copied_all')) <= r)))
    for k in FREQ:
        for t in (0.5, 0.6, 0.75, 0.9):
            for r in (5, 10, 20):
                out.append(('F3 high-frequency sentence templates', f'freq{k}_cover', f'>={t} & residual<={r}',
                            (share(f'freq{k}', 'total') >= t) & ((c('total') - c(f'freq{k}')) <= r)))
    for t in (5, 10, 20):
        out.append(('F2 low information', 'post_body_words', f'<{t}', (c('kind') == 'posts') & (c('body_words') < t)))
    for t in (0.5, 0.7, 0.9):
        out.append(('F2 low information', 'quote_share', f'>={t}',
                    c('quote_chars').cast(pl.Float64) / (c('quote_chars') + c('chars')).cast(pl.Float64) >= t))
    for t in (0.3, 0.5, 0.7):
        out.append(('F2 low information', 'repeated_trigram_share', f'>={t}',
                    (c('n_tok') >= 20) & (1 - share('tri_uniq', 'tri_n') >= t)))
    for t in (0.15, 0.25, 0.35):
        out.append(('F2 low information', 'top_word_share', f'>={t}', (c('n_tok') >= 20) & (share('top1', 'n_tok') >= t)))
    for t in (0.35, 0.45, 0.55):
        out.append(('F2 low information', 'type_token_ratio_20_400w', f'<={t}',
                    c('n_tok').is_between(20, 400) & (share('n_uniq', 'n_tok') <= t)))
    for t in (0.15, 0.25, 0.4):
        out.append(('F2 low information', 'symbol_share', f'>={t}',
                    c('sym_n').cast(pl.Float64) / pl.max_horizontal(c('chars') - c('space_n'), pl.lit(1)) >= t))
    for t in (0.2, 0.35, 0.5):
        out.append(('F2 low information', 'emoji_per_word', f'>={t}', (c('emoji_n') >= 3) & (share('emoji_n', 'n_tok') >= t)))
    for t in (0.5, 0.75):
        out.append(('F2 low information', 'list_table_line_share', f'>={t}',
                    ((c('list_lines') + c('table_lines')) >= 3) & (share('list_lines', 'lines') + share('table_lines', 'lines') >= t)))
    for t in (0.3, 0.5, 0.7):
        out.append(('F2 low information', 'url_char_share', f'>={t}', share('url_chars', 'chars') >= t))
    for t in (1, 2, 3):
        out.append(('F2 low information', 'char_runs', f'>={t}', c('run_n') >= t))
    for t in (0.05, 0.08, 0.12):
        out.append(('F4 language/garble', 'english_function_word_share', f'<{t}', (c('n_tok') >= 20) & (share('fn_n', 'n_tok') < t)))
    for t in (0.5, 0.6, 0.7):
        out.append(('F4 language/garble', 'letter_share', f'<{t}',
                    c('letters').cast(pl.Float64) / pl.max_horizontal(c('chars') - c('space_n'), pl.lit(1)) < t))
    for t in (1, 3):
        out.append(('F4 language/garble', 'mojibake_marks', f'>={t}', c('moji') >= t))
    for t in (0.1, 0.2, 0.3):
        out.append(('F4 language/garble', 'non_ascii_share', f'>={t}', share('nonascii', 'chars') >= t))
    return out


def run_yield(timings):
    started = time.perf_counter()
    frame = (pl.scan_parquet(str(WORK / 'features' / '*.parquet'))
             .join(pl.scan_parquet(str(WORK / 'docstats' / '*.parquet')), on='rowid', how='left')
             .with_columns(pl.col(['total', 'copied_all', 'copied_5', *[f'freq{k}' for k in FREQ]]).fill_null(0)))
    rs = rules()
    base_n, base_chars = frame.select(pl.len(), pl.col('chars').cast(pl.Int64).sum()).collect(engine='streaming').row(0)
    aggs = []
    for i, (_, _, _, e) in enumerate(rs):
        aggs += [e.sum().alias(f'n{i}'), pl.col('chars').cast(pl.Int64).filter(e).sum().alias(f'c{i}')]
    got = frame.select(aggs).collect(engine='streaming').row(0, named=True)
    rows = [dict(family=f, rule=r, threshold=t, hits=int(got[f'n{i}']), chars_removed=int(got[f'c{i}'] or 0),
                 retained_after=base_n - int(got[f'n{i}']), chars_after=base_chars - int(got[f'c{i}'] or 0),
                 hit_pct=round(100 * got[f'n{i}'] / base_n, 3), char_pct=round(100 * (got[f'c{i}'] or 0) / base_chars, 3))
            for i, (f, r, t, _) in enumerate(rs)]
    out = dict(generated_at=datetime.now(timezone.utc).isoformat(), status='first yield table (single rules)',
               baseline=dict(version=str(R6), retained=base_n, characters=base_chars, min_lexical_words=20),
               unit='hits = r6-retained records a rule would additionally remove; chars_removed = their characters',
               rows=rows, timings_seconds={**timings, 'yield': round(time.perf_counter() - started, 1)},
               peak_rss_mib=round(peak_mib()))
    return out


MINHASH, BANDS = 20, 5  # 5 bands x 4 rows: P(candidate) ~0.93 at Jaccard 0.8, ~0.995 at 0.9
A1 = Path('.cache/reddit-cleaning/denoise-x20-a1-2026-09-26')
# Generalized social expressions (congratulation, praise, thanks, wishes, encouragement, sympathy, greeting); no brand,
# identity or topic words. Phrases are removed first so their words are counted once.
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


def features2(frame):
    """Social-expression coverage, residual content words and punctuation density per record."""
    low = pl.col('text').str.to_lowercase().str.replace_all(URL, ' ')
    rest = low.str.replace_all(rf'\b(?:{SOCIAL_PHRASES})\b', ' ')
    n_all, n_rest = low.str.count_matches(LEX), rest.str.count_matches(LEX)
    fn_rest = rest.str.count_matches(rf'\b(?:{FUNCTION})\b')
    soc_rest = rest.str.count_matches(rf'\b(?:{SOCIAL_WORDS})\b')
    return frame.select(pl.col('rowid').cast(pl.UInt32),
                        social_n=((n_all - n_rest) + soc_rest).cast(pl.Int32),
                        content_n=(n_rest - fn_rest - soc_rest).cast(pl.Int32),
                        excl_n=pl.col('text').str.count_matches('!').cast(pl.Int32),
                        sent_n=pl.col('text').str.count_matches(r'[.!?]+(?:\s|$)|\n').cast(pl.Int32))


def run_features2():
    (WORK / 'features2').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for shard in sorted((BASE / 'candidates').glob('part-*.parquet')):
        out = WORK / 'features2' / shard.name
        if not out.exists():
            features2(retained(shard)).write_parquet(out.with_suffix('.tmp'), compression='zstd')
            out.with_suffix('.tmp').replace(out)
    print('features2', f'{time.perf_counter() - started:.0f}s', f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def run_minhash():
    """Word-trigram MinHash signatures (20 x u32) and LSH band keys per retained record."""
    (WORK / 'minhash').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = MINHASH // BANDS
    for shard in sorted((BASE / 'candidates').glob('part-*.parquet')):
        out = WORK / 'minhash' / shard.name
        if out.exists():
            continue
        frame = retained(shard)
        tok = pl.col('text').str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(LEX)
        sh = (frame.select(pl.col('rowid').cast(pl.UInt32), 'created_utc', 'kind',
                           tri=tok.list.eval(pl.concat_str([pl.element(), pl.element().shift(-1), pl.element().shift(-2)],
                                                           separator=' ').drop_nulls()))
              .explode('tri').drop_nulls('tri').with_columns(h=pl.col('tri').hash(7, 11, 13, 17)).drop('tri'))
        sig = sh.group_by('rowid').agg(pl.col('created_utc').first(), pl.col('kind').first(), n_sh=pl.len().cast(pl.UInt32),
                                       **{f'm{i}': (pl.col('h').hash(1000 + i) % 4294967296).min().cast(pl.UInt32) for i in range(MINHASH)})
        sig = sig.with_columns(**{f'b{b}': pl.struct([f'm{b * rows + j}' for j in range(rows)]).hash(99) for b in range(BANDS)})
        sig.write_parquet(out.with_suffix('.tmp'), compression='zstd')
        out.with_suffix('.tmp').replace(out)
    print('minhash', f'{time.perf_counter() - started:.0f}s', f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def run_neardup():
    """Candidate groups per band; each member is compared with its group's earliest record (posts first on ties)."""
    started = time.perf_counter()
    scan = pl.scan_parquet(str(WORK / 'minhash' / '*.parquet'))
    order = (pl.col('created_utc').cast(pl.UInt64) * (1 << 24) + pl.col('rowid').cast(pl.UInt64))
    pairs = []
    for b in range(BANDS):
        g = (scan.select('rowid', key=pl.col(f'b{b}'), order=order).group_by('key')
             .agg(n=pl.len(), rep=pl.col('rowid').sort_by('order').first(), members=pl.col('rowid'))
             .filter(pl.col('n') > 1).select('rep', 'members').explode('members')
             .filter(pl.col('members') != pl.col('rep')).rename({'members': 'rowid'}).collect(engine='streaming'))
        pairs.append(g)
    pairs = pl.concat(pairs).unique()
    ids = pl.concat([pairs['rowid'], pairs['rep']]).unique().to_frame('rowid')
    sig = scan.join(ids.lazy(), on='rowid', how='semi').select('rowid', *[f'm{i}' for i in range(MINHASH)]).collect()
    both = pairs.join(sig, on='rowid').join(sig, left_on='rep', right_on='rowid', suffix='_rep')
    both = both.select('rowid', 'rep', sim=pl.sum_horizontal([(pl.col(f'm{i}') == pl.col(f'm{i}_rep')).cast(pl.UInt8)
                                                               for i in range(MINHASH)]) / MINHASH)
    best = both.group_by('rowid').agg(pl.col('sim').max(), rep=pl.col('rep').sort_by('sim', descending=True).first())
    best.write_parquet(WORK / 'neardup_best.parquet', compression='zstd')
    print('near-dup pairs', pairs.height, 'records with a candidate', best.height, f'{time.perf_counter() - started:.0f}s',
          f'{peak_mib():.0f}MiB', flush=True)
    return time.perf_counter() - started


def joined():
    frame = (pl.scan_parquet(str(WORK / 'features' / '*.parquet'))
             .join(pl.scan_parquet(str(WORK / 'docstats' / '*.parquet')), on='rowid', how='left')
             .with_columns(pl.col(['total', 'copied_all', 'copied_5', *[f'freq{k}' for k in FREQ]]).fill_null(0)))
    if (WORK / 'neardup_best.parquet').exists():
        frame = frame.join(pl.scan_parquet(WORK / 'neardup_best.parquet').select('rowid', 'sim'), on='rowid', how='left')
    return frame.with_columns(sim=pl.col('sim').fill_null(0.0) if 'sim' in frame.collect_schema() else pl.lit(0.0))


def neardup_rows(base_n, base_chars):
    frame = joined()
    rows = []
    for t in (0.8, 0.9, 0.95):
        n, ch = frame.select((pl.col('sim') >= t).sum(), pl.col('chars').cast(pl.Int64).filter(pl.col('sim') >= t).sum()
                             ).collect(engine='streaming').row(0)
        rows.append(dict(family='F1 near-duplicate (MinHash word 3-gram)', rule='similarity_to_group_earliest', threshold=f'>={t}',
                         hits=int(n), chars_removed=int(ch or 0), retained_after=base_n - int(n), chars_after=base_chars - int(ch or 0),
                         hit_pct=round(100 * n / base_n, 3), char_pct=round(100 * (ch or 0) / base_chars, 3)))
    return rows


def combine(selected, sim=None):
    """Union, pairwise overlaps and exclusive contribution for chosen (rule, threshold) pairs (+ near-dup similarity)."""
    pool = {(r, t): e for _, r, t, e in rules()}
    named = [(f'{r} {t}', pool[(r, t)]) for r, t in selected]
    if sim is not None:
        named.append((f'near-dup sim>={sim}', pl.col('sim') >= sim))
    frame = joined().with_columns([e.alias(f'f{i}') for i, (_, e) in enumerate(named)])
    flags = [pl.col(f'f{i}') for i in range(len(named))]
    union = pl.any_horizontal(flags)
    count = pl.sum_horizontal([f.cast(pl.UInt8) for f in flags])
    aggs = [union.sum().alias('union'), pl.col('chars').cast(pl.Int64).filter(union).sum().alias('union_chars'),
            pl.len().alias('base'), pl.col('chars').cast(pl.Int64).sum().alias('base_chars')]
    for i in range(len(named)):
        aggs.append((flags[i] & (count == 1)).sum().alias(f'only{i}'))
        aggs.append(pl.col('chars').cast(pl.Int64).filter(flags[i] & (count == 1)).sum().alias(f'onlyc{i}'))
        for j in range(i + 1, len(named)):
            aggs.append((flags[i] & flags[j]).sum().alias(f'x{i}_{j}'))
    got = frame.select(aggs).collect(engine='streaming').row(0, named=True)
    return dict(rules=[n for n, _ in named], union_hits=got['union'], union_chars=got['union_chars'],
                retained_after=got['base'] - got['union'], chars_after=got['base_chars'] - got['union_chars'],
                union_pct=round(100 * got['union'] / got['base'], 2), char_pct=round(100 * got['union_chars'] / got['base_chars'], 2),
                exclusive={n: dict(hits=got[f'only{i}'], chars=got[f'onlyc{i}']) for i, (n, _) in enumerate(named)},
                pairwise_overlap={f'{named[i][0]} & {named[j][0]}': got[f'x{i}_{j}']
                                  for i in range(len(named)) for j in range(i + 1, len(named))})


def hit_examples(rule, threshold, n=12, seed=7, extra=None):
    """Random hits of one rule with text (printed locally for loss assessment, never written to results)."""
    pool = {(r, t): e for _, r, t, e in rules()}
    e = pool[(rule, threshold)] if rule != 'near-dup' else pl.col('sim') >= float(threshold)
    if extra is not None:
        e = e & extra
    hits = joined().filter(e).select('rowid').collect(engine='streaming')
    pick = hits.sample(min(n, hits.height), seed=seed)['rowid'].cast(pl.Int64).to_list()
    text = (pl.scan_parquet(str(BASE / 'candidates' / '*.parquet')).select('rowid', 'kind', 'subreddit', 'text', 'quoted_text')
            .filter(pl.col('rowid').is_in(pick)).collect())
    feats = joined().filter(pl.col('rowid').is_in(pick)).collect()
    return hits.height, text.join(feats.with_columns(pl.col('rowid').cast(pl.Int64)), on='rowid', suffix='_f')


def report(extra=None):
    """Write the acceptance files from the single-rule yield (and optional extra sections)."""
    data = json.loads((WORK / 'single-rule-yield.json').read_text())
    data.update(extra or {})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'aggressive-rule-yield.json').write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    rows = data['rows'] + data.get('neardup_rows', [])
    pl.DataFrame(rows).select('family', 'rule', 'threshold', 'hits', 'hit_pct', 'chars_removed', 'char_pct', 'retained_after',
                              'chars_after').write_csv(RESULTS / 'aggressive-rule-yield.csv')
    b = data['baseline']
    top = sorted(rows, key=lambda r: -r['chars_removed'])
    lines = ['# 激进结构规则收益表（全库实测，基于冻结 r6 保留集）', '',
             f"生成：{data['generated_at']}。基线 r6 保留 {b['retained']:,} 条、正文 {b['characters']:,} 字符，x 仍为 20。",
             '“命中”= r6 保留记录中该规则会额外删除的条数（相对 r6 的净新增）；“字符”= 这些记录的正文字符数。单规则之间有重叠，不能相加；联合结果见下文。', '']
    if data.get('progress'):
        lines += ['## 进度', ''] + [f'- {p}' for p in data['progress']] + ['']
    if data.get('combination'):
        lines += ['## 组合与重叠', ''] + [f'- {p}' for p in data['combination']] + ['']
    for key, heading in (('a2_audit', 'a2 新增删除的逐条审计与实测冗余核对'), ('feasibility_1m', '缩到约 100 万条的可行性（历史评估）'),
                         ('strict_subset', '约 100 万条严格子集（有损选择，单列于 x=20 之后）')):
        if data.get(key):
            lines += [f'## {heading}', ''] + [f'- {p}' for p in data[key]] + ['']
    lines += ['## 按字符减量排序（前 25）', '', '| 族 | 规则 | 阈值 | 命中 | 占保留 % | 字符减量 | 占字符 % |', '|---|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['family']} | {r['rule']} | {r['threshold']} | {r['hits']:,} | {r['hit_pct']} | {r['chars_removed']:,} | {r['char_pct']} |"
              for r in top[:25]]
    lines += ['', f"完整阈值曲线：[aggressive-rule-yield.csv](aggressive-rule-yield.csv)。耗时（秒）：{data['timings_seconds']}；峰值 RSS {data['peak_rss_mib']} MiB。", '',
              '## 说明', '',
              '- F1（句子级复制压缩）与近重复：删除的是内容大部分已出现在更早记录里的副本，保留最早者；属于有损冗余压缩，不是纯噪声，也不是逐字无损。句子只做小写、空白和首尾标点规范化，数字、小数点、否定词保留。',
              '- F3（高频句模板）：按“句子在多少条记录中出现”判定模板句，覆盖率和原创残余词数共同决定；这是基于重复结构的轻度语义判断，不按品牌、身份或“谢谢”开头一刀切。',
              '- F2/F4：结构与字符统计（词汇多样性、重复 n-gram、符号/emoji、列表/URL/引用占比、英文功能词比例、字母比例、乱码标记）；没有药名、症状、第一人称、主题或 subreddit 条件。',
              '- 全局最小词数仍是 20；帖子正文词数（去标题）等额外门槛单列，不改全局门槛。']
    (RESULTS / 'aggressive-rule-yield.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['features', 'sentences', 'docstats', 'yield', 'all', 'minhash', 'neardup', 'report', 'features2'])
    args = parser.parse_args()
    if args.step == 'minhash':
        return print(run_minhash())
    if args.step == 'features2':
        return print(run_features2())
    if args.step == 'neardup':
        return print(run_neardup())
    if args.step == 'report':
        return report()
    timings = {}
    if args.step in ('features', 'all'):
        timings['features'] = round(run_features(), 1)
    if args.step in ('sentences', 'all'):
        timings['sentences'] = round(run_sentences(), 1)
    if args.step in ('docstats', 'all'):
        timings['docstats'] = round(run_docstats(), 1)
    if args.step in ('yield', 'all'):
        out = run_yield(timings)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (WORK / 'single-rule-yield.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
