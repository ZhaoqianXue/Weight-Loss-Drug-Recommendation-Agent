"""Retained-corpus QA features for an accepted reddit_denoise run (read-only input).

Computes per-record structural features for every x=10 candidate with Polars
(one pass over the candidate/record shards, never the 33 GB source), plus a
fastText language guess for the 10-19-word band the English-scope rule skips.
Writes private feature shards under .cache and aggregate JSON/CSV to results/.
"""
import argparse
import json
from pathlib import Path
import time

import polars as pl

URL = r'(?:https?://|www\.)[^\s<>]+'
RUN = Path('.cache/reddit-cleaning/denoise-x10-2026-09-25')
OUT = Path('.cache/reddit-cleaning/denoise-x10-qa-2026-09-25')
RUNS = r'(?i)(?:' + '|'.join(f'{c}{{8,}}' for c in 'abcdefghijklmnopqrstuvwxyz') + r'|[!?.*=~_\-]{12,})'
BOT_PROBE = (r"(?i)\b(?:i am a bot|i'm a bot|beep boop|automatically|automod(?:erator)?|moderators?|mod team|"
             r"has been removed|was removed|rule ?#?\d|this sub(?:reddit)?\b|megathread|faq|flair|pinned)\b")


def features(frame, lexical_hint=True):
    text = pl.col('text')
    tokens = text.str.to_lowercase().str.replace_all(URL, ' ').str.extract_all(r"[\p{L}\p{N}]+(?:['’.][\p{L}\p{N}]+)*")
    masked = (text.str.to_lowercase().str.replace_all(URL, ' ').str.replace_all(r'(?:^|\s)/?[ur]/[\w-]+', ' ')
              .str.replace_all(r'\d+(?:[.,]\d+)*', '0').str.replace_all(r"[^\p{L}\p{N}']+", ' ')
              .str.replace_all(r'\s+', ' ').str.strip_chars())
    return frame.with_columns(
        chars=text.str.len_chars(),
        url_chars=text.str.extract_all(URL).list.eval(pl.element().str.len_chars()).list.sum().fill_null(0),
        urls=text.str.count_matches(URL),
        letters=text.str.count_matches(r'\p{L}'),
        latin=text.str.count_matches(r'\p{Latin}'),
        visible=text.str.count_matches(r'\S'),
        digits=text.str.count_matches(r'\p{N}'),
        tokens=tokens.list.len(),
        unique_tokens=tokens.list.unique().list.len(),
        char_run=text.str.contains(RUNS),
        bot_probe=text.str.contains(BOT_PROBE),
        title_only=(pl.col('kind') == 'posts') & (text.str.strip_chars() == pl.col('title').fill_null('').str.strip_chars()),
        quote_chars=pl.col('quoted_text').str.len_chars().fill_null(0),
        no_parent=(pl.col('kind') == 'comments') & pl.col('context_parent').is_null(),
        masked_key=masked.hash(seed=1), masked_key2=masked.hash(seed=2), masked_words=masked.str.count_matches(r'\S+'),
    ).drop('text', 'quoted_text', 'title')


def language(frame, model_path):
    """fastText whole-text guess only for the 10-19 lexical-word band."""
    import fasttext
    model = fasttext.load_model(str(model_path))
    band = frame.filter(pl.col('lexical_words') < 20)
    langs, scores = [], []
    for text in band['text']:
        labels, probs = model.predict(' '.join(text.split()), k=1)
        langs.append(labels[0].removeprefix('__label__'))
        scores.append(float(probs[0]))
    return band.select('rowid').with_columns(lang=pl.Series(langs, dtype=pl.String), lang_score=pl.Series(scores, dtype=pl.Float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=RUN)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--model', type=Path, default=Path('.cache/models/lid.176.bin'))
    args = ap.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['status'] != 'full_denoise_complete' or manifest['min_lexical_words'] != 10:
        raise ValueError('Expected the accepted x=10 run')
    target = args.output / 'features'
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for shard in sorted((args.run / 'candidates').glob('part-*.parquet')):
        out = target / shard.name
        if out.exists():
            continue
        cand = pl.read_parquet(shard)
        keys = pl.read_parquet(args.run / 'records' / shard.name, columns=['rowid', 'skeleton'])
        lang = language(cand, args.model)
        feats = features(cand).join(keys, on='rowid', how='left').join(lang, on='rowid', how='left')
        tmp = out.with_suffix('.tmp')
        feats.write_parquet(tmp, compression='zstd')
        tmp.replace(out)
    print(json.dumps(dict(seconds=time.perf_counter() - started, shards=len(list(target.glob('part-*.parquet'))))))


if __name__ == '__main__':
    main()
