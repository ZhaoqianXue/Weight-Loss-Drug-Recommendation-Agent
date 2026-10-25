"""Lookup keys. A key is only used to find candidates; the verbatim surface is never replaced by it."""
from __future__ import annotations

import re
import unicodedata

_PUNCT_MAP = str.maketrans({'‘': "'", '’': "'", 'ʼ': "'", '“': '"', '”': '"',
                            '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '−': '-'})
_SPACE = re.compile(r'\s+')
_EDGE_PUNCT = re.compile(r'^[\s"\'.,;:!?()\[\]{}*_~]+|[\s"\'.,;:!?()\[\]{}*_~]+$')
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def fold_key(text):
    """Case-, width- and whitespace-insensitive key; inner punctuation (/, -, +, &) is kept because it can carry
    meaning (and/or, dose ranges)."""
    value = unicodedata.normalize('NFKC', str(text)).translate(_PUNCT_MAP).casefold()
    return _EDGE_PUNCT.sub('', _SPACE.sub(' ', value)).strip()


def loose_key(text):
    """Recall-only key: letters and digits separated by single spaces (never used to accept a mapping)."""
    return ' '.join(_TOKEN.findall(fold_key(text)))


def tokens(text):
    return _TOKEN.findall(fold_key(text))


def trigrams(text):
    padded = f'  {loose_key(text)} '
    return {padded[i:i + 3] for i in range(len(padded) - 2)}
