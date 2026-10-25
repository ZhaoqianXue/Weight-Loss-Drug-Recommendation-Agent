"""Opt-in deterministic localization of model strings (surfaces and evidence quotes) in the input.

Policy ``localize-1.0.0``. An exact match always wins and behaves exactly as without a policy. Only a string that is not
verbatim in the input is matched again after two meaning-preserving normalizations, applied identically to the input and
the string: runs of whitespace become one space, and letters are lower-cased (per character, only where lower-casing
keeps the length). The match is mapped back to original offsets, and the stored string becomes the original substring,
so records stay verbatim and the existing span checks run unchanged. A quote must resolve to exactly one span; a surface
must resolve to exactly one distinct original substring. A normalized match must also sit on token boundaries: it may
not start or end inside a word ("MS" in "symptoms") or inside a number (".5" in "10.5"). Leading/trailing whitespace of
the model string is ignored for the fallback (the stored value is the original substring). Typography, punctuation,
digits, words and word order are never changed and nothing is fuzzy: an unresolved or ambiguous string is left as it
was and stays an anomaly downstream. A case-only repair keeps the model's string in the log; case can carry meaning
(abbreviations), which is why an ambiguous casing is never resolved.

Basis: in the 2026-09-29 post-Jev study, 15 of 36 evidence/surface anomalies differed from the source only by whitespace
(a double space or line break joined into one space) or letter case (a sentence-initial capital); the other 21 were
paraphrases, dropped or added words, an added negation or a hallucinated word, which this policy never matches.

Policy ``localize-1.2.0`` (contract 2.0.0 specs) = localize-1.1.0 with the version-2 number boundaries of
splits_token_v2, applied to exact hits too. localize-1.0.0 and 1.1.0 behave exactly as before.

Policy ``localize-1.1.0`` = localize-1.0.0 unchanged, plus one fail-closed rule for quotes only (2026-09-30, val200
technical repair): a quote that is still not found and contains an explicit internal omission marker ("...", "…",
"[...]") is split at the markers. Every fragment must have at least 3 words and 12 characters and resolve, with the
localize-1.0.0 matching, to exactly one span. The fragments must lie in one field, in order, without overlap, and every
omitted gap must be at most 200 characters with no blank line. The stored quote is then the exact original span from the
first fragment to the last one, so the omitted words are restored, never dropped. Anything else is left for the verbatim
check to reject, e.g. a leading or trailing marker, a short or repeated fragment, an unmarked join or a changed word.

Policy ``localize-1.3.0`` (contract 2.1.0 specs, iteration-20260930-01a0f4c2) = localize-1.2.0 plus two rules:
- **As-written fields** (dose, duration, event_dose, onset, event_duration, subject_label, scope_text): a value that is not a
  bounded exact literal is matched again with the same two normalizations (whitespace runs, letter case) and the
  version-2 boundaries (compound units kept whole for dose fields). Exactly one distinct original substring replaces it;
  anything else is left unchanged for the contract check to reject. A line break the model wrote as a space therefore
  never reaches a model repair.
- **Unmarked joins** (quotes only, after every earlier rule failed): a quote that is two or more sentences, each a
  verbatim passage of the input, joined without the text between them, is split into those passages as separate
  quotes. Every sentence fragment must have at least 3 words and 12 characters and resolve to exactly one span; the
  fragments lie in one field, in order, without overlap, with gaps of at most 200 characters and no blank line, and the
  item keeps at most 8 quotes. Nothing is restored, dropped or reworded: each stored quote is an exact original span. A
  join inside a sentence, a changed word or any failed condition leaves the quote for the verbatim check to reject.
"""
import re
import unicodedata

LOCALIZE_1 = 'localize-1.0.0'
LOCALIZE_1_1 = 'localize-1.1.0'
LOCALIZE_1_2 = 'localize-1.2.0'  # = 1.1.0 with the version-2 number boundaries (splits_token_v2); contract 2.0.0 specs
LOCALIZE_1_3 = 'localize-1.3.0'  # = 1.2.0 + as-written fields + unmarked-join split; contract 2.1.0 specs
POLICIES = (LOCALIZE_1, LOCALIZE_1_1, LOCALIZE_1_2, LOCALIZE_1_3)
GAP_POLICIES = (LOCALIZE_1_1, LOCALIZE_1_2, LOCALIZE_1_3)
V2_POLICIES = (LOCALIZE_1_2, LOCALIZE_1_3)  # version-2 token/number boundaries, exact hits included
JOIN_SENTENCE_END = re.compile(r'[.!?]+["”’\')\]]*(?=\s|$)')  # a sentence ends before whitespace: "0.25", "1.5 mg" stay whole
MAX_QUOTES = 8
FIELD_NAMES = {'purposes': ('dose', 'duration', 'subject_label'),
               'side_effects': ('dose', 'duration', 'event_dose', 'onset', 'event_duration', 'subject_label'),
               'no_side_effects': ('dose', 'duration', 'subject_label', 'scope_text')}  # = contract2.AS_WRITTEN
DOSE_FIELDS = ('dose', 'event_dose')
GAP_MARKER = re.compile(r'\s*(?:\[\s*(?:\.{3,}|…)\s*\]|\.{3,}|…)\s*')
MIN_FRAGMENT_WORDS, MIN_FRAGMENT_CHARS, MAX_GAP_CHARS = 3, 12, 200
BLANK_LINE = re.compile(r'\n[^\S\n]*\n')


def _normalize(text):
    """Normalized text and, for each normalized character, the index of the original character it came from."""
    out, index = [], []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            out.append(' ')
            index.append(i)
            i = j
            continue
        low = c.lower()
        out.append(low if len(low) == 1 else c)
        index.append(i)
        i += 1
    return ''.join(out), index


def _wordish(c):
    return c.isalnum() or unicodedata.category(c).startswith('M')


def _splits_token(hay, start, end):
    """True when hay[start:end] starts or ends inside a word or a number (decimal/thousands separators included)."""
    first, last = hay[start], hay[end - 1]
    before = hay[start - 1] if start > 0 else ''
    after = hay[end] if end < len(hay) else ''
    if before and _wordish(first) and _wordish(before):
        return True
    if after and _wordish(last) and _wordish(after):
        return True
    if before and first in '.,' and end - start > 1 and hay[start + 1].isdigit() and before.isdigit():
        return True
    if after and last in '.,' and end - start > 1 and hay[end - 2].isdigit() and after.isdigit():
        return True
    if before in ('.', ',') and first.isdigit() and start > 1 and hay[start - 2].isdigit():
        return True
    if after in ('.', ',') and last.isdigit() and end + 1 < len(hay) and hay[end + 1].isdigit():
        return True
    return False


def splits_token_v2(hay, start, end, compound_units=False):
    """Version-2 token/number boundaries: the version-1 rules plus
    - a digit right after a decimal mark: "25" inside ".25" (also at the start of the text) or "5" inside "0,5";
    - a digit right after a sign or dash: "5" inside "-5" or "2-5";
    - a leading sign or dash right after a digit: "-5" inside "2-5";
    - a trailing digit right before a dash and a digit: "2" inside "2-5";
    - a leading decimal after a sign: ".5 mg" inside "-.5 mg";
    - with compound_units (dose fields only): either side of a compound unit, "5 mg" inside "5 mg/kg", "kg" inside "mg/kg"."""
    if _splits_token(hay, start, end):
        return True
    first, last = hay[start], hay[end - 1]
    before = hay[start - 1] if start > 0 else ''
    after = hay[end] if end < len(hay) else ''
    after2 = hay[end + 1] if end + 1 < len(hay) else ''
    if before and before in '.,' and first.isdigit():
        return True
    if before and before in '+-−–' and first.isdigit():
        return True
    if before and before in '+-−–' and first in '.,' and end - start > 1 and hay[start + 1].isdigit():  # ".5 mg" inside "-.5 mg"
        return True
    if before and before.isdigit() and first in '+-−–':
        return True
    if after and after in '-−–' and last.isdigit() and after2.isdigit():
        return True
    if compound_units:  # dose fields: "5 mg" is no literal of "5 mg/kg", "5 mg / kg", "5 mg/ kg" or "5 mg /kg"; nor "kg" of them
        j = end
        while j < len(hay) and hay[j] in ' \t':
            j += 1
        if j < len(hay) and hay[j] == '/' and last.isalnum():
            k = j + 1
            while k < len(hay) and hay[k] in ' \t':
                k += 1
            if k < len(hay) and hay[k].isalpha():
                return True
        i = start - 1
        while i >= 0 and hay[i] in ' \t':
            i -= 1
        if i >= 0 and hay[i] == '/' and first.isalpha():
            h = i - 1
            while h >= 0 and hay[h] in ' \t':
                h -= 1
            if h >= 0 and hay[h].isalnum():
                return True
    return False


def normalized_spans(value, fields, splits=None):
    """Every (field, start, end), in original offsets, where the normalized value occurs in a normalized field on token
    boundaries (version 1 by default; `splits` selects another boundary rule)."""
    splits = splits or _splits_token
    needle, _ = _normalize(value.strip())
    if not needle:
        return []
    hits = []
    for name, text in fields.items():
        hay, index = _normalize(text)
        start = hay.find(needle)
        while start != -1:
            end = start + len(needle)
            if not splits(hay, start, end):
                hits.append((name, index[start], index[end - 1] + 1))
            start = hay.find(needle, start + 1)
    return hits


def _splits_for(policy, compound_units=False):
    if policy in V2_POLICIES:
        return (lambda hay, start, end: splits_token_v2(hay, start, end, True)) if compound_units else splits_token_v2
    return _splits_token


def _exact(value, fields, policy, compound_units=False):
    """An exact hit short-circuits localization; under localize-1.2.0+ it must also lie on version-2 boundaries."""
    if policy not in V2_POLICIES:
        return any(value in text for text in fields.values())
    for text in fields.values():
        start = text.find(value)
        while start != -1:
            if not splits_token_v2(text, start, start + len(value), compound_units):
                return True
            start = text.find(value, start + 1)
    return False


def resolve_field(value, fields, name, policy=LOCALIZE_1_3):
    """localize-1.3.0 as-written field: (stored value, repair or None); unresolvable values are returned unchanged."""
    cu = name in DOSE_FIELDS
    if _exact(value, fields, policy, cu):
        return value, None
    hits = normalized_spans(value, fields, _splits_for(policy, cu))
    originals = sorted({fields[f][s:e] for f, s, e in hits})
    if len(originals) != 1:
        return value, None
    return originals[0], {'from': value, 'to': originals[0], 'spans': ['{}:{}-{}'.format(*h) for h in hits[:5]]}


def join_fragments(quote):
    """Sentence fragments of a quote (split after sentence punctuation followed by whitespace, and at line breaks)."""
    out = []
    for line in re.split(r'\n+', quote):
        start = 0
        for m in JOIN_SENTENCE_END.finditer(line):
            out.append(line[start:m.end()].strip())
            start = m.end()
        out.append(line[start:].strip())
    return [f for f in out if f]


def resolve_join_quote(quote, fields, splits=None):
    """localize-1.3.0 unmarked join: ([exact original passages], repair) or (None, None) when any condition fails."""
    if GAP_MARKER.search(quote):
        return None, None
    parts = join_fragments(quote)
    if len(parts) < 2:
        return None, None
    spans = []
    for fragment in parts:
        if len(fragment.split()) < MIN_FRAGMENT_WORDS or len(fragment) < MIN_FRAGMENT_CHARS:
            return None, None
        hits = normalized_spans(fragment, fields, splits)
        if len(hits) != 1:
            return None, None
        spans.append(hits[0])
    name = spans[0][0]
    if any(f != name for f, _, _ in spans):
        return None, None
    text, gaps = fields[name], []
    for (_, _, end), (_, start, _) in zip(spans, spans[1:]):
        if start < end or start - end > MAX_GAP_CHARS or BLANK_LINE.search(text[end:start]):
            return None, None
        gaps.append((end, start))
    if not any(text[a:b].strip() for a, b in gaps):
        return None, None  # nothing was left out between the passages: not a join
    out = [text[s:e] for _, s, e in spans]
    return out, {'kind': 'quote_split', 'from': quote, 'to': out, 'spans': ['{}:{}-{}'.format(*s) for s in spans],
                 'gaps': [f'{name}:{a}-{b}' for a, b in gaps]}


def resolve_surface(value, fields, policy=LOCALIZE_1):
    """(stored value, repair or None). Unresolvable values are returned unchanged for the verbatim check to reject."""
    if _exact(value, fields, policy):
        return value, None
    hits = normalized_spans(value, fields, _splits_for(policy))
    originals = sorted({fields[f][s:e] for f, s, e in hits})
    if len(originals) != 1:
        return value, None
    return originals[0], {'from': value, 'to': originals[0], 'spans': ['{}:{}-{}'.format(*h) for h in hits[:5]]}


def resolve_gap_quote(quote, fields, splits=None):
    """localize-1.1.0 omission rule: (exact original span, repair) or (quote, None) when any condition fails."""
    parts = GAP_MARKER.split(quote.strip())
    if len(parts) < 2 or any(not p.strip() for p in parts):
        return quote, None
    spans = []
    for part in parts:
        fragment = part.strip()
        if len(fragment.split()) < MIN_FRAGMENT_WORDS or len(fragment) < MIN_FRAGMENT_CHARS:
            return quote, None
        hits = normalized_spans(fragment, fields, splits)
        if len(hits) != 1:
            return quote, None
        spans.append(hits[0])
    name = spans[0][0]
    if any(f != name for f, _, _ in spans):
        return quote, None
    text, gaps = fields[name], []
    for (_, _, end), (_, start, _) in zip(spans, spans[1:]):
        if start < end or start - end > MAX_GAP_CHARS or BLANK_LINE.search(text[end:start]):  # order, overlap, gap bound
            return quote, None
        gaps.append(f'{name}:{end}-{start}')
    start, end = spans[0][1], spans[-1][2]
    return text[start:end], {'kind': 'quote_gap', 'from': quote, 'to': text[start:end], 'span': f'{name}:{start}-{end}',
                             'fragments': ['{}:{}-{}'.format(*s) for s in spans], 'gaps': gaps}


def resolve_quote(quote, fields, policy=LOCALIZE_1):
    """(stored quote, repair or None). Exact hits (unique or not) are left to the existing span check."""
    if _exact(quote, fields, policy):
        return quote, None
    hits = normalized_spans(quote, fields, _splits_for(policy))
    if not hits and policy in GAP_POLICIES:
        return resolve_gap_quote(quote, fields, _splits_for(policy))
    if len(hits) != 1:
        return quote, None
    f, s, e = hits[0]
    return fields[f][s:e], {'from': quote, 'to': fields[f][s:e], 'span': f'{f}:{s}-{e}'}


def apply(wire, fields, policy=LOCALIZE_1):
    """Wire answer with resolvable surfaces/quotes replaced by their original substrings, plus the repair log."""
    if policy not in POLICIES:
        raise ValueError(f'unknown localization policy {policy!r}')
    repairs = []

    def fix_surface(obj, path, name):
        value, repair = resolve_surface(getattr(obj, name), fields, policy)
        if repair:
            repairs.append({'path': f'{path}.{name}', 'kind': 'surface', **repair})
        return value

    def fix_quotes(item, path):
        resolved, splits = [], {}
        for q, quote in enumerate(item.quotes):
            value, repair = resolve_quote(quote, fields, policy)
            if repair:
                repairs.append({'path': f'{path}.quotes[{q}]', 'kind': 'quote', **repair})  # a gap repair carries kind quote_gap
            elif policy == LOCALIZE_1_3 and not _exact(quote, fields, policy):
                parts, split = resolve_join_quote(quote, fields, _splits_for(policy))
                if parts is not None:
                    splits[q] = (parts, split)
            resolved.append(value)
        if not splits or len(resolved) - len(splits) + sum(len(parts) for parts, _ in splits.values()) > MAX_QUOTES:
            return resolved  # no join, or the split would overflow the item: the quote stays as written
        out = []
        for q, value in enumerate(resolved):
            if q in splits:
                out.extend(splits[q][0])
                repairs.append({'path': f'{path}.quotes[{q}]', **splits[q][1]})
            else:
                out.append(value)
        return out

    def fix_fields(item, group, path):
        change = {}
        for name in FIELD_NAMES[group]:
            value = getattr(item, name, None)
            if value in (None, ''):
                continue
            new, repair = resolve_field(value, fields, name, policy)
            if repair:
                repairs.append({'path': f'{path}.{name}', 'kind': 'field', **repair})
                change[name] = new
        return change

    update = {'drugs': [d.model_copy(update={'surface': fix_surface(d, f'drugs[{i}]', 'surface')}) for i, d in enumerate(wire.drugs)]}
    for group, name in (('purposes', 'indication'), ('side_effects', 'effect'), ('no_side_effects', None)):
        items = []
        for i, item in enumerate(getattr(wire, group)):
            path = f'{group}[{i}]'
            change = {'quotes': fix_quotes(item, path)}
            if name:
                change[name] = fix_surface(item, path, name)
            if policy == LOCALIZE_1_3:
                change.update(fix_fields(item, group, path))
            items.append(item.model_copy(update=change))
        update[group] = items
    return wire.model_copy(update=update), repairs
