"""Residual review of a concept count: the PTs and uncoded wordings of the denominator that no counted member covers,
and which of them are related in wording to the counted members (the same word, an inflection or a misspelling).

Used by query_graph (not_counted), by residual() in the agent's code sandbox and by the verifier, so the three apply one
rule. The sandbox loads this module by path, so it imports only the standard library and numpy.

Closeness in meaning between PTs comes from the BioLORD-2023 vectors of the MedDRA LLT names that the corpus was coded
with (.cache/meddra-coding-20261015/index; a PT's own LLT has the PT's code and name). It ranks rows for review only:
similar is not the same (Nausea and Vomiting are close in meaning), so a row is enforced through meaning only when it
is nearly synonymous.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[3] / '.cache/meddra-coding-20261015/index'
NEAR_SYNONYM = 0.85   # cosine at which a PT counts as the same as a counted member (Abdominal pain / Abdominal discomfort)
CLOSE = 0.6           # cosine at which a PT is shown as close in meaning
MAX_CLOSE = 8         # rows per platform shown for closeness in meaning alone
_VECTORS = None

WORD = re.compile(r"[a-z][a-z0-9']+")
# Words that relate nothing by themselves: grammar, intensity, time, direction of change, body sites (different
# symptoms share them), drugs.
GENERIC = frozenset("""
a an the of for and or to in on with my me i im i'm is are was were be been being by at as it its from into out up down
this that these those than then so but if when while about after before during because due per via
not no non nothing any all every some more less much many lot lots little bit few most very really quite too also even
just still again only other others another same such like get gets getting got have has had having make makes made
keep kept go going went come came take takes taking took start started starting stop stopped stopping try trying tried
need needed want wanted help helps helped able use used using way thing things stuff something everything
day days daily night nights week weeks weekly month months year years time times hour hours first last new old long
abnormal abnormality decreased decrease decreasing increased increase increasing reduced reduce reduction lower low high
higher disorder disorders syndrome disease diseases condition conditions problem problems issue issues symptom symptoms
side effect effects level levels poor nec test tests type general generalised generalized related unspecified other
feel feels feeling feelings felt bad worse worst severe severely mild mildly moderate extreme extremely constant
constantly occasional occasionally sometimes always never slight slightly major minor terrible horrible awful huge
pain pains painful ache aches aching loss lose loses losing lost gain gains gaining gained change changes changed
control controlled management manage managing treat treated treatment therapy goal goals health healthy better good
blood body skin back person people life kid kids family finally enough around stay well ever else own whole part
place point sure kind sort maybe probably actually basically literally definitely though although yet since until without
within over under through off each both either here there where what which who whose why how myself them they their our
your can could would should will may might must does did doing done say said know think thought see look looking put
give let seem seemed become find found work working purpose purposes reason reasons hard soft dry
upper lower left right site area region lack random peripheral acute chronic local localised localized central primary
secondary
glp glp1 semaglutide tirzepatide liraglutide dulaglutide ozempic wegovy rybelsus mounjaro zepbound victoza saxenda
trulicity sema tirz zep mounj drug drugs med meds medication medications shot shots injection injections dose doses
one two three
""".split())
# Body sites: different symptoms share them (Abdominal pain, Abdominal distension), so a site relates rows only when it is
# all a member's name says (a concept defined by its site, such as Abdominal pain).
SITES = frozenset("""
abdominal abdomen stomach belly tummy gut chest head neck throat mouth tongue eye eyes ear nose face facial leg legs arm
arms hand hands foot feet toe finger joint joints muscle muscular bone heart liver kidney bladder bowel intestinal
gastrointestinal gastric
""".split())


TYPE_N = re.compile(r'\btype\s*([12])\b')


def words(text, sites=False):
    """Content words of a PT name or a wording, lower case, a final plural s dropped ("type 1" -> "type1"); body sites
    only with sites=True."""
    out = set()
    for w in WORD.findall(TYPE_N.sub(r'type\1', str(text or '').lower().replace('’', "'"))):
        w = w.strip("'")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) > 4 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        if len(w) >= 3 and w not in GENERIC and (sites or w not in SITES):
            out.add(w)
    return out


def name_words(name):
    """The words that define a member: its content words, or its body site when it has nothing else."""
    return words(name) or words(name, sites=True)


# Endings that turn one form of a word into another (nausea/nauseous, flatulence/flatulent, diabetes/diabetic; words
# arrive with a final s dropped, so -ous appears as -ou). A shared beginning followed by other letters names different
# things (haemorrhoid/haemorrhage).
SUFFIXES = frozenset('a e i o u y s es ed ing ou ous iou al ial ic ical ia ion ions ness ity ent ence ency ant ance er or ly ive '
                     'ize ise ised ized ism ist ment tic um'.split() + [''])


def _close(a, b, misspelling=True):
    """Two content words name the same thing: equal, one stem with endings from SUFFIXES (nausea/nauseous,
    flatulence/flatulent) or, with misspelling, a one-letter misspelling of a longer word (lipedema/lipidema,
    diarrhea/diarrhoea; from 5 letters a dropped or added letter, nausa/nausea, or two swapped letters, nuasea/nausea).
    Misspellings are looked for only in uncoded wordings: coded rows are spelled correctly, and two real words can differ
    by one letter (farting/fasting)."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5:
        return False
    for k in range(5, min(len(a), len(b)) + 1):
        if a[:k] != b[:k]:
            break
        if a[k:] in SUFFIXES and b[k:] in SUFFIXES:
            return True
    if not misspelling:
        return False
    if min(len(a), len(b)) >= 6 and _one_edit(a, b) or _sounds_alike(a, b):
        return True
    return a[0] == b[0] and (abs(len(a) - len(b)) == 1 and _one_edit(a, b) or _swapped(a, b))


def _swapped(a, b):
    """a and b differ only by two adjacent letters in swapped order."""
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y] if len(a) == len(b) else []
    return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]


def _one_edit(a, b):
    """a and b differ by one substitution, insertion or deletion."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)):] == b[i + 1:]


def soundex(word):
    """American Soundex code of a word (letter + three digits)."""
    codes = {**dict.fromkeys('bfpv', '1'), **dict.fromkeys('cgjkqsxz', '2'), **dict.fromkeys('dt', '3'), 'l': '4',
             **dict.fromkeys('mn', '5'), 'r': '6'}
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return ''
    out, last = word[0], codes.get(word[0])
    for ch in word[1:]:
        c = codes.get(ch)
        if c and c != last:
            out += c
        if ch not in 'hw':
            last = c
    return (out + '000')[:4]


def _sounds_alike(a, b):
    """A misspelling far from the spelling but close in sound (hemeroid/haemorrhoid): long words, same Soundex code."""
    return min(len(a), len(b)) >= 7 and abs(len(a) - len(b)) <= 4 and soundex(a) == soundex(b)


def closeness(counted_codes, codes):
    """{pt code: highest cosine to any counted PT} from the BioLORD LLT vectors; {} when the index is missing."""
    global _VECTORS
    try:
        import numpy as np
        if _VECTORS is None:
            ids = json.loads((INDEX / 'llt_codes.json').read_text())
            _VECTORS = (np.load(INDEX / 'llt_vectors.npy', mmap_mode='r'), {c: i for i, c in enumerate(ids)})
        vectors, pos = _VECTORS
        have = [pos[c] for c in sorted(set(counted_codes)) if c in pos]
        want = [c for c in sorted(set(codes)) if c in pos]
        if not have or not want:
            return {}
        sims = np.asarray(vectors[sorted(pos[c] for c in want)]) @ np.asarray(vectors[sorted(have)]).T
        order = sorted(want, key=lambda c: pos[c])
        return {c: float(sims[i].max()) for i, c in enumerate(order)}
    except (OSError, ValueError, ImportError):
        return {}


def related_word(candidate, counted_words, misspelling=True):
    """The counted word that a candidate PT name or wording shares (or nearly shares), else None. A body site of the
    candidate matches only a site that defines a counted member (name_words)."""
    for w in sorted(words(candidate, sites=True)):
        for c in sorted(counted_words):
            if _close(w, c, misspelling):
                return c
    return None


def check_threshold(counted_records):
    """A related row adding at least this many records to the concept must be counted or named in the answer."""
    return max(2, math.ceil(0.01 * (counted_records or 0)))


def inside_word_only(pattern, wording):
    """The pattern matches the wording, but only inside a longer word ("gas" in "orgasmic")."""
    try:
        hits = list(re.finditer(pattern, wording or '', re.I))
    except re.error:
        return False
    return bool(hits) and not any(m.start() == 0 or not wording[m.start() - 1].isalnum() for m in hits)


def patterns_in(sql):
    """The REGEXP patterns written in an SQL condition."""
    found = re.findall(r"REGEXP\s*'((?:[^']|'')*)'", sql or '', re.I) + re.findall(r'REGEXP\s*"((?:[^"]|"")*)"', sql or '', re.I)
    return [p.replace("''", "'").replace('""', '"') for p in found]


def counted_words(names, wordings, counted_records):
    """(strong, weak) for the counted concepts. strong: one word set per member name (PT, MedDRA group, wording member or
    pattern alternative), the words that define it (name_words). weak: other content words of the counted wordings that
    occur in at least max(2, 2%) of the counted records (so that an incidental word of one wording, such as "back" in
    "get my weight back down", relates nothing). wordings: (wording, records) pairs."""
    strong = []
    for x in names:
        ws = frozenset(name_words(x))
        if ws and ws not in strong:
            strong.append(ws)
    defined = set().union(*strong) if strong else set()
    weight = {}
    for text, records in wordings:
        for w in words(text):
            weight[w] = weight.get(w, 0) + (records or 0)
    floor = max(2, 0.02 * (counted_records or 0))
    return strong, {w for w, n in weight.items() if n >= floor} - defined


def _covers(text, name, misspelling):
    """The text holds every defining word of a member name (each equal, same stem or, with misspelling, one letter off)."""
    have = words(text, sites=True)
    return all(any(_close(h, w, misspelling) for h in have) for w in name)


def mark(rows, vocab, similarity=None):
    """Add related_to, strong and similarity to each residual row (dicts with platform, kind 'pt' or 'wording', code,
    name, records and, for PTs, common_wordings). Strong: the row's own name holds all the defining words of a member
    name (Weight increased for Weight decreased; not Diabetes mellitus for Type 1 diabetes mellitus), or the PT is nearly
    synonymous with a counted PT (similarity, from closeness, at least NEAR_SYNONYM); related_to then names the member's
    words or "meaning". Weak: only a PT's common wordings hold them, or the row shares one frequent word of the counted
    wordings, or the PT is close in meaning (at least CLOSE). related_to None if unrelated."""
    strong, weak = vocab
    similarity = similarity or {}
    out = []
    for r in rows:
        coded = r['kind'] == 'pt'
        sim = similarity.get(r['code']) if coded else None
        word, is_strong = None, False
        for name in strong:
            if _covers(r['name'], name, not coded):
                word, is_strong = ' '.join(sorted(name)), True
                break
        if word is None and sim is not None and sim >= NEAR_SYNONYM:
            word, is_strong = f'meaning ({sim:.2f})', True
        if word is None:  # weak: a counted member's word in the row's common wordings, or a frequent counted-wording word
            texts = [r['name']] + ([' '.join(r.get('common_wordings') or [])] if coded else [])
            for t in texts[1:]:
                for name in strong:
                    if _covers(t, name, False):
                        word = ' '.join(sorted(name))
                        break
                if word:
                    break
            for t in texts:
                if word:
                    break
                word = related_word(t, weak, misspelling=not coded)
        if word is None and sim is not None and sim >= CLOSE:
            word = f'meaning ({sim:.2f})'
        out.append(dict(r, related_to=word, strong=is_strong, similarity=round(sim, 2) if sim is not None else None))
    return out


MAX_CHECKS = 6  # rows per platform that must be counted or named; more would turn answers into exclusion lists


def select(platforms, rows, counted_records, keep_related=25, keep_other=5):
    """The rows to show per platform: related rows (strong before weak, then those adding the most records), then the
    largest other rows. A related row has new_records (records it would add to the concept); check is true for the
    MAX_CHECKS strong rows adding the most records, when they add at least check_threshold records: each must be counted
    or named in the answer."""
    shown = []
    for p in platforms:
        threshold = check_threshold(counted_records.get(p, 0))
        mine = [r for r in rows if r['platform'] == p]
        close_only = lambda r: not r['strong'] and r['related_to'].startswith('meaning')
        meaning = sorted((r for r in mine if r['related_to'] and close_only(r)), key=lambda r: -r['similarity'])[:MAX_CLOSE]
        related = sorted((dict(r, check=False) for r in mine if r['related_to'] and (not close_only(r) or r in meaning)),
                         key=lambda r: (not r['strong'], -r['new_records'], -r['records'], r['name']))
        for r in [r for r in related if r['strong'] and r['new_records'] >= threshold][:MAX_CHECKS]:
            r['check'] = True
        other = sorted((dict(r, new_records=None, check=False) for r in mine if not r['related_to']),
                       key=lambda r: (-r['records'], r['name']))
        shown += related[:keep_related] + [r for r in related[keep_related:] if r['check']] + other[:keep_other]
    return shown
