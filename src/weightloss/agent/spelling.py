"""Uncoded mentions whose wording misspells a MedDRA PT name ("nausa", "diahrrea", "consitpation", "insulin resistanc").
The coder left them without a term, but they name the PT with a typing error, so the agent's counts treat them as that PT:
query_graph counts them with the PT (as a target member and in PT rankings) and the statement view of run_python carries
the PT on them. The FAERS comparison keeps coded PTs only on both sides (RQ2 rules), so compare_faers and the
comparable_statement view leave them out.

The rule is deliberately narrow, so that a real word is never read as a misspelling: token by token the wording equals
the PT name, with at least one token a typing error of the name's token. A typing error has the same first letter and at
least 5 letters, is at most one edit away (two when either word has 8 or more letters; an edit inserts, deletes or substitutes a
letter or swaps two adjacent ones), sounds alike (same Soundex code) and is not the word with an ending (plural, -y, -ed,
-ing). British PT spellings are compared in their American form (Diarrhoea/diarrhea). A wording that fits more than one
PT name, or that the coder gave a PT elsewhere ("floating"), is left uncoded."""
import collections
import functools
import re
import sqlite3

from .vocabulary import soundex

TOKEN = re.compile(r'[a-z]+')
ENDINGS = frozenset('s es y ed d ing ly al ic ous ness'.split())


def _american(token):
    return token.replace('rrhoea', 'rrhea').replace('oe', 'e').replace('ae', 'e')


def tokens(text):
    return tuple(_american(t) for t in TOKEN.findall(str(text or '').lower().replace('’', "'")))


def _distance(a, b, cap):
    """Edit distance with adjacent swaps (optimal string alignment); any value above cap is returned as cap + 1."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    before, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if before is not None and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], before[j - 2] + 1)
        before, prev = prev, cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def misspelt(word, name_word):
    """word is a typing error for name_word (see the module docstring)."""
    if word == name_word or word[:1] != name_word[:1] or min(len(word), len(name_word)) < 5:
        return False
    for longer, shorter in ((word, name_word), (name_word, word)):
        if longer.startswith(shorter) and longer[len(shorter):] in ENDINGS:
            return False
    cap = 2 if max(len(word), len(name_word)) >= 8 else 1
    return _distance(word, name_word, cap) <= cap and soundex(word) == soundex(name_word)


def variants(uncoded, pts):
    """{wording: pt_code} for the uncoded wordings (lower case, trimmed) that misspell exactly one PT name; pts maps
    pt_code to name."""
    index = collections.defaultdict(list)
    for code, name in pts.items():
        tk = tokens(name)
        if tk:
            index[(len(tk), tk[0][0])].append((code, tk))
    out = {}
    for wording in uncoded:
        tk = tokens(wording)
        if not tk:
            continue
        hits = {code for code, name in index.get((len(tk), tk[0][0]), ())
                if all(a == b or misspelt(a, b) for a, b in zip(tk, name)) and tk != name}
        if len(hits) == 1:
            out[wording] = hits.pop()
    return out


@functools.lru_cache(maxsize=4)
def table(path):
    """{role: {wording: (pt_code, pt)}} for the analysis database at path (read once per process)."""
    con = sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True)
    try:
        out = {}
        for (role,) in con.execute('SELECT DISTINCT role FROM statement'):
            pts = dict(con.execute('SELECT DISTINCT pt_code, pt FROM statement WHERE role = ? AND pt_code IS NOT NULL', (role,)))
            coded = {w for (w,) in con.execute('SELECT DISTINCT lower(trim(wording)) FROM statement WHERE role = ? AND '
                                               'pt_code IS NOT NULL', (role,))}
            uncoded = [w for (w,) in con.execute('SELECT DISTINCT lower(trim(wording)) FROM statement WHERE role = ? AND '
                                                 'pt_code IS NULL AND wording IS NOT NULL', (role,)) if w not in coded]
            out[role] = {w: (c, pts[c]) for w, c in variants(uncoded, pts).items()}
        return out
    finally:
        con.close()


# Python run against an open analysis database `db` (in the agent's sandbox, or by install): a temporary statement view
# that shadows the stored table for unqualified names and gives each misspelled uncoded statement its PT and MedDRA path.
VIEW = """
db.execute("CREATE TEMP TABLE spelling_variant (role TEXT, wording TEXT, pt_code TEXT, PRIMARY KEY (role, wording))")
db.executemany("INSERT INTO spelling_variant VALUES (?, ?, ?)", {rows!r})
_c = [r[1] for r in db.execute("PRAGMA main.table_info(statement)")]
_mapped = {{"pt_code": "m.pt_code", "pt": "m.pt", "hlt_code": "m.hlt_code", "hlt": "m.hlt", "hlgt_code": "m.hlgt_code",
           "hlgt": "m.hlgt", "soc_code": "m.soc_code", "soc": "m.soc", "nonclinical_group": "m.nonclinical_group"}}
db.execute("CREATE TEMP VIEW statement AS SELECT " + ", ".join(
    f"CASE WHEN v.pt_code IS NULL THEN s.{{c}} ELSE {{_mapped[c]}} END AS {{c}}" if c in _mapped else f"s.{{c}}" for c in _c)
    + " FROM main.statement s LEFT JOIN spelling_variant v ON s.pt_code IS NULL AND v.role = s.role "
      "AND v.wording = lower(trim(s.wording)) LEFT JOIN main.meddra m ON m.pt_code = v.pt_code")
del _c, _mapped
"""


def view_code(path):
    rows = sorted((role, w, code) for role, words in table(path).items() for w, (code, _) in words.items())
    return VIEW.format(rows=rows)


def install(db, path):
    """Create the statement view on an open connection to the analysis database at path."""
    exec(view_code(path), {'db': db})
