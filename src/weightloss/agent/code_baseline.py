"""Code-agent baseline: the same model given the same data as files and a Python sandbox, as a chat app with a code
interpreter would be after the knowledge graph is uploaded. Used only to compare against the agent.

The model gets one tool, run_python, that runs code in a sandboxed subprocess (macOS sandbox-exec: no network, writes only
to the session's temporary folder). Earlier successful cells are re-run silently before each new cell, so variables
persist as in a notebook. The data are read-only: the analysis database (weightloss.graph.analysis_db, the same tables
the agent's run_sql sees), the raw kg.sqlite with the source sentences, and the frozen FAERS files. The instructions give
the same data facts, counting rules and medical-advice policy as the agent; there are no graph tools, no citation or quote
verification and no repair round. The final answer is strict JSON (answer, answer_type, limitations).
"""
from __future__ import annotations

import collections
import functools
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from weightloss.extraction.structured import (ExtractionAnomaly, compact_json, cost_usd, final_output_text, mask,
                                              parse_json_object, sha256_text, strict_json_schema, token_split, validate_model)
from weightloss.graph import analysis_db, neo4j_db

from . import agent, vocabulary

MODEL, EFFORT = agent.MODEL, agent.EFFORT
MAX_TOOL_ROUNDS = 12
CELL_SECONDS = 120
OUTPUT_CHARS = 12000
VERSION = 'code-baseline-1.2.1'
FAERS_DIR = neo4j_db.FROZEN

SETUP = f'''
import sqlite3, json, math, re
import pandas as pd
ANALYSIS_DB = {str(analysis_db.PATH)!r}
KG_DB = {str(neo4j_db.KG)!r}
FAERS_DIR = {str(FAERS_DIR)!r}
def _wilson(k, n, sign):
    if not n: return None
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * min(1.0, max(0.0, c + sign * h)), 2)
def connect(path):
    con = sqlite3.connect(f"file:{{path}}?mode=ro&immutable=1", uri=True)
    con.create_function("wilson_low", 2, lambda k, n: _wilson(k, n, -1))
    con.create_function("wilson_high", 2, lambda k, n: _wilson(k, n, 1))
    con.create_function("regexp", 2, lambda p, v: v is not None and re.search(p, v, re.I) is not None)
    return con
db = connect(ANALYSIS_DB)
kg = connect(KG_DB)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 60)
'''

RUNNER = r'''
import ast, contextlib, io, json, sys, traceback
cells = json.load(open(sys.argv[1]))
g = {"__name__": "__main__", "_REPLAY": True}
exec(compile(open(sys.argv[2]).read(), "<setup>", "exec"), g)
for c in cells[:-1]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            exec(compile(c, "<cell>", "exec"), g)
        except Exception:
            pass
code = cells[-1]
g["_REPLAY"] = False
try:
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    if last is not None:
        tree.body = tree.body[:-1]
    exec(compile(tree, "<cell>", "exec"), g)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), g)
        if value is not None:
            print(repr(value) if not hasattr(value, "to_string") else value.to_string(max_rows=60))
except Exception:
    traceback.print_exc(limit=3)
    sys.exit(3)
'''

PROFILE = '''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (subpath "/private{tmp}") (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))
'''

TOOLS = [{'type': 'function', 'name': 'run_python', 'strict': True,
          'description': 'Run Python in a notebook-like sandbox (state persists between calls). Already defined: db (read-only '
                         'SQLite connection to the analysis tables), kg (read-only connection to the raw knowledge graph with '
                         'source sentences), pd (pandas), ANALYSIS_DB, KG_DB, FAERS_DIR. The value of a final expression is '
                         'printed. No network; at most 120 s and about 12,000 characters of output per call.',
          'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['code'],
                         'properties': {'code': {'type': 'string', 'description': 'Python code to run'}}}}]


class BaselineAnswer(BaseModel):
    answer: str = Field(description='Markdown answer to the question')
    answer_type: Literal['answered', 'limited', 'declined'] = Field(
        description='answered: the data answer the question; limited: only part of it can be answered from the data, or the '
                    'data are insufficient; declined: medical advice or another question the data cannot answer')
    limitations: list[str] = Field(description='Short statements of data limits relevant to this answer')


FINAL_SCHEMA = strict_json_schema(BaselineAnswer)

DATA = """Data files (read-only):
1. Analysis database (variable db): the knowledge graph as flat tables.
   - statement: one row per statement a writer made. role 'side_effect' (a symptom the writer reports having and links to the drug) or 'purpose' (the writer's stated reason for using it); edge_id; platform 'reddit' or 'webmd'; record_id and record_key (one Reddit post or comment, or one WebMD review); date (YYYY-MM-DD) and year; drug_id, ingredient (semaglutide, tirzepatide, liraglutide, dulaglutide or unspecified), brand (Ozempic, Wegovy, Rybelsus, Mounjaro, Zepbound, Victoza, Saxenda, Trulicity, or 'unspecified' when no brand is named); ingredient_basis ('name' or 'profile' when named, 'community' when assigned from a single-drug Reddit community); writer_age_group (13-18, 19-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75 or over) and writer_sex ('female', 'male', 'other'), NULL when not stated (Reddit: stated by the writer about themselves; WebMD: reviewer profile); source ('profile' for a WebMD profile condition chosen from the site's list, else 'text'); wording (the writer's words); pt_code and pt (MedDRA 29.1 Preferred Term; NULL = no adequate term), hlt/hlgt/soc codes and names (the PT's MedDRA primary path), nonclinical_group (NULL for clinical events); side effects only: intensity (mild, moderate, severe), impact (daily_life, medical_visit, emergency_visit, hospitalization), action_taken (JSON list) with 0/1 columns stopped, dose_reduced, dose_skipped, escalation_paused, dose_text and dose_mg (parsed number of mg), duration_text and duration_bucket (under 1 day, 1-6 days, 1-4 weeks, over 4 weeks).
   - faers_report(report_id, receivedate, year, sex, age_group, hospitalization 0/1, in_primary_analysis 0/1, has_clinical_pt 0/1), faers_stratum(report_id, level 'brand'|'ingredient'|'unspecified', ingredient, brand, in_primary 0/1, actions, action_known 0/1, withdrawn 0/1, dose_reduced 0/1, window_start), faers_reaction(report_id, pt_code, pt, hlt_code, hlt, hlgt_code, hlgt, soc_code, soc, nonclinical_group, clinical 0/1): validated FAERS spontaneous reports (openFDA release 2026-07-30, primary configuration).
   - meddra(pt_code, pt, hlt_code, hlt, hlgt_code, hlgt, soc_code, soc, nonclinical_group); approval(brand, ingredient, approval_date).
   SQL functions: wilson_low(k, n), wilson_high(k, n) (95% interval in percent) and REGEXP (case-insensitive).
2. Raw knowledge graph (variable kg): tables drug, purpose, side_effect, edge_used_for, edge_side_effect; edge tables carry text (wording) and evidence (JSON list of the cited source sentences, field 'text') and writer_evidence.
3. FAERS_DIR: the frozen FAERS CSV files the faers_* tables come from.

Counting rules: the unit is a record (record_key); a concept counts once per record. A share's denominator is the number of records with at least one statement of the same role under the same filters. Reddit and WebMD are reported separately and never pooled. Brand-level results use named brands only. A condition such as "people who take it for X" means a purpose statement for the same drug in the same record. A FAERS scope is the reports with a faers_stratum row of that level and brand or ingredient with in_primary = 1 (all drugs: faers_report.in_primary_analysis = 1); FAERS denominators are reports with has_clinical_pt = 1 and numerators count reactions with clinical = 1; when comparing with FAERS, keep statements dated from the brand's approval date (an ingredient's first approval) to 2026-06-30 with nonclinical_group IS NULL. Describe FAERS and Reddit/WebMD shares side by side; never compute ratios or differences between them. Call one share higher than another only when their 95% intervals do not overlap. Groups with a denominator below 10 support no conclusion; below 30, call them small.

Concepts: writers' wordings for one meaning are often coded to several PTs or left uncoded (pt NULL); build each concept from all PTs and uncoded wordings that express it, and name its members. A purpose PT names either the goal (Weight decreased, Weight control, Blood glucose decreased) or the condition the drug is taken for (Obesity, Overweight, Weight, Weight increased, Type 2 diabetes mellitus, Glycosylated haemoglobin increased); a goal and the condition it addresses are one reason and belong to one concept. Count a PT or wording only when its wordings mean the asked concept; a symptom that often comes with it (pain with constipation, choking with vomiting) stays out. Never report a count you know to be partial; finish the analysis instead. When the question rests on a premise the data contradict, say so with the numbers."""


def instructions(variant='rules'):
    """'rules': the agent's data facts, counting rules and medical-advice policy (the strongest fair baseline); 'naive': the
    file and table description only, as a user would give after uploading the data."""
    if variant == 'naive':
        return ('You answer questions about patient-reported purposes and side effects of GLP-1 receptor agonists from the data '
                'below, using the run_python tool to query it.\n\n' + DATA.split('\n\nCounting rules:', 1)[0] +
                '\n\nFinal answer (JSON): answer = markdown that answers the question; answer_type and limitations as defined '
                'in the schema.')
    policy = agent.INSTRUCTIONS.split('Medical questions:', 1)[1].split('How to work:', 1)[0].strip()
    return ('You answer questions about patient-reported purposes and side effects of GLP-1 receptor agonists from the data '
            'below, using the run_python tool to query it. Use only results you computed from the data for any statement about '
            'the data.\n\n' + DATA + '\n\nMedical questions: ' + policy +
            '\n\nFinal answer (JSON): answer = concise markdown that answers the question directly, separated by platform, with '
            'counts and denominators for the shares you report (say in plain words what each denominator counts) and the caveats '
            'specific to the question; quote writers only from the evidence sentences, a short clause that shows exactly what '
            'the question asks, never ids or personal details; answer_type and limitations as '
            'defined in the schema. Keep the answer under about 220 words.')


INSTRUCTIONS = {'rules': instructions('rules'), 'naive': instructions('naive')}


SETUP_NO_SENTENCES = '\n'.join(l for l in SETUP.splitlines() if not l.startswith(('KG_DB', 'kg = ')))
# The agent's sandbox also gets temporary views that apply the RQ2 comparison rules, so code does not re-derive them.
AGENT_VIEWS = '''
db.executescript("""
CREATE TEMP VIEW faers_scope AS
  SELECT 'all' AS scope, r.report_id, r.has_clinical_pt FROM faers_report r WHERE r.in_primary_analysis = 1
  UNION ALL SELECT DISTINCT 'brand:' || s.brand, s.report_id, r.has_clinical_pt FROM faers_stratum s
    JOIN faers_report r ON r.report_id = s.report_id WHERE s.level = 'brand' AND s.in_primary = 1
  UNION ALL SELECT DISTINCT 'ingredient:' || s.ingredient, s.report_id, r.has_clinical_pt FROM faers_stratum s
    JOIN faers_report r ON r.report_id = s.report_id WHERE s.level = 'ingredient' AND s.in_primary = 1
  UNION ALL SELECT DISTINCT 'unspecified_brand:' || s.ingredient, s.report_id, r.has_clinical_pt FROM faers_stratum s
    JOIN faers_report r ON r.report_id = s.report_id WHERE s.level = 'unspecified' AND s.in_primary = 1;
CREATE TEMP VIEW scope_start AS
  SELECT 'all' AS scope, min(approval_date) AS start FROM approval
  UNION ALL SELECT 'brand:' || brand, approval_date FROM approval
  UNION ALL SELECT 'ingredient:' || ingredient, min(approval_date) FROM approval GROUP BY ingredient
  UNION ALL SELECT 'unspecified_brand:' || ingredient, min(approval_date) FROM approval GROUP BY ingredient;
CREATE TEMP VIEW comparable_statement AS
  SELECT ss.scope, st.* FROM main.statement st JOIN scope_start ss ON
    (ss.scope = 'all' OR ss.scope = 'brand:' || st.brand OR ss.scope = 'ingredient:' || st.ingredient
     OR (ss.scope = 'unspecified_brand:' || st.ingredient AND st.brand = 'unspecified'))
  WHERE st.role = 'side_effect' AND st.nonclinical_group IS NULL AND st.date >= ss.start AND st.date <= '2026-06-30';
""")
'''
# residual(): the agent's concept review in code (weightloss.agent.vocabulary). It does nothing while earlier cells are
# replayed; otherwise it prints the review and appends it as JSON to residual.jsonl, which Sandbox.run collects for the
# verifier.
RESIDUAL_HELPER = r'''
def residual(where, counted, table="statement"):
    """Concept review: rows of `table` matching `where` (the denominator's statements) that `counted` (the SQL condition
    of every counted member, all concepts joined with OR) leaves out."""
    if _REPLAY:
        return None
    import importlib.util as _iu, json as _json, re as _re
    _spec = _iu.spec_from_file_location("_vocabulary", _VOCABULARY)
    _v = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_v)
    if table not in ("statement", "comparable_statement"):
        raise ValueError("table must be 'statement' or 'comparable_statement'")
    base = f"FROM {table} WHERE ({where})"
    inside = f"COALESCE(({counted}), 0)"  # a NULL comparison (an uncoded row's pt_code) means not counted
    q = lambda sql, args=(): db.execute(sql, args).fetchall()
    roles = sorted(r for (r,) in q(f"SELECT DISTINCT role {base}"))
    universe = dict(q(f"SELECT platform, COUNT(DISTINCT record_key) {base} GROUP BY platform"))
    platforms = [p for p in ("reddit", "webmd") if p in universe]
    counted_n = dict(q(f"SELECT platform, COUNT(DISTINCT record_key) {base} AND {inside} GROUP BY platform"))
    names = []
    for c in sorted(set(_re.findall(r"\b1\d{7}\b", counted))):
        hit = q("SELECT pt FROM meddra WHERE pt_code = ? UNION SELECT hlt FROM meddra WHERE hlt_code = ? UNION "
                "SELECT hlgt FROM meddra WHERE hlgt_code = ? UNION SELECT soc FROM meddra WHERE soc_code = ? LIMIT 1", (c,) * 4)
        names += [h[0] for h in hit]
    for lit in _re.findall(r"'([^']{3,80})'", counted):
        if q("SELECT 1 FROM meddra WHERE lower(pt) = lower(?) OR lower(hlt) = lower(?) OR lower(hlgt) = lower(?) "
             "OR lower(soc) = lower(?) LIMIT 1", (lit,) * 4):
            names.append(lit)
    patterns = _v.patterns_in(counted)
    for x in patterns:  # each alternative of a counting pattern names a member (exclusion patterns name none)
        if not _re.search(r"NOT\s+(?:\w+\s+)?REGEXP\s*['\"]" + _re.escape(x), counted, _re.I):
            names += [_re.sub(r"\\[a-z]|[^A-Za-z0-9 ]+", " ", alt) for alt in _re.split(r"\|", x)]
    wordings = q(f"SELECT lower(trim(wording)), COUNT(DISTINCT record_key) c {base} AND {inside} GROUP BY 1 ORDER BY c DESC LIMIT 300")
    vocab = _v.counted_words(names, wordings, max(counted_n.values(), default=0))
    examples = {}
    for p, code, w, c in q(f"SELECT platform, pt_code, lower(trim(wording)), COUNT(DISTINCT record_key) c {base} "
                           f"AND pt_code IS NOT NULL AND NOT {inside} GROUP BY 1, 2, 3 ORDER BY c DESC"):
        examples.setdefault((p, code), []).append(w)
    rows = [{"platform": p, "kind": "pt", "code": code, "name": pt, "records": n, "common_wordings": examples.get((p, code), [])[:2]}
            for p, code, pt, n in q(f"SELECT platform, pt_code, pt, COUNT(DISTINCT record_key) n {base} AND pt_code IS NOT NULL "
                                    f"AND NOT {inside} GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 1500")]
    rows += [{"platform": p, "kind": "wording", "code": "uncoded", "name": w, "records": n}
             for p, w, n in q(f"SELECT platform, lower(trim(wording)), COUNT(DISTINCT record_key) n {base} AND pt_code IS NULL "
                              f"AND NOT {inside} GROUP BY 1, 2 ORDER BY n DESC LIMIT 1500")]
    counted_codes = [c for (c,) in q(f"SELECT DISTINCT pt_code {base} AND {inside} AND pt_code IS NOT NULL")]
    rows = _v.mark(rows, vocab, _v.closeness(counted_codes, [r["code"] for r in rows if r["kind"] == "pt"]))
    related = [r for r in rows if r["related_to"]]
    if related:
        done = set(q(f"SELECT DISTINCT platform, record_key {base} AND {inside}"))
        codes = sorted({r["code"] for r in related if r["kind"] == "pt"})
        texts = sorted({r["name"] for r in related if r["kind"] == "wording"})
        keys = {}
        for p, code, w, k in q(f"SELECT DISTINCT platform, pt_code, CASE WHEN pt_code IS NULL THEN lower(trim(wording)) END, "
                               f"record_key {base} AND NOT {inside} AND (pt_code IN ({','.join('?' * len(codes)) or 'NULL'}) "
                               f"OR (pt_code IS NULL AND lower(trim(wording)) IN ({','.join('?' * len(texts)) or 'NULL'})))",
                               tuple(codes) + tuple(texts)):
            keys.setdefault((p, code or "uncoded", w), set()).add((p, k))
        for r in related:
            r["new_records"] = len(keys.get((r["platform"], r["code"], None if r["kind"] == "pt" else r["name"]), set()) - done)
    shown = _v.select(platforms, rows, counted_n)
    counted_uncoded = [{"platform": p, "wording": w, "records": n,
                        "inside_word_only": any(_v.inside_word_only(x, w) for x in patterns)
                                            and not any(not _v.inside_word_only(x, w) and _re.search(x, w, _re.I) for x in patterns)}
                       for p, w, n in q(f"SELECT platform, lower(trim(wording)), COUNT(DISTINCT record_key) n {base} "
                                        f"AND {inside} AND pt_code IS NULL GROUP BY 1, 2 ORDER BY n DESC")]
    flagged = [u for u in counted_uncoded if u["inside_word_only"]]
    members = []
    for code, pt, n in q(f"SELECT pt_code, pt, COUNT(DISTINCT record_key) n {base} AND {inside} AND pt_code IS NOT NULL "
                         "GROUP BY 1, 2 ORDER BY n DESC LIMIT 25"):
        said = [w for (w,) in q(f"SELECT lower(trim(wording)) w {base} AND {inside} AND pt_code = ? GROUP BY 1 "
                                "ORDER BY COUNT(DISTINCT record_key) DESC LIMIT 3", (code,))]
        members.append({"pt": pt, "code": code, "records": n, "wordings_here": said})
    counted_uncoded = counted_uncoded[:40] + [u for u in counted_uncoded[40:] if u["inside_word_only"]]
    with open("residual.jsonl", "a") as fh:
        fh.write(_json.dumps({"where": where, "counted": counted, "table": table, "roles": roles, "universe": universe,
                              "counted_records": counted_n, "rows": shown, "counted_uncoded": counted_uncoded,
                              "counted_members": members}) + "\n")
    print(f"Concept review ({table}, roles {', '.join(roles)}):")
    for p in platforms:
        print(f"  {p}: {universe.get(p, 0):,} records in the filter, {counted_n.get(p, 0):,} counted; a related row marked * adds "
              f"at least {_v.check_threshold(counted_n.get(p, 0))} records and must be counted or named in the answer.")
    rel = [r for r in shown if r["related_to"]]
    print("Not counted, related to the counted members in wording or meaning (platform | row | records | records it would "
          "add | shared word, or BioLORD closeness in meaning: close is not the same, judge from the wordings):"
          if rel else "Not counted and related in wording or meaning: none.")
    for r in rel:
        label = f"PT {r['name']} [{r['code']}] ({'; '.join(r['common_wordings'])})" if r["kind"] == "pt" else f"uncoded \"{r['name']}\""
        print(f"{'*' if r['check'] else ' '} {r['platform']} | {label} | {r['records']} | {r['new_records']} | {r['related_to']}"
              + ("" if r["strong"] or r["related_to"].startswith("meaning") else " (weak: related through wordings only)"))
    print("Largest other rows not counted (check whether any says the same in other words):")
    for r in shown:
        if not r["related_to"]:
            label = f"PT {r['name']} [{r['code']}] ({'; '.join(r['common_wordings'])})" if r["kind"] == "pt" else f"uncoded \"{r['name']}\""
            print(f"  {r['platform']} | {label} | {r['records']}")
    if members:
        print("Counted PTs with their most frequent wordings here (drop any member whose wordings mean something else):")
        for m in members:
            print(f"  PT {m['pt']} [{m['code']}] | {m['records']} | {'; '.join(m['wordings_here'])}")
    if counted_uncoded:
        print("Uncoded wordings counted (check each belongs; ! = a pattern matches it only inside a longer word: fix the pattern):")
        for u in counted_uncoded:
            print(f"{'!' if u['inside_word_only'] else ' '} {u['platform']} | {u['wording']} | {u['records']}")
    return None
'''
SETUP_AGENT = SETUP_NO_SENTENCES + AGENT_VIEWS + f'_VOCABULARY = {str(Path(vocabulary.__file__))!r}\n' + RESIDUAL_HELPER


@functools.lru_cache(maxsize=1)
def agent_setup():
    """SETUP_AGENT plus the statement view in which uncoded statements that misspell a PT name carry that PT
    (weightloss.agent.spelling), as in query_graph; comparable_statement keeps the stored statements (RQ2: coded PTs only)."""
    from . import spelling
    return SETUP_AGENT + spelling.view_code(str(analysis_db.PATH))


def readable_paths():
    """What sandboxed code may read inside the user's home folder: the Python installation that runs it, the analysis
    database, the raw graph and the frozen FAERS files (the agent's sandbox denies the raw graph again), the vocabulary
    module and the BioLORD index that residual() loads. Everything else there (the project's .env, other files) is closed."""
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(sys.executable).resolve().parents[1]}
    paths = [('subpath', p) for p in sorted(prefixes)]
    paths += [('subpath', analysis_db.PATH.parent.resolve()), ('literal', Path(neo4j_db.KG).resolve()),
              ('subpath', Path(FAERS_DIR).resolve()), ('literal', Path(vocabulary.__file__).resolve()),
              ('subpath', vocabulary.INDEX.resolve())]
    return paths


SAFE_ENV = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ')


class Sandbox:
    """Reads under the user's home folder are limited to readable_paths(); deny_read closes more files (the agent's sandbox
    cannot read kg.sqlite, which holds the sentences). The network and writes outside the temporary folder are closed."""

    def __init__(self, setup=SETUP, deny_read=()):
        self.tmp = Path(tempfile.mkdtemp(prefix='code-baseline-'))
        (self.tmp / 'setup.py').write_text(setup)
        (self.tmp / 'runner.py').write_text(RUNNER)
        home = Path.home().resolve()
        closed = {Path(p).resolve() for p in deny_read}
        allow = ''.join(f'(allow file-read-data ({kind} "{path}"))\n' for kind, path in readable_paths() if path not in closed)
        deny = ''.join(f'(deny file-read-data (literal "{path}"))\n' for path in sorted(closed))
        # Later rules win: close the contents of the home folder, reopen what the code needs, then close the files denied
        # for this sandbox. Metadata stays readable, since opening a file looks up each folder on its path.
        (self.tmp / 'profile.sb').write_text(PROFILE.format(tmp=str(self.tmp)) + f'(deny file-read-data (subpath "{home}"))\n'
                                             + allow + deny)
        self.cells = []

    def env(self):
        """The environment of a cell: locale, PATH and HOME only. Model-written code never sees the server's own variables,
        which hold the OpenAI key once the agent has loaded it."""
        env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
        env['TMPDIR'] = str(self.tmp)
        return env

    def run(self, code):
        cells = self.cells + [code]
        path = self.tmp / f'cells-{uuid.uuid4().hex[:6]}.json'
        path.write_text(json.dumps(cells))
        cmd = ['sandbox-exec', '-f', str(self.tmp / 'profile.sb'), sys.executable, '-I', str(self.tmp / 'runner.py'), str(path),
               str(self.tmp / 'setup.py')]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=CELL_SECONDS, cwd=self.tmp, env=self.env())
            out, ok = (p.stdout + ('\n' + p.stderr if p.stderr else '')), p.returncode == 0
        except subprocess.TimeoutExpired:
            out, ok = f'Execution stopped after {CELL_SECONDS} s.', False
        if ok:
            self.cells.append(code)
        if len(out) > OUTPUT_CHARS:
            out = out[:OUTPUT_CHARS] + f'\n... [{len(out) - OUTPUT_CHARS} more characters not shown]'
        reviews = []
        marks = self.tmp / 'residual.jsonl'
        if marks.exists():
            reviews = [json.loads(line) for line in marks.read_text().splitlines() if line.strip()]
            marks.unlink()
        return {'output': out or '(no output)', 'ok': ok, 'seconds': round(time.time() - t0, 2), 'reviews': reviews}

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def request_body(items, allow_tools, variant='rules', effort=EFFORT):
    return {'model': MODEL, 'instructions': INSTRUCTIONS[variant], 'input': items, 'tools': TOOLS,
            'tool_choice': 'auto' if allow_tools else 'none', 'parallel_tool_calls': False,
            'text': {'format': {'type': 'json_schema', 'name': 'baseline_answer', 'strict': True, 'schema': FINAL_SCHEMA}},
            'reasoning': {'effort': effort}, 'include': ['reasoning.encrypted_content'], 'store': False,
            'max_output_tokens': agent.MAX_OUTPUT_TOKENS}


def run(question, graph=None, *, client=None, trace_path=None, label=None, budget_usd=agent.BUDGET_USD, variant='rules',
        effort=EFFORT):
    """Answer one question with the code baseline; returns a result dict shaped like agent.run's."""
    started = dt.datetime.now(dt.timezone.utc)
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    box = Sandbox()
    items = [{'role': 'user', 'content': f'Question: {question}'}]
    system = f'{VERSION}:{variant}'
    trace = {'run_id': run_id, 'label': label, 'system': system, 'started_utc': started.isoformat(timespec='seconds'),
             'question': question, 'model': MODEL, 'reasoning_effort': effort,
             'instructions_sha256': sha256_text(INSTRUCTIONS[variant]), 'tools_sha256': sha256_text(compact_json(TOOLS)),
             'instructions': INSTRUCTIONS[variant], 'responses': [], 'tool_calls': []}
    usage_total, cost, rounds, stop_reason, final = collections.Counter(), 0.0, 0, None, None
    try:
        if client is None:
            agent.load_key()
            from openai import OpenAI
            client = OpenAI(max_retries=5, timeout=300)
        while True:
            if cost >= budget_usd:
                stop_reason = f'budget reached (${cost:.4f} of ${budget_usd})'
                break
            if len(trace['responses']) >= MAX_TOOL_ROUNDS + 3:
                stop_reason = 'response cap reached'
                break
            t0 = time.time()
            try:
                raw = client.responses.create(**request_body(items, rounds < MAX_TOOL_ROUNDS, variant, effort)).model_dump()
            except Exception as error:
                stop_reason = f'transport_error: {type(error).__name__}: {mask(error)[:300]}'
                break
            usd = cost_usd(MODEL, raw.get('usage') or {}) or 0.0
            cost += usd
            for k, v in token_split(raw.get('usage') or {}).items():
                usage_total[k] += v
            trace['responses'].append({'seconds': round(time.time() - t0, 2), 'usd': usd, 'raw': raw})
            calls = [i for i in raw.get('output') or [] if i.get('type') == 'function_call']
            if calls and raw.get('status') == 'completed':
                rounds += 1
                items.extend(agent._strip_none(i) for i in raw['output'])
                for c in calls:
                    try:
                        code = json.loads(c['arguments'])['code']
                        res = box.run(code)
                    except (json.JSONDecodeError, KeyError) as error:
                        code, res = None, {'output': f'bad arguments: {error}', 'ok': False, 'seconds': 0}
                    trace['tool_calls'].append({'call': f't{len(trace["tool_calls"]) + 1}', 'tool': 'run_python',
                                                'arguments': {'code': code}, 'output': res})
                    items.append({'type': 'function_call_output', 'call_id': c['call_id'], 'output': res['output']})
                continue
            try:
                parsed = validate_model(BaselineAnswer, parse_json_object(final_output_text(raw)))
                final = {**parsed.model_dump(), 'citations': [], 'quotes': []}
            except ExtractionAnomaly as error:
                stop_reason = f'anomaly: {error.kind}'
            break
    finally:
        box.close()
        result = {'run_id': run_id, 'label': label, 'question': question, 'system': system, 'verified': None,
                  'answer': final, 'problems': [] if final else [stop_reason or 'no answer'], 'warnings': [],
                  'repair_used': False, 'stop_reason': stop_reason, 'tool_rounds': rounds,
                  'tool_calls': [{'call': c['call'], 'tool': c['tool'], 'arguments': c['arguments'], 'seconds': c['output']['seconds'],
                                  'error': None if c['output']['ok'] else c['output']['output'][-300:]} for c in trace['tool_calls']],
                  'usage': dict(usage_total), 'cost_usd': round(cost, 6)}
        trace.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), result=result)
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            with agent.TRACE_LOCK, open(trace_path, 'a') as f:
                f.write(mask(json.dumps(trace, ensure_ascii=False, default=str)) + '\n')
    return result
