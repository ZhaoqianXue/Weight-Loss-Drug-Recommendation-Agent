"""Read-only analysis database: the knowledge graph and the frozen FAERS layer as flat SQLite tables.

  python -m weightloss.graph.analysis_db build

The agent's run_sql tool and the code-agent baseline query this file. It holds the same content as the Neo4j copy
(weightloss.graph.neo4j_db), flattened so that one row is one statement, one FAERS report, stratum or reaction:

- statement: one row per USED_FOR (role 'purpose') or SIDE_EFFECT (role 'side_effect') edge of kg.sqlite, with the
  record key, date and year, drug (ingredient, brand, ingredient_basis), writer age group and sex, the source of a
  purpose ('profile' for a WebMD profile condition, else 'text'), the writer's wording, the PT and its MedDRA 29.1
  primary path, the non-clinical group flag, side-effect details with one 0/1 column per action taken, and the
  parsed dose (dose_mg) and duration bucket. Source sentences are not included.
- faers_report, faers_stratum, faers_reaction: the validated FAERS layer (results/fda-validation-20261017/frozen).
- meddra: the primary path of every PT; approval: FDA approval date per brand.
Nothing here calls a model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from .build import BRAND_INGREDIENT
from .neo4j_db import FROZEN, KG, ROOT, approval_dates, dose_mg, duration_bucket, sha256

VERSION = 'analysis-db-1.0.0'
PATH = ROOT / '.cache/analysis/analysis.sqlite'
BRAND_BY_LOWER = {b.lower(): b for b in BRAND_INGREDIENT}

SCHEMA = """
CREATE TABLE statement (
  role TEXT, edge_id TEXT PRIMARY KEY, platform TEXT, record_id TEXT, record_key TEXT, date TEXT, year TEXT,
  drug_id TEXT, ingredient TEXT, brand TEXT, ingredient_basis TEXT, writer_age_group TEXT, writer_sex TEXT, source TEXT,
  wording TEXT, pt_code TEXT, pt TEXT, hlt_code TEXT, hlt TEXT, hlgt_code TEXT, hlgt TEXT, soc_code TEXT, soc TEXT,
  nonclinical_group TEXT, intensity TEXT, impact TEXT, action_taken TEXT, stopped INTEGER, dose_reduced INTEGER,
  dose_skipped INTEGER, escalation_paused INTEGER, dose_text TEXT, dose_mg REAL, duration_text TEXT, duration_bucket TEXT);
CREATE TABLE faers_report (
  report_id TEXT PRIMARY KEY, receivedate TEXT, year TEXT, sex TEXT, age_group TEXT, hospitalization INTEGER,
  in_primary_analysis INTEGER, has_clinical_pt INTEGER);
CREATE TABLE faers_stratum (
  report_id TEXT, level TEXT, ingredient TEXT, brand TEXT, in_primary INTEGER, actions TEXT, action_known INTEGER,
  withdrawn INTEGER, dose_reduced INTEGER, window_start TEXT);
CREATE TABLE faers_reaction (
  report_id TEXT, pt_code TEXT, pt TEXT, hlt_code TEXT, hlt TEXT, hlgt_code TEXT, hlgt TEXT, soc_code TEXT, soc TEXT,
  nonclinical_group TEXT, clinical INTEGER);
CREATE TABLE meddra (pt_code TEXT PRIMARY KEY, pt TEXT, hlt_code TEXT, hlt TEXT, hlgt_code TEXT, hlgt TEXT, soc_code TEXT,
  soc TEXT, nonclinical_group TEXT);
CREATE TABLE approval (brand TEXT PRIMARY KEY, ingredient TEXT, approval_date TEXT);
"""
INDEXES = [
    'CREATE INDEX st_role_platform ON statement(role, platform)', 'CREATE INDEX st_pt ON statement(pt_code)',
    'CREATE INDEX st_record ON statement(record_key)', 'CREATE INDEX st_brand ON statement(brand)',
    'CREATE INDEX st_ingredient ON statement(ingredient)', 'CREATE INDEX st_hlt ON statement(hlt_code)',
    'CREATE INDEX st_soc ON statement(soc_code)', 'CREATE INDEX fs_report ON faers_stratum(report_id)',
    'CREATE INDEX fs_scope ON faers_stratum(level, ingredient, brand)', 'CREATE INDEX fr_report ON faers_reaction(report_id)',
    'CREATE INDEX fr_pt ON faers_reaction(pt_code)',
]


def _gz(path):
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f)


def build(out=PATH, kg=KG):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript(SCHEMA)
    hierarchy = {r['pt_code']: r for r in csv.DictReader(open(FROZEN / 'meddra_pt_hierarchy.csv', newline='', encoding='utf-8'))}
    path = lambda code: (lambda h: [h.get('hlt_code'), h.get('hlt_name'), h.get('hlgt_code'), h.get('hlgt_name'), h.get('soc_code'),
                                    h.get('soc_name'), h.get('nonclinical_group') or None])(hierarchy.get(code or '', {}))
    db.executemany('INSERT INTO meddra VALUES (?,?,?,?,?,?,?,?,?)',
                   [[c, h['pt_name']] + path(c) for c, h in hierarchy.items()])
    db.executemany('INSERT INTO approval VALUES (?,?,?)', [(b, BRAND_INGREDIENT[b], d) for b, d in approval_dates().items()])

    kgdb = sqlite3.connect(f'file:{kg}?mode=ro', uri=True)
    kgdb.row_factory = sqlite3.Row
    rows = []
    for role, table, col, node in (('purpose', 'edge_used_for', 'purpose_id', 'purpose'),
                                   ('side_effect', 'edge_side_effect', 'side_effect_id', 'side_effect')):
        se = role == 'side_effect'
        for r in kgdb.execute(f'SELECT e.*, d.ingredient, d.brand, n.pt, n.pt_code FROM {table} e JOIN drug d ON d.id = e.drug_id '
                              f'JOIN {node} n ON n.id = e.{col}'):
            spans = json.loads(r['evidence'])
            source = 'profile' if spans and all(s.get('field') == 'profile_condition' for s in spans) else 'text'
            acts = json.loads(r['action_taken']) if se else []
            rows.append([role, r['id'], r['platform'], r['record_id'], f"{r['platform']}:{r['record_id']}", r['date'],
                         (r['date'] or '')[:4] or None, r['drug_id'], r['ingredient'], r['brand'], r['ingredient_basis'],
                         r['writer_age_group'], r['writer_sex'], source, r['text'], r['pt_code'], r['pt']] + path(r['pt_code'])
                        + ([r['intensity'], r['impact'], json.dumps(acts), int('stopped' in acts), int('dose_reduced' in acts),
                            int('dose_skipped' in acts), int('escalation_paused' in acts), r['dose'], dose_mg(r['dose']),
                            r['duration'], duration_bucket(r['duration'])] if se else [None] * 11))
    db.executemany(f'INSERT INTO statement VALUES ({",".join("?" * 35)})', rows)
    counts = {'statement': len(rows)}
    kgdb.close()

    clinical = set()
    reactions = []
    for r in _gz(FROZEN / 'faers_report_pts.csv.gz'):
        if not r['pt_code']:
            continue
        group = hierarchy.get(r['pt_code'], {}).get('nonclinical_group') or None
        if group is None:
            clinical.add(r['safetyreportid'])
        name = hierarchy[r['pt_code']]['pt_name'] if r['pt_code'] in hierarchy else r['pt_name']
        reactions.append([r['safetyreportid'], r['pt_code'], name] + path(r['pt_code'])[:6] + [group, int(group is None)])
    db.executemany('INSERT INTO faers_reaction VALUES (?,?,?,?,?,?,?,?,?,?,?)', reactions)
    reports = []
    for r in _gz(FROZEN / 'faers_reports.csv.gz'):
        d = r['receivedate']
        date = f'{d[:4]}-{d[4:6]}-{d[6:]}' if len(d) == 8 else None
        reports.append([r['safetyreportid'], date, d[:4] or None, r['sex'] or None, r['age_group'] or None,
                        int(r['hospitalization'] == 'yes'), int(r['in_primary_analysis'] == 'True'), int(r['safetyreportid'] in clinical)])
    db.executemany('INSERT INTO faers_report VALUES (?,?,?,?,?,?,?,?)', reports)
    strata = []
    for r in _gz(FROZEN / 'faers_report_strata.csv.gz'):
        acts = [a for a in (r['actiondrug_primary'] or '').split(';') if a]
        brand = BRAND_BY_LOWER[r['brand']] if r['level'] == 'brand' else ('unspecified' if r['level'] == 'unspecified' else None)
        ws = r['window_start']
        strata.append([r['safetyreportid'], r['level'], r['ingredient'], brand, int(r['in_primary'] == 'True'), ';'.join(acts),
                       int(bool(set(acts) & {'1', '2', '3', '4'})), int('1' in acts), int('2' in acts),
                       f'{ws[:4]}-{ws[4:6]}-{ws[6:]}' if len(ws) == 8 else None])
    db.executemany('INSERT INTO faers_stratum VALUES (?,?,?,?,?,?,?,?,?,?)', strata)
    counts.update(faers_report=len(reports), faers_stratum=len(strata), faers_reaction=len(reactions), meddra=len(hierarchy))
    for q in INDEXES:
        db.execute(q)
    manifest = {'version': VERSION, 'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'counts': counts,
                'inputs_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(kg), FROZEN / 'meddra_pt_hierarchy.csv', FROZEN / 'faers_reports.csv.gz',
                                   FROZEN / 'faers_report_strata.csv.gz', FROZEN / 'faers_report_pts.csv.gz']},
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    db.execute('CREATE TABLE build (key TEXT, value TEXT)')
    db.executemany('INSERT INTO build VALUES (?,?)', [(k, json.dumps(v)) for k, v in manifest.items()])
    db.commit()
    db.close()
    tmp.replace(out)
    return manifest


def connect(path=PATH):
    """A read-only connection with the helper functions the tools document: wilson_low/high(k, n) give the 95% Wilson
    interval of k/n in percent, and REGEXP matches case-insensitively."""
    import math
    import re
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True, check_same_thread=False)
    db.execute('PRAGMA query_only = 1')

    def wilson(k, n, sign):
        if not n:
            return None
        p, z = k / n, 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return round(100 * min(1.0, max(0.0, centre + sign * half)), 2)

    db.create_function('wilson_low', 2, lambda k, n: wilson(k, n, -1), deterministic=True)
    db.create_function('wilson_high', 2, lambda k, n: wilson(k, n, 1), deterministic=True)
    db.create_function('regexp', 2, lambda pattern, value: value is not None and re.search(pattern, value, re.I) is not None,
                       deterministic=True)
    return db


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['build'])
    ap.parse_args(argv)
    print(json.dumps(build(), indent=1)[:1500])


if __name__ == '__main__':
    main()
