"""Local Neo4j copy of the merged knowledge graph, for the agent (project document Sections 3.4 and 3.7).

  python -m weightloss.graph.neo4j_db start | stop | status
  python -m weightloss.graph.neo4j_db load [--kg <kg.sqlite>]

The instance is project-local: configuration, data and logs live under .cache/neo4j, it listens on 127.0.0.1 only
(bolt 7688, http 7475) and runs the Homebrew Neo4j 2025.04 Community server with authentication off. `load` exports
CSV files, imports them with neo4j-admin (the server is stopped for the import and started again), creates the
indexes and writes a manifest. Nothing here calls a model.

Graph = the Section 3.4 graph unchanged plus three layers taken from frozen inputs:
- Section 3.4: Drug, Purpose and SideEffect nodes (ids as in kg.sqlite, uncoded nodes without pt/pt_code) and one
  USED_FOR / SIDE_EFFECT relationship per statement with every edge property; record_key = platform:record_id and
  order_key = sha256(edge id) are added for record-level counting and deterministic example order, source ('profile' or
  'text') tells whether a statement comes from the WebMD profile or from the text, and each coded PT node carries its
  most frequent wordings (common_wordings, lowercased, by statement count).
- Side-effect statements also carry dose_mg (a free-text dose that is only a number with an optional mg unit, 0.1-20) and
  duration_bucket (under 1 day, 1-6 days, 1-4 weeks, over 4 weeks, from the first number and time unit of the free-text
  duration); both are empty when the text does not allow it.
- MedDRA 29.1 primary path: PT -IN_HLT-> HLT -IN_HLGT-> HLGT -IN_SOC-> SOC, from the FAERS validation's frozen
  meddra_pt_hierarchy.csv (one primary path per PT, with its nonclinical_group flag); Drug -HAS_INGREDIENT-> Ingredient.
- FAERS (validated and frozen, results/fda-validation-20261017/frozen): Report nodes; REPORTS_DRUG to the brand or
  unspecified-brand Drug node and REPORTS_INGREDIENT to the Ingredient node, one per stratum row with its configuration
  flags and action codes; REACTION to the SideEffect node of each report PT (MedDRA 29.1). FAERS-only PTs get their own
  SideEffect node.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

from . import VERSION as KG_VERSION
from .build import BRAND_INGREDIENT, UNSPECIFIED

ROOT = Path(__file__).resolve().parents[3]
HOME = ROOT / '.cache/neo4j'
CONF_DIR = HOME / 'conf'
EXPORT_DIR = HOME / 'import'
MANIFEST = HOME / 'manifest.json'
KG = ROOT / '.cache/reextraction-20261016/kg/full-v7/kg.sqlite'
FROZEN = ROOT / 'results/fda-validation-20261017/frozen'
APPROVAL_CONFIG = ROOT / 'configs/reddit.json'


def installation():
    """(neo4j, neo4j-admin, stock conf directory) of the installed Neo4j 5 or 2025: $NEO4J_HOME when set (a tarball
    install), else the `neo4j` on PATH (Homebrew keeps the stock conf in libexec/conf). None when Neo4j is not installed."""
    home = os.environ.get('NEO4J_HOME')
    if home:
        bin_dir = Path(home) / 'bin'
        neo4j, admin = bin_dir / 'neo4j', bin_dir / 'neo4j-admin'
    else:
        found = shutil.which('neo4j')
        if not found:
            return None
        neo4j = Path(found)
        admin = Path(shutil.which('neo4j-admin') or neo4j.with_name('neo4j-admin'))
    prefix = neo4j.resolve().parent.parent
    for conf in (prefix / 'libexec/conf', prefix / 'conf'):
        if (conf / 'neo4j.conf').is_file():
            return str(neo4j), str(admin), conf
    return None


def _installed():
    found = installation()
    if found is None:
        raise SystemExit('Neo4j is not installed. On macOS: brew install neo4j (it brings Java 21); elsewhere install '
                         'Neo4j 5 or 2025 and set NEO4J_HOME to its folder.')
    return found
BOLT_PORT, HTTP_PORT = 7688, 7475
URI = f'bolt://127.0.0.1:{BOLT_PORT}'
LOADER_VERSION = 'neo4j-load-1.3.0'
INGREDIENTS = ('semaglutide', 'tirzepatide', 'liraglutide', 'dulaglutide')
BRAND_BY_LOWER = {b.lower(): b for b in BRAND_INGREDIENT}

CONF = """# Project-local Neo4j for the AskGLP1 agent (written by weightloss.graph.neo4j_db; do not edit by hand)
server.directories.data={home}/data
server.directories.logs={home}/logs
server.directories.run={home}/run
server.directories.import={home}/import
server.directories.transaction.logs.root={home}/data/transactions
server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=127.0.0.1:{bolt}
server.bolt.advertised_address=127.0.0.1:{bolt}
server.http.enabled=true
server.http.listen_address=127.0.0.1:{http}
server.http.advertised_address=127.0.0.1:{http}
server.https.enabled=false
dbms.security.auth_enabled=false
server.memory.heap.initial_size=1g
server.memory.heap.max_size=2g
server.memory.pagecache.size=1g
db.tx_log.rotation.retention_policy=2 files
"""

# Indexes created after the import. Full-text indexes use the English analyzer (stemming), so "vomited" finds "vomit".
INDEXES = [
    'CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (n:Drug) REQUIRE n.id IS UNIQUE',
    'CREATE CONSTRAINT purpose_id IF NOT EXISTS FOR (n:Purpose) REQUIRE n.id IS UNIQUE',
    'CREATE CONSTRAINT side_effect_id IF NOT EXISTS FOR (n:SideEffect) REQUIRE n.id IS UNIQUE',
    'CREATE CONSTRAINT report_id IF NOT EXISTS FOR (n:Report) REQUIRE n.id IS UNIQUE',
    'CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.name IS UNIQUE',
    'CREATE CONSTRAINT hlt_code IF NOT EXISTS FOR (n:HLT) REQUIRE n.code IS UNIQUE',
    'CREATE CONSTRAINT hlgt_code IF NOT EXISTS FOR (n:HLGT) REQUIRE n.code IS UNIQUE',
    'CREATE CONSTRAINT soc_code IF NOT EXISTS FOR (n:SOC) REQUIRE n.code IS UNIQUE',
    'CREATE INDEX purpose_pt_code IF NOT EXISTS FOR (n:Purpose) ON (n.pt_code)',
    'CREATE INDEX side_effect_pt_code IF NOT EXISTS FOR (n:SideEffect) ON (n.pt_code)',
    'CREATE INDEX used_for_record IF NOT EXISTS FOR ()-[r:USED_FOR]-() ON (r.record_key)',
    'CREATE INDEX side_effect_record IF NOT EXISTS FOR ()-[r:SIDE_EFFECT]-() ON (r.record_key)',
    'CREATE INDEX used_for_id IF NOT EXISTS FOR ()-[r:USED_FOR]-() ON (r.id)',
    'CREATE INDEX side_effect_edge_id IF NOT EXISTS FOR ()-[r:SIDE_EFFECT]-() ON (r.id)',
    "CREATE FULLTEXT INDEX pt_names IF NOT EXISTS FOR (n:Purpose|SideEffect) ON EACH [n.pt] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
    "CREATE FULLTEXT INDEX group_names IF NOT EXISTS FOR (n:HLT|HLGT|SOC) ON EACH [n.name] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
    "CREATE FULLTEXT INDEX verbatims IF NOT EXISTS FOR ()-[r:USED_FOR|SIDE_EFFECT]-() ON EACH [r.text] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
    # the cited source sentences (JSON spans): search by what a writer said around the extracted wording
    "CREATE FULLTEXT INDEX sentences IF NOT EXISTS FOR ()-[r:USED_FOR|SIDE_EFFECT]-() ON EACH [r.evidence] "
    "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def approval_dates():
    """FDA approval dates (configs/reddit.json, the dates that set the Reddit collection start and the FAERS window)."""
    return json.loads(APPROVAL_CONFIG.read_text())['brands']


# ---------------------------------------------------------------------------------------------------- server control
def _env():
    write_conf()
    return {**os.environ, 'NEO4J_CONF': str(CONF_DIR)}


def write_conf():
    """neo4j.conf = CONF plus the stock JVM options; neo4j-admin.conf and the log configs are copied unchanged."""
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    for d in ('data', 'logs', 'run', 'import'):
        (HOME / d).mkdir(parents=True, exist_ok=True)
    default_conf = _installed()[2]
    stock = (default_conf / 'neo4j.conf').read_text().splitlines()
    jvm = [l for l in stock if l.startswith('server.jvm.additional=') and 'udc.source' not in l]
    text = CONF.format(home=HOME, bolt=BOLT_PORT, http=HTTP_PORT) + '\n'.join(jvm) + '\n'
    path = CONF_DIR / 'neo4j.conf'
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    for name in ('neo4j-admin.conf', 'server-logs.xml', 'user-logs.xml'):
        target = CONF_DIR / name
        if not target.exists():
            shutil.copyfile(default_conf / name, target)


def _run(cmd, check=True):
    return subprocess.run(cmd, env=_env(), check=check, capture_output=True, text=True)


def status():
    if installation() is None:  # CI and fresh machines: no database to ask
        return 'Neo4j is not installed'
    return _run([_installed()[0], 'status'], check=False).stdout.strip()


def is_running():
    return 'is running' in status()


def connect(timeout=30):
    """A driver for the local instance (raises when it does not answer within `timeout` seconds)."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(URI, auth=None)
    deadline = time.time() + timeout
    while True:
        try:
            driver.verify_connectivity()
            return driver
        except Exception:
            if time.time() > deadline:
                driver.close()
                raise
            time.sleep(1)


def start(wait=120):
    if not is_running():
        _run([_installed()[0], 'start'])
    connect(timeout=wait).close()


def stop():
    if is_running():
        _run([_installed()[0], 'stop'])


# ---------------------------------------------------------------------------------------------------- export
DOSE = re.compile(r'^\s*(\d*\.?\d+)\s*(?:mg|mgs|milligrams?)?\s*$', re.I)
NUMBER_WORDS = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                'nine': 9, 'ten': 10, 'couple': 2, 'few': 3, 'several': 4}
UNIT_DAYS = {'hour': 1 / 24, 'hr': 1 / 24, 'night': 1, 'day': 1, 'week': 7, 'wk': 7, 'month': 30, 'year': 365}
DURATION = re.compile(r'(\d+(?:\.\d+)?|' + '|'.join(NUMBER_WORDS) + r')?(?:\s*(?:-|to|or)\s*(\d+(?:\.\d+)?))?\s*'
                      r'(?:of\s+)?(hours?|hrs?|nights?|days?|weeks?|wks?|months?|years?)\b', re.I)
DURATION_BUCKETS = ('under 1 day', '1-6 days', '1-4 weeks', 'over 4 weeks')


def dose_mg(text):
    """Dose in mg from a free-text dose that is only a number with an optional mg unit (GLP-1 doses are all in mg),
    kept when 0.1 to 20 mg; otherwise None."""
    m = DOSE.match(text or '')
    if not m:
        return None
    value = float(m.group(1))
    return value if 0.1 <= value <= 20 else None


def duration_bucket(text):
    """Coarse bucket of a free-text symptom duration from its first number and time unit ("2-3 days" -> 1-6 days,
    "a few weeks" -> 1-4 weeks, "all night" -> 1-6 days, a bare plural such as "months" counts as two); None when no time
    unit is named (e.g. "quickly")."""
    m = DURATION.search((text or '').lower())
    if not m:
        return None
    first, second, unit = m.groups()
    n = float(second or (NUMBER_WORDS.get(first) if first in NUMBER_WORDS else first) or (2 if unit.endswith('s') else 1))
    days = n * UNIT_DAYS[unit.rstrip('s') if unit.rstrip('s') in UNIT_DAYS else unit[:-1]]
    return DURATION_BUCKETS[0] if days < 1 else DURATION_BUCKETS[1] if days < 7 else DURATION_BUCKETS[2] if days <= 31 else DURATION_BUCKETS[3]


def _writer(path, header):
    f = open(path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\n')
    w.writerow(header)
    return f, w


def _read_gz(path):
    with gzip.open(path, 'rt', newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f)


def export(kg=KG, out=EXPORT_DIR):
    """Write the node and relationship CSV files for neo4j-admin; returns {file: rows} and the input hashes."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = {}
    con = sqlite3.connect(f'file:{kg}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row

    hierarchy = {}
    with open(FROZEN / 'meddra_pt_hierarchy.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            hierarchy[r['pt_code']] = r

    # Drug and Ingredient nodes
    f, w = _writer(out / 'nodes_drug.csv', ['id:ID', 'ingredient', 'brand', ':LABEL'])
    drugs = sorted(con.execute('SELECT id, ingredient, brand FROM drug'), key=lambda r: r['id'])
    for r in drugs:
        w.writerow([r['id'], r['ingredient'], r['brand'], 'Drug'])
    f.close()
    counts['nodes_drug'] = len(drugs)
    f, w = _writer(out / 'nodes_ingredient.csv', ['id:ID', 'name', ':LABEL'])
    for i in INGREDIENTS:
        w.writerow([f'ingredient:{i}', i, 'Ingredient'])
    f.close()
    counts['nodes_ingredient'] = len(INGREDIENTS)
    f, w = _writer(out / 'rels_has_ingredient.csv', [':START_ID', ':END_ID', ':TYPE'])
    n = 0
    for r in drugs:
        if r['ingredient'] != UNSPECIFIED:
            w.writerow([r['id'], f"ingredient:{r['ingredient']}", 'HAS_INGREDIENT'])
            n += 1
    f.close()
    counts['rels_has_ingredient'] = n

    # PT nodes: patient purposes and side effects (kg.sqlite) plus FAERS-only side-effect PTs
    faers_pts = {}
    for r in _read_gz(FROZEN / 'faers_report_pts.csv.gz'):
        if r['pt_code']:
            faers_pts.setdefault(r['pt_code'], r['pt_name'])
    pt_nodes = []
    for kind, label in (('purpose', 'Purpose'), ('side_effect', 'SideEffect')):
        seen = set()
        for r in con.execute(f'SELECT id, pt, pt_code FROM {kind} ORDER BY id'):
            pt_nodes.append((r['id'], r['pt'], r['pt_code'], label))
            seen.add(r['pt_code'])
        if kind == 'side_effect':
            for code in sorted(set(faers_pts) - seen):
                name = hierarchy[code]['pt_name'] if code in hierarchy else faers_pts[code]
                pt_nodes.append((f'side_effect:pt:{code}', name, code, label))
    wordings = {}
    for kind, table, col in (('purpose', 'edge_used_for', 'purpose_id'), ('side_effect', 'edge_side_effect', 'side_effect_id')):
        top = {}
        for r in con.execute(f'SELECT {col} AS node, lower(trim(text)) AS t, count(*) AS c FROM {table} GROUP BY 1, 2'):
            top.setdefault(r['node'], []).append((-r['c'], r['t']))
        wordings.update({node: [t for _, t in sorted(v)[:3]] for node, v in top.items()})
    f, w = _writer(out / 'nodes_pt.csv', ['id:ID', 'pt', 'pt_code', 'nonclinical_group', 'common_wordings:string[]', ':LABEL'])
    for node_id, pt, code, label in pt_nodes:
        group = hierarchy.get(code, {}).get('nonclinical_group', '') if code else ''
        common = ';'.join(x.replace(';', ',') for x in wordings.get(node_id, [])) if code else ''
        w.writerow([node_id, pt or '', code or '', group, common, label])
    f.close()
    counts['nodes_pt'] = len(pt_nodes)

    # MedDRA primary path for every coded PT node
    hlt, hlgt, soc = {}, {}, {}
    f, w = _writer(out / 'rels_in_hlt.csv', [':START_ID', ':END_ID', ':TYPE'])
    n = 0
    for node_id, _, code, _ in pt_nodes:
        if not code:
            continue
        h = hierarchy.get(code)
        if h is None:
            raise SystemExit(f'PT {code} has no MedDRA 29.1 primary path in meddra_pt_hierarchy.csv')
        hlt[h['hlt_code']] = (h['hlt_name'], h['hlgt_code'])
        hlgt[h['hlgt_code']] = (h['hlgt_name'], h['soc_code'])
        soc[h['soc_code']] = h['soc_name']
        w.writerow([node_id, f"hlt:{h['hlt_code']}", 'IN_HLT'])
        n += 1
    f.close()
    counts['rels_in_hlt'] = n
    f, w = _writer(out / 'nodes_hierarchy.csv', ['id:ID', 'code', 'name', ':LABEL'])
    for code, (name, _) in sorted(hlt.items()):
        w.writerow([f'hlt:{code}', code, name, 'HLT'])
    for code, (name, _) in sorted(hlgt.items()):
        w.writerow([f'hlgt:{code}', code, name, 'HLGT'])
    for code, name in sorted(soc.items()):
        w.writerow([f'soc:{code}', code, name, 'SOC'])
    f.close()
    counts['nodes_hierarchy'] = len(hlt) + len(hlgt) + len(soc)
    f, w = _writer(out / 'rels_hierarchy.csv', [':START_ID', ':END_ID', ':TYPE'])
    for code, (_, parent) in sorted(hlt.items()):
        w.writerow([f'hlt:{code}', f'hlgt:{parent}', 'IN_HLGT'])
    for code, (_, parent) in sorted(hlgt.items()):
        w.writerow([f'hlgt:{code}', f'soc:{parent}', 'IN_SOC'])
    f.close()
    counts['rels_hierarchy'] = len(hlt) + len(hlgt)

    # Patient statements (Section 3.4 edges)
    common = ['id', 'platform', 'record_id', 'record_key', 'date', 'writer_age_group', 'writer_sex', 'writer_evidence',
              'ingredient_basis', 'text', 'llt_code', 'evidence', 'order_key', 'source']
    for table, target, rel in (('edge_used_for', 'purpose_id', 'USED_FOR'), ('edge_side_effect', 'side_effect_id', 'SIDE_EFFECT')):
        se = rel == 'SIDE_EFFECT'
        header = [':START_ID', ':END_ID', ':TYPE'] + common + (['dose', 'duration', 'intensity', 'impact', 'action_taken:string[]',
                                                               'dose_mg:float', 'duration_bucket'] if se else [])
        f, w = _writer(out / f'rels_{rel.lower()}.csv', header)
        n = 0
        for r in con.execute(f'SELECT * FROM {table} ORDER BY id'):
            row = [r['drug_id'], r[target], rel, r['id'], r['platform'], r['record_id'], f"{r['platform']}:{r['record_id']}",
                   r['date'] or '', r['writer_age_group'] or '', r['writer_sex'] or '', r['writer_evidence'],
                   r['ingredient_basis'] or '', r['text'], r['llt_code'] or '', r['evidence'],
                   hashlib.sha256(r['id'].encode()).hexdigest(),
                   'profile' if all(s.get('field') == 'profile_condition' for s in json.loads(r['evidence']) or [{}]) else 'text']
            if se:
                mg = dose_mg(r['dose'])
                row += [r['dose'] or '', r['duration'] or '', r['intensity'] or '', r['impact'] or '',
                        ';'.join(json.loads(r['action_taken'])), '' if mg is None else mg, duration_bucket(r['duration']) or '']
            w.writerow(row)
            n += 1
        f.close()
        counts[f'rels_{rel.lower()}'] = n
    build = {r['key']: json.loads(r['value']) for r in con.execute('SELECT key, value FROM build')}
    con.close()

    # FAERS reports, strata and reactions. has_clinical_pt (at least one PT outside the six non-clinical MedDRA groups)
    # is precomputed because every RQ2 denominator needs it.
    clinical = set()
    for r in _read_gz(FROZEN / 'faers_report_pts.csv.gz'):
        if r['pt_code'] and not hierarchy.get(r['pt_code'], {}).get('nonclinical_group'):
            clinical.add(r['safetyreportid'])
    f, w = _writer(out / 'nodes_report.csv', ['id:ID', 'safetyreportid', 'receivedate', 'sex', 'age_group',
                                              'hospitalization:boolean', 'in_primary_analysis:boolean',
                                              'has_clinical_pt:boolean', ':LABEL'])
    n = 0
    for r in _read_gz(FROZEN / 'faers_reports.csv.gz'):
        d = r['receivedate']
        w.writerow([f"report:{r['safetyreportid']}", r['safetyreportid'], f'{d[:4]}-{d[4:6]}-{d[6:]}' if len(d) == 8 else '',
                    r['sex'], r['age_group'], str(r['hospitalization'] == 'yes').lower(),
                    str(r['in_primary_analysis'] == 'True').lower(), str(r['safetyreportid'] in clinical).lower(), 'Report'])
        n += 1
    f.close()
    counts['nodes_report'] = n
    strata_header = [':START_ID', ':END_ID', ':TYPE', 'level', 'compounded:boolean', 'in_primary:boolean',
                     'in_any_role:boolean', 'in_broad_identity:boolean', 'actions:string[]', 'actions_any_role:string[]',
                     'window_start']
    fd, wd = _writer(out / 'rels_reports_drug.csv', strata_header)
    fi, wi = _writer(out / 'rels_reports_ingredient.csv', strata_header)
    nd = ni = 0
    for r in _read_gz(FROZEN / 'faers_report_strata.csv.gz'):
        level, ing = r['level'], r['ingredient']
        flags = [str(r[k] == 'True').lower() for k in ('compounded', 'in_primary', 'in_sensitivity_any_role',
                                                         'in_sensitivity_broad_identity')]
        ws = r['window_start']
        tail = flags + [r['actiondrug_primary'], r['actiondrug_any_role'], f'{ws[:4]}-{ws[4:6]}-{ws[6:]}' if len(ws) == 8 else '']
        start_id = f"report:{r['safetyreportid']}"
        if level == 'ingredient':
            wi.writerow([start_id, f'ingredient:{ing}', 'REPORTS_INGREDIENT', level] + tail)
            ni += 1
        else:
            brand = BRAND_BY_LOWER[r['brand']] if level == 'brand' else UNSPECIFIED
            if level == 'brand' and BRAND_INGREDIENT[brand] != ing:
                raise SystemExit(f'FAERS stratum brand {brand} does not match ingredient {ing}')
            wd.writerow([start_id, f'drug:{ing}:{brand}', 'REPORTS_DRUG', level] + tail)
            nd += 1
    fd.close()
    fi.close()
    counts['rels_reports_drug'], counts['rels_reports_ingredient'] = nd, ni
    f, w = _writer(out / 'rels_reaction.csv', [':START_ID', ':END_ID', ':TYPE', 'map_method'])
    n = 0
    for r in _read_gz(FROZEN / 'faers_report_pts.csv.gz'):
        if r['pt_code']:
            w.writerow([f"report:{r['safetyreportid']}", f"side_effect:pt:{r['pt_code']}", 'REACTION', r['map_method']])
            n += 1
    f.close()
    counts['rels_reaction'] = n

    inputs = {os.path.relpath(p, ROOT): sha256(p) for p in
              [Path(kg), FROZEN / 'meddra_pt_hierarchy.csv', FROZEN / 'faers_reports.csv.gz',
               FROZEN / 'faers_report_strata.csv.gz', FROZEN / 'faers_report_pts.csv.gz', APPROVAL_CONFIG]}
    return counts, inputs, build


# ---------------------------------------------------------------------------------------------------- import
def load(kg=KG):
    counts, inputs, build = export(kg)
    files = sorted(EXPORT_DIR.glob('*.csv'))
    nodes = [p for p in files if p.name.startswith('nodes_')]
    rels = [p for p in files if p.name.startswith('rels_')]
    stop()
    cmd = [_installed()[1], 'database', 'import', 'full', 'neo4j', '--overwrite-destination=true',
           '--ignore-empty-strings=true', '--array-delimiter=;', '--multiline-fields=true', '--bad-tolerance=0',
           '--report-file', str(HOME / 'logs/import-report.txt')]
    cmd += [f'--nodes={p}' for p in nodes] + [f'--relationships={p}' for p in rels]
    result = _run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f'neo4j-admin import failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}')
    start()
    driver = connect()
    with driver.session() as s:
        for q in INDEXES:
            s.run(q).consume()
        s.run('CALL db.awaitIndexes(600)').consume()
        db_counts = {
            'nodes': s.run('MATCH (n) RETURN count(n) AS c').single()['c'],
            'relationships': s.run('MATCH ()-[r]->() RETURN count(r) AS c').single()['c'],
            'neo4j_version': s.run('CALL dbms.components() YIELD versions RETURN versions[0] AS v').single()['v'],
        }
    driver.close()
    expected_nodes = sum(v for k, v in counts.items() if k.startswith('nodes_'))
    expected_rels = sum(v for k, v in counts.items() if k.startswith('rels_'))
    if (db_counts['nodes'], db_counts['relationships']) != (expected_nodes, expected_rels):
        raise SystemExit(f'imported counts {db_counts} differ from exported nodes {expected_nodes} / relationships {expected_rels}')
    manifest = {'loader_version': LOADER_VERSION, 'kg_version': KG_VERSION,
                'loaded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                'neo4j_version': db_counts['neo4j_version'], 'uri': URI, 'inputs_sha256': inputs,
                'kg_build': {k: build.get(k) for k in ('version', 'built_utc', 'meddra_version', 'extraction', 'counts')},
                'export_rows': counts, 'export_sha256': {p.name: sha256(p) for p in files},
                'db_counts': {k: db_counts[k] for k in ('nodes', 'relationships')},
                'code_sha256': sha256(Path(__file__))}
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return manifest


def manifest():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['start', 'stop', 'status', 'load'])
    ap.add_argument('--kg', default=str(KG))
    a = ap.parse_args(argv)
    if a.command == 'start':
        start()
        print(status())
    elif a.command == 'stop':
        stop()
        print(status())
    elif a.command == 'status':
        print(status())
    else:
        m = load(a.kg)
        print(json.dumps({k: m[k] for k in ('neo4j_version', 'export_rows', 'db_counts')}, indent=1))


if __name__ == '__main__':
    main()
