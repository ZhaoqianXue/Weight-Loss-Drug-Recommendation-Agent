"""Build the merged v9 knowledge graph (project document Section 3.4) into one SQLite file.

  python -m weightloss.graph.build --results <extraction results.jsonl> --mapping <mapping-k10.jsonl> --out <dir>

Inputs: the frozen extraction inputs (sentences with character spans, Reddit community and time, WebMD profile), the
extraction results (one row per record: platform, record_id, ok, output) and the MedDRA mapping keyed by mention_id
(results/reextraction-20261016/scripts/meddra_mentions.py). The output holds MedDRA names and codes, so it belongs under
the git-ignored .cache directory.

Tables: drug, purpose, side_effect (nodes); edge_used_for, edge_side_effect (one row per statement, with the edge
properties of Section 3.4; JSON lists are stored as text); build (provenance, one row per key).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from . import VERSION

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / '.cache/reextraction-20261016/full/inputs.jsonl'
FREEZE = ROOT / 'results/reextraction-20261016/freeze-v7.json'
AGE_GROUPS = ('13-18', '19-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75 or over')
PROFILE_SEX = {'Female': 'female', 'Male': 'male', 'Transgender': 'other'}
BRAND_INGREDIENT = {'Ozempic': 'semaglutide', 'Wegovy': 'semaglutide', 'Rybelsus': 'semaglutide', 'Mounjaro': 'tirzepatide',
                    'Zepbound': 'tirzepatide', 'Victoza': 'liraglutide', 'Saxenda': 'liraglutide', 'Trulicity': 'dulaglutide'}
# Single-ingredient communities (brand and ingredient communities). Cross-drug and type 2 diabetes communities have none.
COMMUNITY_INGREDIENT = {'zepbound': 'tirzepatide', 'mounjaro': 'tirzepatide', 'tirzepatiderx': 'tirzepatide',
                        'ozempic': 'semaglutide', 'wegovy': 'semaglutide', 'wegovyweightloss': 'semaglutide',
                        'rybelsuspill': 'semaglutide', 'semaglutide': 'semaglutide', 'victoza': 'liraglutide',
                        'saxendaforweightloss': 'liraglutide', 'liraglutide': 'liraglutide', 'trulicity': 'dulaglutide'}
UNSPECIFIED = 'unspecified'

SCHEMA = """
CREATE TABLE drug (id TEXT PRIMARY KEY, ingredient TEXT NOT NULL, brand TEXT NOT NULL);
CREATE TABLE purpose (id TEXT PRIMARY KEY, pt TEXT, pt_code TEXT);
CREATE TABLE side_effect (id TEXT PRIMARY KEY, pt TEXT, pt_code TEXT);
CREATE TABLE edge_used_for (
  id TEXT PRIMARY KEY, drug_id TEXT NOT NULL REFERENCES drug(id), purpose_id TEXT NOT NULL REFERENCES purpose(id),
  platform TEXT NOT NULL, record_id TEXT NOT NULL, date TEXT, writer_age_group TEXT, writer_sex TEXT, writer_evidence TEXT NOT NULL,
  ingredient_basis TEXT, text TEXT NOT NULL, llt_code TEXT, evidence TEXT NOT NULL);
CREATE TABLE edge_side_effect (
  id TEXT PRIMARY KEY, drug_id TEXT NOT NULL REFERENCES drug(id), side_effect_id TEXT NOT NULL REFERENCES side_effect(id),
  platform TEXT NOT NULL, record_id TEXT NOT NULL, date TEXT, writer_age_group TEXT, writer_sex TEXT, writer_evidence TEXT NOT NULL,
  ingredient_basis TEXT, text TEXT NOT NULL, llt_code TEXT, evidence TEXT NOT NULL,
  dose TEXT, duration TEXT, intensity TEXT, impact TEXT, action_taken TEXT NOT NULL);
CREATE TABLE build (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE INDEX uf_record ON edge_used_for(platform, record_id);
CREATE INDEX uf_drug ON edge_used_for(drug_id);
CREATE INDEX uf_target ON edge_used_for(purpose_id);
CREATE INDEX se_record ON edge_side_effect(platform, record_id);
CREATE INDEX se_drug ON edge_side_effect(drug_id);
CREATE INDEX se_target ON edge_side_effect(side_effect_id);
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record_date(rec):
    if rec['platform'] == 'reddit':
        return dt.datetime.fromtimestamp(rec['created_utc'], dt.timezone.utc).date().isoformat()
    try:
        return dt.datetime.strptime(rec['date'].split()[0], '%m/%d/%Y').date().isoformat()
    except (ValueError, IndexError, AttributeError):
        return None


def spans(rec, ids):
    by_id = {s['id']: s for s in rec['sentences']}
    return [{'field': by_id[i]['field'], 'start': by_id[i]['start'], 'end': by_id[i]['end'], 'text': by_id[i]['text']} for i in ids]


def profile_span(field, value):
    return {'field': field, 'start': 0, 'end': len(value), 'text': value}


def uses_profile(profile):
    """Patient or blank profiles supply condition, age and sex; caregiver profiles supply none of them."""
    return profile['patient_type'] in ('Patient', '')


def writer(rec, output):
    """(age_group, sex, evidence spans) of the record's writer."""
    if rec['platform'] == 'reddit':
        w = output['writer']
        return w['age_group'], w['sex'], spans(rec, w['evidence'])
    p = rec['profile']
    if not uses_profile(p):
        return None, None, []
    age = p['age'] if p['age'] in AGE_GROUPS else None
    sex = PROFILE_SEX.get(p['gender'])
    ev = ([profile_span('profile_age', p['age'])] if age else []) + ([profile_span('profile_gender', p['gender'])] if sex else [])
    return age, sex, ev


def drug_identity(rec, group):
    """(ingredient, brand, ingredient_basis) for one extracted drug group."""
    brand, ingredient = group['brand'], group['ingredient']
    if rec['platform'] == 'webmd':
        p = rec['profile']
        if brand == p['brand'] and ingredient == p['ingredient']:
            return ingredient, brand, 'profile'
    if ingredient:
        return ingredient, brand or UNSPECIFIED, 'name'
    if rec['platform'] == 'reddit' and rec.get('community') in COMMUNITY_INGREDIENT:
        return COMMUNITY_INGREDIENT[rec['community']], UNSPECIFIED, 'community'
    return UNSPECIFIED, UNSPECIFIED, None


def profile_condition(rec):
    p = rec['profile']
    if not uses_profile(p):
        return None
    c = p['condition'].strip()
    return c if c and c != 'Other' else None


def build(results_path, mapping_path, out_dir, allow_unmapped=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / 'kg.sqlite'
    if db_path.exists():
        raise SystemExit(f'{db_path} exists; nothing written')
    results = [json.loads(l) for l in open(results_path)]
    want = {f"{r['platform']}:{r['record_id']}" for r in results}
    inputs = {}
    for line in open(INPUTS):
        rec = json.loads(line)
        k = f"{rec['platform']}:{rec['record_id']}"
        if k in want:
            inputs[k] = rec
    if len(inputs) != len(want):
        raise SystemExit(f'{len(want) - len(inputs)} result records are not in the frozen inputs')
    mapping = {m['mention_id']: m for m in map(json.loads, open(mapping_path))}

    drugs, purposes, effects = {}, {'purpose:uncoded': (None, None)}, {'side_effect:uncoded': (None, None)}
    used_for, side = [], []
    unmapped, stats = [], collections.Counter()

    def coded(mention_id, kind):
        m = mapping.get(mention_id)
        if m is None:
            unmapped.append(mention_id)
            return f'{kind}:uncoded', None
        if m['status'] != 'coded':
            return f'{kind}:uncoded', None
        node = f"{kind}:pt:{m['pt_code']}"
        (purposes if kind == 'purpose' else effects)[node] = (m['pt_name'], m['pt_code'])
        return node, m['llt_code']

    def drug_node(ingredient, brand):
        if brand != UNSPECIFIED and BRAND_INGREDIENT[brand] != ingredient:
            raise ValueError(f'brand {brand} does not match ingredient {ingredient}')
        node = f'drug:{ingredient}:{brand}'
        drugs[node] = (ingredient, brand)
        return node

    for r in results:
        key = f"{r['platform']}:{r['record_id']}"
        rec = inputs[key]
        stats[f"records_{r['platform']}"] += 1
        if not r.get('ok'):
            stats[f"records_anomaly_{r['platform']}"] += 1
            continue
        out = r['output']
        age, sex, w_ev = writer(rec, out)
        common = {'platform': r['platform'], 'record_id': r['record_id'], 'date': record_date(rec), 'writer_age_group': age,
                  'writer_sex': sex, 'writer_evidence': json.dumps(w_ev, ensure_ascii=False)}
        for g, group in enumerate(out['drugs']):
            ingredient, brand, basis = drug_identity(rec, group)
            d = drug_node(ingredient, brand)
            for i, p in enumerate(group['purposes']):
                mid = f'{key}:d{g}:p{i}'
                node, llt = coded(mid, 'purpose')
                used_for.append({'id': mid, 'drug_id': d, 'purpose_id': node, **common, 'ingredient_basis': basis, 'text': p['purpose'],
                                 'llt_code': llt, 'evidence': json.dumps(spans(rec, p['evidence']), ensure_ascii=False)})
            for i, s in enumerate(group['side_effects']):
                mid = f'{key}:d{g}:s{i}'
                node, llt = coded(mid, 'side_effect')
                side.append({'id': mid, 'drug_id': d, 'side_effect_id': node, **common, 'ingredient_basis': basis, 'text': s['effect'],
                             'llt_code': llt, 'evidence': json.dumps(spans(rec, s['evidence']), ensure_ascii=False),
                             'dose': s['dose'], 'duration': s['duration'], 'intensity': s['intensity'], 'impact': s['impact'],
                             'action_taken': json.dumps(s['action_taken'])})
        if r['platform'] == 'webmd':
            condition = profile_condition(rec)
            if condition:
                p = rec['profile']
                d = drug_node(p['ingredient'], p['brand'])
                mid = f'{key}:profile'
                node, llt = coded(mid, 'purpose')
                used_for.append({'id': mid, 'drug_id': d, 'purpose_id': node, **common, 'ingredient_basis': 'profile', 'text': condition,
                                 'llt_code': llt, 'evidence': json.dumps([profile_span('profile_condition', condition)], ensure_ascii=False)})
    if unmapped and not allow_unmapped:
        raise SystemExit(f'{len(unmapped)} mentions have no MedDRA mapping row (first: {unmapped[0]}); nothing written')

    tmp = out_dir / 'kg.sqlite.tmp'
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    con.executemany('INSERT INTO drug VALUES (?,?,?)', [(k, *v) for k, v in sorted(drugs.items())])
    con.executemany('INSERT INTO purpose VALUES (?,?,?)', [(k, *v) for k, v in sorted(purposes.items())])
    con.executemany('INSERT INTO side_effect VALUES (?,?,?)', [(k, *v) for k, v in sorted(effects.items())])
    uf_cols = ['id', 'drug_id', 'purpose_id', 'platform', 'record_id', 'date', 'writer_age_group', 'writer_sex', 'writer_evidence',
               'ingredient_basis', 'text', 'llt_code', 'evidence']
    se_cols = [c if c != 'purpose_id' else 'side_effect_id' for c in uf_cols] + ['dose', 'duration', 'intensity', 'impact', 'action_taken']
    con.executemany(f"INSERT INTO edge_used_for VALUES ({','.join('?' * len(uf_cols))})", [[e[c] for c in uf_cols] for e in used_for])
    con.executemany(f"INSERT INTO edge_side_effect VALUES ({','.join('?' * len(se_cols))})", [[e[c] for c in se_cols] for e in side])
    freeze = json.loads(FREEZE.read_text())
    manifest = {
        'version': VERSION, 'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'inputs': str(INPUTS.relative_to(ROOT)), 'inputs_sha256': sha256(INPUTS),
        'results': os.path.relpath(results_path, ROOT), 'results_sha256': sha256(results_path),
        'mapping': os.path.relpath(mapping_path, ROOT), 'mapping_sha256': sha256(mapping_path),
        'extraction': {k: freeze[k] for k in ('prompt_version', 'prompt_sha256', 'schema_sha256', 'model', 'reasoning_effort')},
        'meddra_version': '29.1', 'unmapped_mentions': len(unmapped),
        'counts': {**stats, 'drug_nodes': len(drugs), 'purpose_nodes': len(purposes), 'side_effect_nodes': len(effects),
                   'used_for_edges': len(used_for), 'side_effect_edges': len(side),
                   'records_with_edges': len({(e['platform'], e['record_id']) for e in used_for + side})},
        'code_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    con.executemany('INSERT INTO build VALUES (?,?)', [(k, json.dumps(v, ensure_ascii=False)) for k, v in manifest.items()])
    con.commit()
    con.close()
    os.replace(tmp, db_path)
    (out_dir / 'manifest.json').write_text(json.dumps({**manifest, 'kg_sha256': sha256(db_path)}, indent=1))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--allow-unmapped', action='store_true', help='link mentions without a mapping row to the uncoded node')
    a = ap.parse_args(argv)
    print(json.dumps(build(a.results, a.mapping, a.out, a.allow_unmapped)['counts'], indent=1))


if __name__ == '__main__':
    main()
