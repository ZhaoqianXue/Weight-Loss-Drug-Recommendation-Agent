"""Reproducible mapping run: bind the requested KG nodes, decide each one, write a new output directory atomically.

Modes: `formal` needs a pinned dictionary (dictionary_class formal) or reports resource_blocked; `dev` accepts an
unpinned dictionary or none; `test` only accepts a synthetic test dictionary and marks every output as such. The KG
file hash is taken before and after the run and must not change. Payload files carry no clock time (audit.json
holds it), so two runs on the same inputs are byte-identical.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

from . import VERSION, context as ctx
from .decide import Decider
from .knowledge import load_registry, load_rules, registry_alignment, rule_problems
from .meddra import ResourceBlocked, load_meddra
from .source import check_evidence, file_sha256, load_bindings, load_catalog, load_source_records

MODES = ('formal', 'dev', 'test')
EXIT_COMPLETED, EXIT_FAILED, EXIT_RESOURCE_BLOCKED = 0, 2, 3


class RunError(RuntimeError):
    pass


def code_pins():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def read_roster(path):
    """Node ids in file order from a JSONL roster (rows with node.id or node_id)."""
    ids = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        node_id = (row.get('node') or {}).get('id') or row.get('node_id')
        if not node_id:
            raise RunError('roster row without node.id / node_id')
        ids.append(node_id)
    if len(ids) != len(set(ids)):
        raise RunError('roster repeats a node id')
    return ids


def read_proposals(path, bindings, kg_sha256, records=None):
    """Proposal rows bound to this KG and these carrier edges. Unknown nodes, duplicates and stale bindings fail;
    anomalies and unsent rows are counted, never used."""
    if path is None:
        return {}, {}
    from .llm import PROMPT_VERSION, context_sha256
    by_node = {b.node['id']: b for b in bindings}
    out, seen, counts = {}, set(), {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        node_id = row['node_id']
        b = by_node.get(node_id)
        if b is None:
            raise RunError(f'proposal for a node outside the roster: {node_id}')
        if node_id in seen:
            raise RunError(f'duplicate proposal for {node_id}')
        seen.add(node_id)
        if (row['kg_sha256'], row['carrier_sha256'], row['record_id'], row['item_path'], row['prompt_version']) != \
                (kg_sha256, b.carrier_sha256, b.record_id, b.node['item_path'], PROMPT_VERSION):
            raise RunError(f'stale proposal binding for {node_id}')
        record_text = None
        if row.get('record_context'):
            if not records or b.record_id not in records or row.get('source_sha256') != b.source_sha256:
                raise RunError(f'{node_id}: the proposal used the full record; this run has no matching source snapshot')
            record_text = '\n'.join(v for v in records[b.record_id].values() if isinstance(v, str))
        if row.get('context_sha256') != context_sha256(b, record_text):
            raise RunError(f'{node_id}: proposal context digest differs from the context rebuilt for this run')
        counts[row['status']] = counts.get(row['status'], 0) + 1
        if row['status'] == 'ok':
            out[node_id] = {'terms': [c['normalized_term'] for c in row['proposal']['components'] if c.get('role_fit') != 'not_target']}
    return out, counts


def _jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)


def mapping_rows(decision, mode):
    d = decision
    for c in d.components:
        t = c.target
        ext = t.external_alignment if t is not None else None
        yield {'run_id': d.run_id, 'mode': mode, 'dictionary_class': d.dictionary_class, 'record_id': d.record_id,
               'node_id': d.node_id, 'role': d.role, 'item_path': d.item_path, 'surface': d.surface,
               'component_id': c.component_id, 'component_text': c.text, 'semantic_status': c.semantic_status,
               'coding_status': c.coding_status, 'target_kind': t.kind if t else None, 'llt_code': t.llt_code if t else None,
               'llt_name': t.llt_name if t else None, 'pt_code': t.pt_code if t else None, 'pt_name': t.pt_name if t else None,
               'concept_id': t.concept_id if t else None, 'intent': c.intent, 'goal_direction': c.goal_direction,
               'external_alignment_status': ext.status if ext else None,
               'external_alignment_relation': ext.relation if ext else None,
               'external_anchored': ext.anchored if ext else None,
               'method': c.method, 'rule_id': c.rule_id, 'specificity': c.specificity, 'coverage': d.coverage,
               'reasons': list(c.reasons)}


def compute(*, kg_path, roster_path, rules_side_effect, rules_indication, registry_path, catalog_path, mode, run_id,
            meddra_dir=None, meddra_encoding=None, meddra_version=None, meddra_language='English', meddra_pins=None,
            source_path=None, proposals_path=None, expected_kg_sha256=None):
    """Everything a run decides, without writing: (decision text, mapping-row text, manifest, decisions)."""
    if mode not in MODES:
        raise RunError(f'mode must be one of {MODES}')
    kg_before = file_sha256(kg_path)
    if expected_kg_sha256 and kg_before != expected_kg_sha256:
        raise RunError('KG sha256 differs from the expected formal KG')
    registry = load_registry(registry_path)
    rules = {'side_effect': load_rules(rules_side_effect, 'side_effect', registry),
             'indication': load_rules(rules_indication, 'indication', registry)}
    catalog = load_catalog(catalog_path)
    blocked = None
    try:
        dictionary = load_meddra(meddra_dir, encoding=meddra_encoding, expected_version=meddra_version,
                                 expected_language=meddra_language, pinned_sha256=meddra_pins)
    except ResourceBlocked as error:
        dictionary, blocked = None, {'reason': error.reason, 'detail': error.detail}
    if dictionary is not None:
        allowed = {'formal': {'formal'}, 'dev': {'unpinned', 'formal'}, 'test': {'synthetic_test'}}[mode]
        if dictionary.dictionary_class not in allowed:
            raise RunError(f'{mode} mode cannot use a {dictionary.dictionary_class} dictionary')
    node_ids = read_roster(roster_path)
    bindings = load_bindings(kg_path, node_ids)
    records = {}
    if mode == 'formal' and source_path is None:
        raise RunError('a formal run binds every node to the frozen source snapshot (--source)')
    if source_path is not None:
        source_sha = file_sha256(source_path)
        if any(b.source_sha256 != source_sha for b in bindings):
            raise RunError('source snapshot sha256 differs from the one the KG packages were built from')
        records = load_source_records(source_path, bindings)
        for b in bindings:
            check_evidence(b, records[b.record_id])
    proposals, proposal_counts = read_proposals(proposals_path, bindings, kg_before, records)
    decider = Decider(run_id=run_id, kg_sha256=kg_before, rules=rules, dictionary=dictionary, registry=registry,
                      catalog=catalog)
    decisions = [decider.decide(b, records.get(b.record_id), proposals.get(b.node['id'])) for b in bindings]
    decisions.sort(key=lambda d: (d.role, d.record_id, d.node_id))
    if file_sha256(kg_path) != kg_before:
        raise RunError('the KG file changed during the run')
    status = 'resource_blocked' if dictionary is None else 'completed'
    decision_text = _jsonl(d.model_dump(mode='json') for d in decisions)
    rows_text = _jsonl(row for d in decisions for row in mapping_rows(d, mode))
    external = {}
    for concept in registry.concepts.values():
        align = registry_alignment(concept, registry, dictionary)['status']
        external[align] = external.get(align, 0) + 1
    manifest = {
        'manifest_version': 'semantic-run-0.1.2', 'mapper_version': VERSION, 'context_rules_version': ctx.RULES_VERSION,
        'run_id': run_id, 'mode': mode, 'status': status, 'code_sha256': code_pins(),
        'inputs': {
            'kg': {'path': str(kg_path), 'sha256': kg_before, 'unchanged_after_run': True},
            'roster': {'path': str(roster_path), 'sha256': file_sha256(roster_path), 'nodes': len(node_ids)},
            'rules': {role: {'path': str(p), 'sha256': rs.sha256, 'version': rs.version, 'rules': len(rs.rules),
                             'certified': sum(r.certified for r in rs.rules.values())}
                      for role, p, rs in (('side_effect', rules_side_effect, rules['side_effect']),
                                          ('indication', rules_indication, rules['indication']))},
            'rule_target_problems': sorted((p for rs in rules.values() for p in rule_problems(rs, dictionary, registry)),
                                           key=lambda p: (p['rule_id'], p['target'], p['status'])),
            'registry': {'path': str(registry_path), 'sha256': registry.sha256, 'version': registry.version,
                         'concepts': len(registry.concepts), 'reviewed': sum(c.reviewed for c in registry.concepts.values()),
                         # a completed run does not anchor registry concepts in MedDRA; only a recorded check does
                         'external_alignment': dict(sorted(external.items()))},
            'catalog': {'path': str(catalog_path), 'sha256': catalog.sha256},
            'source_snapshot': {'path': str(source_path), 'sha256': file_sha256(source_path)} if source_path else None,
            'proposals': {'path': str(proposals_path), 'sha256': file_sha256(proposals_path), 'rows': proposal_counts}
            if proposals_path else None,
            'meddra': dictionary.snapshot() if dictionary is not None else {'status': 'resource_blocked', **blocked},
        },
        'counts': aggregate(decisions),
        'outputs': {'decisions.jsonl': {'sha256': hashlib.sha256(decision_text.encode()).hexdigest(), 'lines': len(decisions)},
                    'mappings.jsonl': {'sha256': hashlib.sha256(rows_text.encode()).hexdigest(),
                                       'lines': rows_text.count('\n')}},
    }
    return decision_text, rows_text, manifest, decisions


def aggregate(decisions):
    counts = {}
    for d in decisions:
        role = counts.setdefault(d.role, {})
        for key, value in (('coverage', d.coverage), ('coding', d.coding), ('intent_coverage', d.intent_coverage),
                           ('drug_category', d.drug_category)):
            role.setdefault(key, {}).setdefault(value, 0)
            role[key][value] += 1
        for c in d.components:
            pairs = [('semantic_status', c.semantic_status), ('coding_status', c.coding_status)]
            if c.semantic_status == 'accepted' and c.target.kind == 'registry':
                pairs.append(('registry_external_alignment', c.target.external_alignment.status))
            for key, value in pairs:
                role.setdefault(key, {}).setdefault(value, 0)
                role[key][value] += 1
            for reason in c.reasons:
                role.setdefault('reasons', {}).setdefault(reason, 0)
                role['reasons'][reason] += 1
    return counts


def manifest_text(manifest):
    return json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False) + '\n'


def run(*, out_dir, **inputs):
    """compute() and write a new directory atomically (decisions, mapping rows, manifest; clock time only in audit)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise RunError(f'{out_dir} exists; every run writes a new directory')
    decision_text, rows_text, manifest, _ = compute(**inputs)
    audit = {'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'pid': os.getpid()}
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out_dir.name + '.', dir=out_dir.parent))
    (tmp / 'decisions.jsonl').write_text(decision_text)
    (tmp / 'mappings.jsonl').write_text(rows_text)
    (tmp / 'run-manifest.json').write_text(manifest_text(manifest))
    (tmp / 'audit.json').write_text(json.dumps(audit, sort_keys=True, indent=1) + '\n')
    os.rename(tmp, out_dir)
    return manifest
