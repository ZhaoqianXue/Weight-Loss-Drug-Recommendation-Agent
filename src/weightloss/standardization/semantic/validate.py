"""Independent validation of a written run directory. Returns a list of problems (empty = valid).

The validator does not trust the run's own hashes: it recomputes every decision in memory from the inputs it is
given (KG, roster, catalog, source snapshot and proposals recorded in the manifest; rules, registry and dictionary
passed explicitly) and compares decisions.jsonl and mappings.jsonl byte for byte, field by field where they differ.
That covers bindings (KG hash and line, carrier hash, drug node and category, review flags, source support), intents,
coverage, coding, duplicate or missing nodes and the aggregate counts. Independent invariants are checked on top:
accepted components only from certified context-free rules, coded LLTs current and PT-derived in the frozen
dictionary, registry targets reviewed with consistent directions, and no code in a resource_blocked run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .knowledge import load_registry, load_rules, registry_alignment, resolve_target
from .meddra import ResourceBlocked, load_meddra
from .models import NodeDecision
from .runner import compute
from .textnorm import fold_key

MANIFEST_KEYS = ('manifest_version', 'mapper_version', 'context_rules_version', 'run_id', 'mode', 'status', 'code_sha256',
                 'counts', 'outputs')


def _field_diff(stored, expected, prefix=''):
    if isinstance(stored, dict) and isinstance(expected, dict):
        out = []
        for key in sorted(set(stored) | set(expected)):
            out += _field_diff(stored.get(key), expected.get(key), f'{prefix}.{key}' if prefix else key)
        return out
    if isinstance(stored, list) and isinstance(expected, list) and len(stored) == len(expected):
        out = []
        for i, (a, b) in enumerate(zip(stored, expected)):
            out += _field_diff(a, b, f'{prefix}[{i}]')
        return out
    return [] if stored == expected else [prefix or '$']


def _compare_lines(name, stored_text, expected_text, key):
    stored = [json.loads(l) for l in stored_text.splitlines() if l.strip()]
    expected = [json.loads(l) for l in expected_text.splitlines() if l.strip()]
    problems = []
    stored_keys = [key(r) for r in stored]
    if len(stored_keys) != len(set(stored_keys)):
        problems.append(f'{name}: duplicate rows {sorted({k for k in stored_keys if stored_keys.count(k) > 1})[:3]}')
    s_map, e_map = {key(r): r for r in stored}, {key(r): r for r in expected}
    for k in sorted(set(e_map) - set(s_map)):
        problems.append(f'{name}: missing {k}')
    for k in sorted(set(s_map) - set(e_map)):
        problems.append(f'{name}: unexpected {k}')
    for k in sorted(set(s_map) & set(e_map)):
        fields = _field_diff(s_map[k], e_map[k])
        if fields:
            problems.append(f'{name}: {k} differs from a fresh computation in {fields[:8]}')
    if not problems and stored_text != expected_text:
        problems.append(f'{name}: order or serialization differs from a fresh computation')
    return problems


def validate(out_dir, *, kg_path, rules_side_effect, rules_indication, registry_path, meddra_dir=None, meddra_encoding=None,
             meddra_version=None, meddra_language='English', meddra_pins=None, source_path=None, rerun=None):
    out_dir = Path(out_dir)
    problems = []
    manifest = json.loads((out_dir / 'run-manifest.json').read_text())
    stored = {name: (out_dir / name).read_text() for name in ('decisions.jsonl', 'mappings.jsonl')}
    for name, meta in manifest['outputs'].items():
        data = stored[name].encode()
        if hashlib.sha256(data).hexdigest() != meta['sha256'] or stored[name].count('\n') != meta['lines']:
            problems.append(f'{name}: hash or line count differs from the manifest')
    inputs = manifest['inputs']
    recorded_source = inputs.get('source_snapshot')
    if recorded_source is not None and source_path is None:
        problems.append('the run was bound to a source snapshot; the validator needs it (--source)')
        return problems
    try:
        expected_decisions, expected_rows, expected_manifest, _ = compute(
            kg_path=kg_path, roster_path=inputs['roster']['path'], rules_side_effect=rules_side_effect,
            rules_indication=rules_indication, registry_path=registry_path, catalog_path=inputs['catalog']['path'],
            mode=manifest['mode'], run_id=manifest['run_id'], meddra_dir=meddra_dir, meddra_encoding=meddra_encoding,
            meddra_version=meddra_version, meddra_language=meddra_language, meddra_pins=meddra_pins, source_path=source_path,
            proposals_path=(inputs.get('proposals') or {}).get('path'))
    except Exception as error:  # a broken input is a validation failure, never a silent skip
        return problems + [f'recomputation failed: {type(error).__name__}: {error}']
    for section in ('kg', 'roster', 'catalog', 'registry'):
        if inputs[section]['sha256'] != expected_manifest['inputs'][section]['sha256']:
            problems.append(f'{section} sha256 differs from the run manifest')
    for role in ('side_effect', 'indication'):
        if inputs['rules'][role]['sha256'] != expected_manifest['inputs']['rules'][role]['sha256']:
            problems.append(f'{role} rules sha256 differs from the run manifest')
    for section in ('source_snapshot', 'proposals', 'meddra', 'rule_target_problems'):
        a, b = inputs.get(section), expected_manifest['inputs'].get(section)
        if section in ('source_snapshot', 'proposals'):
            a, b = (a or {}).get('sha256'), (b or {}).get('sha256')
        if a != b:
            problems.append(f'{section} differs from the run manifest')
    if inputs['registry'].get('external_alignment') != expected_manifest['inputs']['registry']['external_alignment']:
        problems.append('registry external alignment differs from the run manifest')
    for key in MANIFEST_KEYS:
        if manifest.get(key) != expected_manifest.get(key):
            problems.append(f'manifest {key} differs from a fresh computation')
    problems += _compare_lines('decisions.jsonl', stored['decisions.jsonl'], expected_decisions, lambda r: r.get('node_id'))
    problems += _compare_lines('mappings.jsonl', stored['mappings.jsonl'], expected_rows,
                               lambda r: (r.get('node_id'), r.get('component_id')))
    problems += invariants(stored['decisions.jsonl'], manifest, rules_side_effect, rules_indication, registry_path,
                           meddra_dir, meddra_encoding, meddra_version, meddra_language, meddra_pins)
    if rerun is not None and not problems:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            again = Path(tmp) / 'rerun'
            rerun(again)
            for name in ('decisions.jsonl', 'mappings.jsonl', 'run-manifest.json'):
                if (again / name).read_bytes() != (out_dir / name).read_bytes():
                    problems.append(f'{name}: a fresh written run is not byte-identical')
    return problems


def invariants(decision_text, manifest, rules_side_effect, rules_indication, registry_path, meddra_dir, meddra_encoding,
               meddra_version, meddra_language, meddra_pins):
    """Rules that hold for any valid output, checked on the stored records themselves."""
    problems = []
    registry = load_registry(registry_path)
    rules = {'side_effect': load_rules(rules_side_effect, 'side_effect', registry),
             'indication': load_rules(rules_indication, 'indication', registry)}
    try:
        dictionary = load_meddra(meddra_dir, encoding=meddra_encoding, expected_version=meddra_version,
                                 expected_language=meddra_language, pinned_sha256=meddra_pins)
    except ResourceBlocked:
        dictionary = None
    if manifest['mode'] == 'formal' and dictionary is not None and dictionary.dictionary_class != 'formal':
        problems.append('formal run without a pinned formal dictionary')
    for i, line in enumerate(decision_text.splitlines(), 1):
        try:
            d = NodeDecision.model_validate_json(line)
        except ValueError as error:
            problems.append(f'decisions.jsonl:{i}: {str(error).splitlines()[0]}')
            continue
        if manifest['status'] == 'resource_blocked' and (d.dictionary_class is not None or d.coding == 'coded'):
            problems.append(f'{d.node_id}: a resource_blocked run carries dictionary output')
        for c in d.components:
            if c.semantic_status != 'accepted':
                continue
            tag = f'{d.node_id} {c.component_id}'
            rule = rules[d.role].rules.get(c.rule_id)
            if rule is None or not rule.certified or rule.context_required:
                problems.append(f'{tag}: accepted without a certified context-free rule')
                continue
            if fold_key(c.core_text) not in rule.keys:
                problems.append(f'{tag}: rule {rule.rule_id} does not match the component text')
            status, payload = resolve_target(rule.targets[0], dictionary, registry)
            t = c.target
            if t.kind == 'registry':
                if not (payload is not None and payload.reviewed and t.concept_id == rule.targets[0].concept_id):
                    problems.append(f'{tag}: registry target missing, unreviewed or not the rule target')
                elif t.external_alignment is None or \
                        t.external_alignment.model_dump(mode='json') != registry_alignment(payload, registry, dictionary):
                    problems.append(f'{tag}: registry external alignment is not the one this registry and dictionary give')
                elif c.intent == 'goal' and payload.inherent_direction and c.goal_direction != payload.inherent_direction:
                    problems.append(f'{tag}: goal direction contradicts the concept direction')
                elif c.intent not in (None, 'unclear') and c.intent not in payload.allowed_intents:
                    problems.append(f'{tag}: intent {c.intent} not allowed for {payload.concept_id}')
            elif c.coding_status == 'coded':
                llt = dictionary.llts.get(t.llt_code) if dictionary is not None else None
                if llt is None or llt.name != t.llt_name or not llt.current or llt.name != rule.targets[0].llt_name:
                    problems.append(f'{tag}: coded LLT not current / not the rule target in the frozen dictionary')
                else:
                    pt = dictionary.pt_of(llt)
                    if (pt.code, pt.name) != (t.pt_code, t.pt_name):
                        problems.append(f'{tag}: PT is not the one derived from the frozen dictionary')
            elif c.coding_status == 'resource_blocked' and (dictionary is not None or t.llt_code is not None):
                problems.append(f'{tag}: resource_blocked component with a dictionary present or a code')
            if rule.inherent_direction and c.intent == 'goal' and c.goal_direction != rule.inherent_direction:
                problems.append(f'{tag}: goal direction contradicts the rule concept direction')
    return problems
