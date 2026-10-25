"""Command line: run | validate | inspect-meddra | check-knowledge | propose.

  python -m weightloss.standardization.semantic run --mode dev --kg KG --roster ROSTER --out DIR \
      --rules-side-effect R1 --rules-indication R2 --registry REG --catalog configs/drugs.json --run-id ID \
      [--meddra-dir DIR --meddra-encoding ENC [--meddra-version 29.0] [--meddra-pins PINS.json]] [--source SRC]

Exit codes: 0 completed, 3 resource_blocked (no MedDRA dictionary: nothing coded, nothing reported as unmapped),
2 failure. `propose` sends paid requests and needs --confirm-paid plus explicit caps; nothing else calls an API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .runner import EXIT_COMPLETED, EXIT_FAILED, EXIT_RESOURCE_BLOCKED, MODES, run


def _meddra_args(p):
    p.add_argument('--meddra-dir')
    p.add_argument('--meddra-encoding', help='explicit file encoding of the ASCII distribution (recorded)')
    p.add_argument('--meddra-version')
    p.add_argument('--meddra-language', default='English')
    p.add_argument('--meddra-pins', help='JSON {file name: sha256} from the resource freeze record (formal runs)')


def _knowledge_args(p):
    p.add_argument('--rules-side-effect', required=True)
    p.add_argument('--rules-indication', required=True)
    p.add_argument('--registry', required=True)


def _pins(args):
    return json.loads(Path(args.meddra_pins).read_text()) if args.meddra_pins else None


def _meddra_kwargs(args):
    return dict(meddra_dir=args.meddra_dir, meddra_encoding=args.meddra_encoding, meddra_version=args.meddra_version,
                meddra_language=args.meddra_language, meddra_pins=_pins(args))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='weightloss.standardization.semantic')
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--mode', choices=MODES, required=True)
    for name in ('--kg', '--roster', '--out', '--catalog', '--run-id'):
        r.add_argument(name, required=True)
    r.add_argument('--source')
    r.add_argument('--proposals')
    r.add_argument('--expected-kg-sha256')
    _knowledge_args(r)
    _meddra_args(r)
    v = sub.add_parser('validate')
    v.add_argument('--out', required=True)
    v.add_argument('--kg', required=True)
    v.add_argument('--rerun', action='store_true', help='also rebuild into a temporary directory and compare bytes')
    v.add_argument('--source')
    _knowledge_args(v)
    _meddra_args(v)
    m = sub.add_parser('inspect-meddra')
    _meddra_args(m)
    k = sub.add_parser('check-knowledge')
    _knowledge_args(k)
    _meddra_args(k)
    pr = sub.add_parser('propose')
    for name in ('--kg', '--roster', '--out', '--model'):
        pr.add_argument(name, required=True)
    pr.add_argument('--max-requests', type=int, required=True)
    pr.add_argument('--max-input-tokens', type=int, required=True)
    pr.add_argument('--max-output-tokens', type=int, required=True)
    pr.add_argument('--max-usd', type=float, required=True)
    pr.add_argument('--confirm-paid', action='store_true')
    pr.add_argument('--source', help='frozen source snapshot; adds the full record as context')
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            manifest = run(kg_path=args.kg, roster_path=args.roster, out_dir=args.out, rules_side_effect=args.rules_side_effect,
                           rules_indication=args.rules_indication, registry_path=args.registry, catalog_path=args.catalog,
                           mode=args.mode, run_id=args.run_id, source_path=args.source, proposals_path=args.proposals,
                           expected_kg_sha256=args.expected_kg_sha256, **_meddra_kwargs(args))
            print(json.dumps({'status': manifest['status'], 'counts': manifest['counts'], 'out': args.out}, sort_keys=True))
            return EXIT_RESOURCE_BLOCKED if manifest['status'] == 'resource_blocked' else EXIT_COMPLETED
        if args.command == 'validate':
            from .validate import validate
            manifest = json.loads((Path(args.out) / 'run-manifest.json').read_text())
            rerun = None
            if args.rerun:
                inputs = manifest['inputs']

                def rerun(target):
                    run(kg_path=args.kg, roster_path=inputs['roster']['path'], out_dir=target,
                        rules_side_effect=args.rules_side_effect, rules_indication=args.rules_indication,
                        registry_path=args.registry, catalog_path=inputs['catalog']['path'], mode=manifest['mode'],
                        run_id=manifest['run_id'], source_path=(inputs['source_snapshot'] or {}).get('path'),
                        proposals_path=(inputs['proposals'] or {}).get('path'), **_meddra_kwargs(args))
            problems = validate(args.out, kg_path=args.kg, rules_side_effect=args.rules_side_effect,
                                rules_indication=args.rules_indication, registry_path=args.registry, rerun=rerun,
                                source_path=args.source,
                                **_meddra_kwargs(args))
            print(json.dumps({'status': 'pass' if not problems else 'fail', 'problems': problems[:50],
                              'problem_count': len(problems)}, indent=1))
            return EXIT_COMPLETED if not problems else EXIT_FAILED
        if args.command == 'inspect-meddra':
            from .meddra import ResourceBlocked, load_meddra
            try:
                d = load_meddra(args.meddra_dir, encoding=args.meddra_encoding, expected_version=args.meddra_version,
                                expected_language=args.meddra_language, pinned_sha256=_pins(args))
            except ResourceBlocked as error:
                print(json.dumps({'status': 'resource_blocked', 'reason': error.reason, 'detail': error.detail}))
                return EXIT_RESOURCE_BLOCKED
            print(json.dumps(d.snapshot(), indent=1, sort_keys=True))
            return EXIT_COMPLETED
        if args.command == 'check-knowledge':
            from .knowledge import load_registry, load_rules, registry_alignment, rule_problems
            from .meddra import ResourceBlocked, load_meddra
            registry = load_registry(args.registry)
            sets = [load_rules(args.rules_side_effect, 'side_effect', registry),
                    load_rules(args.rules_indication, 'indication', registry)]
            try:
                d = load_meddra(args.meddra_dir, encoding=args.meddra_encoding, expected_version=args.meddra_version,
                                expected_language=args.meddra_language, pinned_sha256=_pins(args))
            except ResourceBlocked:
                d = None
            problems = [p for rs in sets for p in rule_problems(rs, d, registry)]
            alignment = {cid: {k: v for k, v in registry_alignment(c, registry, d).items() if k in ('status', 'relation', 'anchored')}
                         for cid, c in sorted(registry.concepts.items())}
            print(json.dumps({'dictionary': d.snapshot()['snapshot_sha256'] if d else 'resource_blocked',
                              'rules': {rs.role: {'version': rs.version, 'rules': len(rs.rules),
                                                  'certified': sum(r.certified for r in rs.rules.values())} for rs in sets},
                              'registry': {'version': registry.version, 'concepts': len(registry.concepts),
                                           'external_alignment': alignment},
                              'unresolved_targets': problems}, indent=1, sort_keys=True))
            return EXIT_COMPLETED
        if args.command == 'propose':
            if not args.confirm_paid:
                print('propose sends paid requests; pass --confirm-paid with explicit caps', file=sys.stderr)
                return EXIT_FAILED
            from weightloss.extraction.runner import make_client, openai_send
            from .llm import Budget, propose
            from .runner import read_roster
            from .source import load_bindings
            out = Path(args.out)
            if out.exists():
                raise RuntimeError(f'{out} exists')
            budget = Budget(model=args.model, max_requests=args.max_requests, max_input_tokens=args.max_input_tokens,
                            max_output_tokens=args.max_output_tokens, max_usd=args.max_usd)
            from .source import check_evidence, file_sha256, load_source_records
            bindings = load_bindings(args.kg, read_roster(args.roster))
            records = None
            if args.source:
                if any(b.source_sha256 != file_sha256(args.source) for b in bindings):
                    raise RuntimeError('source snapshot differs from the KG build input')
                records = load_source_records(args.source, bindings)
                for b in bindings:
                    check_evidence(b, records[b.record_id])
            rows = propose(bindings, openai_send(make_client()), budget, kg_sha256=file_sha256(args.kg), records=records)
            out.write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows))
            print(json.dumps({**budget.summary(), 'ok': sum(r['status'] == 'ok' for r in rows),
                              'anomalies': sum(r['status'] == 'anomaly' for r in rows),
                              'not_sent': sum(r['status'] == 'not_sent' for r in rows)}, sort_keys=True))
            return EXIT_COMPLETED
    except Exception as error:  # every failure is reported, never turned into an empty result
        print(json.dumps({'status': 'failed', 'error': f'{type(error).__name__}: {error}'}), file=sys.stderr)
        return EXIT_FAILED
    return EXIT_FAILED


if __name__ == '__main__':
    sys.exit(main())
