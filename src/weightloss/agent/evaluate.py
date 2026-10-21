"""Run the frozen evaluation questions through the agent (project document Section 3.7).

  PYTHONPATH=src python3 -m weightloss.agent.evaluate --out .cache/agent-runs/eval-v1-20261017 [--workers N]

Questions run in parallel (one thread per question by default; each has its own Responses API conversation and graph
sessions). Writes, under --out: one markdown answer per question (Qnn.md), results.jsonl (one line per question:
answer_type, whether it is in the question's accepted set, verification, repair, tool calls, cost), summary.json (counts
overall and per area) and the full traces (traces.jsonl). The must_not checks need a reader and are not scored here.
A folder that already holds results is resumed: finished questions are skipped (a question stopped by a transport error is
run again and its new line supersedes the old one), and a frozen run is never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import io
import json
import threading
from pathlib import Path

from . import VERSION, agent, code_baseline
from .__main__ import _print_result
from .tools import KnowledgeGraph

QUESTIONS = Path(__file__).with_name('eval_questions_dev.json')
WRITE_LOCK = threading.Lock()


def _counts(rows):
    return {'questions': len(rows), 'verified': sum(bool(r['verified']) for r in rows),
            'type_accepted': sum(r['type_accepted'] for r in rows), 'query_ok': sum(bool(r['query_ok']) for r in rows),
            'repair_used': sum(r['repair_used'] for r in rows), 'cost_usd': round(sum(r['cost_usd'] for r in rows), 4)}


def _norm_arg(key, value):
    if key in ('targets', 'require_purposes', 'require_side_effects') and isinstance(value, list):
        return [f"{c['level']}:{c['code']}" for c in value]
    return value


def _matches(spec, call):
    """A tool call satisfies a requires/forbids spec: same tool; each spec value present in the call arguments (lists as
    subsets, concepts as level:code); target_levels = some target concept at one of these levels."""
    if call['tool'] != spec['tool'] or call['error']:
        return False
    args = call['arguments']
    args = json.loads(args) if isinstance(args, str) else args
    for key, want in spec.get('args', {}).items():
        if key == 'target_levels':
            if not any(c['level'] in want for c in args.get('targets') or []):
                return False
            continue
        got = _norm_arg(key, args.get(key))
        if isinstance(want, list):
            if not isinstance(got, list) or not set(want) <= set(got):
                return False
        elif got != want:
            return False
    return True


def check_calls(q, calls):
    """Query problems: requires specs no call satisfies, and forbids specs some call matches."""
    problems = [f"missing call {json.dumps(s)}" for s in q.get('requires', []) if not any(_matches(s, c) for c in calls)]
    problems += [f"forbidden call {json.dumps(s)}" for s in q.get('forbids', []) if any(_matches(s, c) for c in calls)]
    return problems


def _run_one(q, graph, out, version, system='agent', effort=agent.EFFORT, model=agent.MODEL, budget_usd=agent.BUDGET_USD):
    if system in ('code', 'code-naive'):
        r = code_baseline.run(q['question'], trace_path=out / 'traces.jsonl', label=f"{version}:{q['id']}",
                              variant='naive' if system == 'code-naive' else 'rules', effort=effort)
    else:
        r = agent.run(q['question'], graph, trace_path=out / 'traces.jsonl', label=f"{version}:{q['id']}",
                      critic=system == 'agent-critic', effort=effort, model=model, budget_usd=budget_usd)
    a = r['answer'] or {}
    row = {'id': q['id'], 'area': q['area'], 'expected': q['expected'], 'accept': q['accept'], 'system': system,
           'answer_type': a.get('answer_type'), 'type_accepted': a.get('answer_type') in q['accept'],
           'query_problems': [] if not system.startswith('agent') else check_calls(q, r['tool_calls'][1:]) + (
               ['answer cites no numbers'] if q.get('needs_numbers') and not a.get('citations') else []) + (
               ['answer has no verified quote'] if q.get('needs_quotes') and not a.get('quotes') else []),
           'verified': r['verified'], 'repair_used': r['repair_used'], 'stop_reason': r['stop_reason'],
           'problems': r['problems'], 'tools': [c['tool'] for c in r['tool_calls'][1:]],
           'tool_errors': [c['error'] for c in r['tool_calls'] if c['error']], 'cost_usd': r['cost_usd'],
           'dev_seen': bool(q.get('dev_seen')), 'dev_template': bool(q.get('dev_template')), 'run_id': r['run_id'],
           'critique': (r.get('critique') or {}).get('verdict')}
    row['query_ok'] = None if not system.startswith('agent') else not row['query_problems']
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"{q['id']} ({q['area']}; expected {q['expected']}, accept {q['accept']})\nQuestion: {q['question']}\n"
              f"Check: {q['check']}\nMust not: {q.get('must_not', [])}\n")
        if not system.startswith('agent'):
            print(f"Answer ({(r['answer'] or {}).get('answer_type')}):\n{(r['answer'] or {}).get('answer')}\n\nLimitations: "
                  f"{(r['answer'] or {}).get('limitations')}\nTool calls: {len(r['tool_calls'])}; cost ${r['cost_usd']}; stop {r['stop_reason']}")
        else:
            _print_result(r, out / 'traces.jsonl')
    with WRITE_LOCK:
        (out / f"{q['id']}.md").write_text(buf.getvalue())
        with open(out / 'results.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--questions', default=str(QUESTIONS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*', help='question ids to run (default: all)')
    ap.add_argument('--workers', type=int, default=0, help='parallel questions (default: one thread per question)')
    ap.add_argument('--system', choices=['agent', 'agent-critic', 'code', 'code-naive'], default='agent',
                    help='agent (graph tools, verifier), code (code-agent baseline with the agent\'s rules) or code-naive '
                         '(code-agent baseline with the table description only); see weightloss.agent.code_baseline')
    ap.add_argument('--effort', choices=['low', 'medium', 'high'], default=agent.EFFORT,
                    help=f'reasoning effort of the model for every system (default {agent.EFFORT})')
    ap.add_argument('--model', choices=['gpt-6-luna', 'gpt-6-sol'], default=agent.MODEL,
                    help=f'model of the agent systems (default {agent.MODEL}); the code baselines always use {agent.MODEL}')
    ap.add_argument('--budget-usd', type=float, default=agent.BUDGET_USD,
                    help=f'spending limit per question for the agent systems (default {agent.BUDGET_USD})')
    a = ap.parse_args(argv)
    if a.model != agent.MODEL and not a.system.startswith('agent'):
        raise SystemExit('--model applies to the agent systems only')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.questions).read_bytes()
    spec = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    done = {}
    if (out / 'results.jsonl').exists():
        prev = json.loads((out / 'manifest.json').read_text())
        if prev['questions_sha256'] != sha:
            raise SystemExit(f'{out} holds results for another question file; choose a new folder')
        done = {r['id']: r for r in map(json.loads, open(out / 'results.jsonl'))}
        done = {k: r for k, r in done.items() if not (r.get('stop_reason') or '').startswith('transport_error')}  # rerun these
    questions = [q for q in spec['questions'] if (not a.only or q['id'] in a.only) and q['id'] not in done]
    graph = KnowledgeGraph() if a.system.startswith('agent') else None
    if not done:
        manifest = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), 'agent_version': VERSION,
                    'questions_version': spec['version'], 'questions_sha256': sha,
                    'system': a.system if a.system.startswith('agent') else f"{code_baseline.VERSION}:{'naive' if a.system == 'code-naive' else 'rules'}",
                    'model': a.model, 'reasoning_effort': a.effort, 'budget_usd_per_question': a.budget_usd,
                    'instructions_sha256': agent.sha256_text(agent.INSTRUCTIONS if a.system.startswith('agent') else
                                                             code_baseline.INSTRUCTIONS['naive' if a.system == 'code-naive' else 'rules']),
                    'tools_sha256': agent.sha256_text(agent.compact_json(agent.TOOLS if a.system.startswith('agent') else code_baseline.TOOLS)),
                    'graph_loaded_utc': graph.manifest.get('loaded_utc') if graph else None, 'n_questions': len(spec['questions'])}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    workers = a.workers or max(1, len(questions))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, q, graph, out, spec['version'], a.system, a.effort, a.model, a.budget_usd)
                   for q in questions]
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
            except Exception as error:  # a crash in one question must not stop the run; it is rerun on resume
                print(json.dumps({'error': f'{type(error).__name__}: {error}'[:300]}), flush=True)
                continue
            done[row['id']] = row
            print(json.dumps({k: row[k] for k in ('id', 'answer_type', 'type_accepted', 'query_ok', 'verified', 'repair_used', 'cost_usd')}),
                  flush=True)
    if graph:
        graph.close()
    results = [done[q['id']] for q in spec['questions'] if q['id'] in done]
    by_area = collections.defaultdict(list)
    for r in results:
        by_area[r['area']].append(r)
    summary = {'all': _counts(results),
               'unseen_in_development': _counts([r for r in results if not r['dev_seen'] and not r['dev_template']]),
               'by_area': {k: _counts(v) for k, v in by_area.items()}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['all']))


if __name__ == '__main__':
    main()
