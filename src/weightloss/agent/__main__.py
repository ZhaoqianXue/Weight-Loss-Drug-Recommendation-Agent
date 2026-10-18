"""Ask the evidence-grounded agent one question about the knowledge graph (local Neo4j, weightloss.graph.neo4j_db).

  PYTHONPATH=src python3 -m weightloss.agent "question"
  PYTHONPATH=src python3 -m weightloss.agent --tool query_graph --args '{"target": "side_effect", ...}'

Prints the answer, the verification result, the tool calls and the cost; every agent run appends its full trace to
.cache/agent-runs/traces.jsonl (or --trace). --tool runs one graph tool without any model call. The Neo4j instance must be
running (python -m weightloss.graph.neo4j_db start).
"""
from __future__ import annotations

import argparse
import json
import sys

from . import agent


def _print_result(r, trace):
    a = r['answer'] or {}
    print('# Answer\n')
    print(a.get('answer') or '(no accepted answer)')
    print(f"\nanswer_type: {a.get('answer_type')}")
    if a.get('limitations'):
        print('\n## Limitations')
        for x in a['limitations']:
            print(f'- {x}')
    if a.get('quotes'):
        print('\n## Quotes')
        for q in a['quotes']:
            print(f"- {q['edge_id']}: \"{q['text']}\"")
    if a.get('citations'):
        print('\n## Citations')
        for c in a['citations']:
            print(f"- [{c['id']}] {c['call']} row {c['row']} {c['field']} = {c['value']}")
    print(f"\n## Verification: {'VERIFIED' if r['verified'] else 'NOT VERIFIED'} (repair round used: {'yes' if r['repair_used'] else 'no'})")
    for p in r['problems']:
        print(f'- problem: {p}')
    for w in r['warnings']:
        print(f'- warning: {w}')
    if r['stop_reason']:
        print(f"- stopped: {r['stop_reason']}")
    print(f"\n## Tool calls ({r['tool_rounds']} tool rounds, {r['responses']} responses)")
    for c in r['tool_calls']:
        args = c['arguments'] if isinstance(c['arguments'], str) else json.dumps(c['arguments'])
        print(f"- {c['call']} {c['tool']} ({c['seconds']} s) {args}" + (f"  ERROR: {c['error']}" if c['error'] else ''))
    u = r['usage']
    print(f"\n## Cost\n${r['cost_usd']:.4f} (input {u.get('input_tokens', 0)} tokens, of which cached {u.get('cached_tokens', 0)}; "
          f"output {u.get('output_tokens', 0)}, of which reasoning {u.get('reasoning_tokens', 0)})")
    print(f"trace: {trace} (run_id {r['run_id']})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('question', nargs='?')
    ap.add_argument('--model', default=agent.MODEL)
    ap.add_argument('--effort', default=agent.EFFORT)
    ap.add_argument('--max-tool-rounds', type=int, default=agent.MAX_TOOL_ROUNDS)
    ap.add_argument('--budget-usd', type=float, default=agent.BUDGET_USD)
    ap.add_argument('--trace', default=str(agent.TRACE))
    ap.add_argument('--label', help='free-text label stored with the trace')
    ap.add_argument('--json', action='store_true', help='print the result as JSON instead of text')
    ap.add_argument('--tool', choices=sorted(agent.TOOL_ARGS) + ['overview'], help='run one graph tool without the model')
    ap.add_argument('--args', default='{}', help='JSON arguments for --tool')
    a = ap.parse_args(argv)
    if a.tool:
        graph = agent.KnowledgeGraph()
        session = agent.Session(graph)
        out = session.overview() if a.tool == 'overview' else session.run_tool(a.tool, a.args)
        print(json.dumps(agent.public(out), indent=1, ensure_ascii=False))
        graph.close()
        return 0
    if not a.question:
        ap.error('a question is required unless --tool is given')
    r = agent.run(a.question, model=a.model, effort=a.effort, max_tool_rounds=a.max_tool_rounds, budget_usd=a.budget_usd,
                  trace_path=a.trace, label=a.label)
    if a.json:
        print(json.dumps(r, indent=1, ensure_ascii=False))
    else:
        _print_result(r, a.trace)
    return 0 if r['verified'] else 1


if __name__ == '__main__':
    sys.exit(main())
