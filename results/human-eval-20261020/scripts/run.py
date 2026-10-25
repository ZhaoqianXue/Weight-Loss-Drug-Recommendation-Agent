"""Run baseline B1 or B2 (baselines.py) on a question file, in the layout of weightloss.agent.evaluate.

Writes <out>/manifest.json, results.jsonl (one row per question), <id>.md (readable answer) and traces.jsonl (full traces).
A rerun with the same folder resumes: finished questions are kept, transport errors are run again.

    .venv/bin/python results/human-eval-20261020/scripts/run.py --system b2 --questions <file> --out <dir> [--only W01 W02]
"""
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines  # noqa: E402

LOCK = threading.Lock()


def one(q, system, out, version):
    r = baselines.run(q['question'], system, trace_path=out / 'traces.jsonl', label=f"{version}:{q['id']}")
    a = r['answer'] or {}
    row = {'id': q['id'], 'area': q['area'], 'expected': q['expected'], 'accept': q['accept'], 'system': r['system'],
           'answer_type': a.get('answer_type'), 'type_accepted': a.get('answer_type') in q['accept'],
           'stop_reason': r['stop_reason'], 'problems': r['problems'], 'tool_rounds': r['tool_rounds'],
           'tool_errors': [c['error'] for c in r['tool_calls'] if c['error']], 'cost_usd': r['cost_usd'], 'run_id': r['run_id']}
    lines = [f"{q['id']} ({q['area']}; expected {q['expected']}, accept {q['accept']})", f"Question: {q['question']}", '',
             f"Answer ({a.get('answer_type')}):", a.get('answer') or '', '', f"Limitations: {a.get('limitations')}",
             f"Citations: {a.get('citations')}", f"Quotes: {a.get('quotes')}", '',
             f"Searches ({len(r['tool_calls'])}): " + '; '.join(json.dumps(c['arguments']) for c in r['tool_calls']),
             f"Cost ${r['cost_usd']}; stop {r['stop_reason']}"]
    with LOCK:
        (out / f"{q['id']}.md").write_text('\n'.join(lines) + '\n')
        with open(out / 'results.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--system', choices=['b1', 'b2'], required=True)
    ap.add_argument('--questions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.questions).read_bytes()
    spec, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
    manifest = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                'system': f'{baselines.VERSION}:{a.system}', 'model': baselines.MODEL, 'reasoning_effort': baselines.EFFORT,
                'questions_version': spec['version'], 'questions_sha256': sha, 'n_questions': len(spec['questions']),
                'instructions_sha256': baselines.sha256_text(baselines.INSTRUCTIONS[a.system]),
                'tools_sha256': baselines.sha256_text(baselines.compact_json(baselines.TOOLS[a.system])),
                'schema_sha256': baselines.sha256_text(baselines.compact_json(baselines.SCHEMAS[a.system][1])),
                'index_manifest': json.loads(Path(str(baselines.INDEX) + '.manifest.json').read_text()) if a.system == 'b2' else None}
    done = {}
    if (out / 'results.jsonl').exists():
        prev = json.loads((out / 'manifest.json').read_text())
        for k in ('system', 'questions_sha256', 'instructions_sha256', 'tools_sha256', 'schema_sha256'):
            if prev[k] != manifest[k]:
                raise SystemExit(f'{out} holds results of another configuration ({k}); choose a new folder')
        done = {r['id']: r for r in map(json.loads, open(out / 'results.jsonl'))
                if not (r.get('stop_reason') or '').startswith('transport_error')}
    else:
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    todo = [q for q in spec['questions'] if (not a.only or q['id'] in a.only) and q['id'] not in done]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(one, q, a.system, out, spec['version']) for q in todo]):
            try:
                row = fut.result()
            except Exception as error:
                print(json.dumps({'error': f'{type(error).__name__}: {error}'[:300]}), flush=True)
                continue
            done[row['id']] = row
            print(json.dumps({k: row[k] for k in ('id', 'answer_type', 'type_accepted', 'tool_rounds', 'cost_usd', 'stop_reason')}),
                  flush=True)
    rows = [done[q['id']] for q in spec['questions'] if q['id'] in done]
    summary = {'n': len(rows), 'type_accepted': sum(r['type_accepted'] for r in rows),
               'failed': sum(1 for r in rows if r['problems']), 'cost_usd': round(sum(r['cost_usd'] for r in rows), 4)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
