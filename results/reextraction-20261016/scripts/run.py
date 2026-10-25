"""Synchronous development run of the v9 extraction on a frozen sample (bounded, pre-authorized test spend).

Usage: python run.py --set dev500 --run-id dev-r1 [--limit N] [--workers 16] [--cap-usd 3]
Writes .cache/reextraction-20261016/runs/<run-id>/{responses.jsonl, manifest.json} and prints a summary.
One request per record; one bounded retry for a retriable anomaly or a failed local check.
"""
import argparse, concurrent.futures as cf, json, os, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from weightloss.extraction import reextract as rx
from weightloss.extraction.structured import (ExtractionAnomaly, build_body, cost_usd, final_output_text, parse_json_object,
                                              schema_sha256, sha256_text, strict_json_schema, token_split, validate_model)

MODEL, EFFORT, MAX_OUT = 'gpt-6-luna', 'medium', 16000
SAMPLES = os.path.join(ROOT, '.cache/reextraction-20261016/samples')
RUNS = os.path.join(ROOT, '.cache/reextraction-20261016/runs')


def load_key():
    if not os.environ.get('OPENAI_API_KEY'):
        for line in open(os.path.join(ROOT, '.env')):
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('"').strip("'")


def model_input(rec):
    if rec['platform'] == 'reddit':
        return rx.render('reddit', rec['kind'], rec['sentences'], community=rec['community'])
    p = rec['profile']
    return rx.render('webmd', 'review', rec['sentences'], reviewed=(p['brand'], p['ingredient']))


def body_for(rec):
    model = rx.output_model(rec['platform'])
    return build_body(model=MODEL, instructions=rx.PROMPTS[rec['platform']], user_content=model_input(rec),
                      schema_name=f"v9_{rec['platform']}", schema=strict_json_schema(model), max_output_tokens=MAX_OUT,
                      reasoning_effort=EFFORT, store=False)


def attempt(client, rec):
    t0 = time.time()
    try:
        raw = client.responses.create(**body_for(rec)).model_dump()
    except Exception as e:  # transport or HTTP error: no usage known
        return {'ok': False, 'anomaly': {'kind': 'transport_error', 'detail': type(e).__name__}, 'usage': None, 'seconds': time.time() - t0}
    usage = raw.get('usage')
    try:
        parsed = validate_model(rx.output_model(rec['platform']), parse_json_object(final_output_text(raw)))
        problems = rx.check_output(parsed, rec['sentences'])
        if problems:
            return {'ok': False, 'anomaly': {'kind': 'check_failed', 'detail': problems[:5]}, 'usage': usage, 'problems': problems,
                    'parsed': parsed, 'output': parsed.model_dump(), 'seconds': time.time() - t0, 'effort': (raw.get('reasoning') or {}).get('effort')}
        return {'ok': True, 'output': parsed.model_dump(), 'usage': usage, 'seconds': time.time() - t0,
                'effort': (raw.get('reasoning') or {}).get('effort')}
    except ExtractionAnomaly as e:
        return {'ok': False, 'anomaly': e.as_dict(), 'usage': usage, 'seconds': time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--cap-usd', type=float, default=3.0)
    ap.add_argument('--prompt', help='archived prompt version in results/reextraction-20261016/prompts (default: the module prompt)')
    a = ap.parse_args()
    if a.prompt:
        pdir = os.path.join(ROOT, 'results/reextraction-20261016/prompts')
        rx.PROMPTS = {p: open(os.path.join(pdir, f'{a.prompt}-{p}.txt')).read() for p in ('reddit', 'webmd')}
        rx.PROMPT_VERSION = a.prompt
    recs = [json.loads(l) for l in open(os.path.join(SAMPLES, a.set + '.jsonl'))][:a.limit]
    out_dir = os.path.join(RUNS, a.run_id)
    os.makedirs(out_dir, exist_ok=False)
    load_key()
    from openai import OpenAI
    client = OpenAI(max_retries=2, timeout=300)
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()

    def work(rec):
        tries = []
        for _ in range(2):
            if stop.is_set():
                break
            r = attempt(client, rec)
            usd = cost_usd(MODEL, r['usage']) or 0.0
            with lock:
                spent[0] += usd
                if spent[0] > a.cap_usd:
                    stop.set()
            tries.append({**r, 'usd': usd})
            if r['ok'] or (r.get('anomaly') or {}).get('kind') in ('refusal', 'content_filter'):
                break
        final = tries[-1] if tries else {'ok': False, 'anomaly': {'kind': 'not_sent', 'detail': 'cap reached'}}
        dropped = []
        if not final['ok'] and final.get('parsed') is not None:  # keep the record, drop only the items that failed the checks
            output, dropped = rx.drop_items(final['parsed'], final['problems'])
            removed = output.pop('_removed', [])
            final = {**final, 'ok': True, 'output': output, 'removed': removed}
        return {'platform': rec['platform'], 'record_id': rec['record_id'], 'ok': final['ok'], 'output': final.get('output'),
                'anomaly': None if final['ok'] else final.get('anomaly'), 'dropped_items': dropped, 'removed_items': final.get('removed', []), 'attempts': [{k: t.get(k) for k in ('ok', 'anomaly', 'usage', 'usd', 'seconds', 'effort')} for t in tries]}

    t0 = time.time()
    dropped_total = 0
    with cf.ThreadPoolExecutor(a.workers) as ex, open(os.path.join(out_dir, 'responses.jsonl'), 'w') as f:
        for res in ex.map(work, recs):
            dropped_total += len(res['dropped_items'])
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    wall = time.time() - t0
    schemas = {p: schema_sha256(strict_json_schema(rx.output_model(p))) for p in ('reddit', 'webmd')}
    manifest = {'run_id': a.run_id, 'set': a.set, 'records': len(recs), 'model': MODEL, 'reasoning_effort': EFFORT,
                'max_output_tokens': MAX_OUT, 'prompt_version': rx.PROMPT_VERSION,
                'prompt_sha256': {p: sha256_text(t) for p, t in rx.PROMPTS.items()}, 'schema_sha256': schemas,
                'workers': a.workers, 'dropped_items': dropped_total, 'wall_seconds': round(wall, 1), 'spent_usd_standard': round(spent[0], 4),
                'cap_usd': a.cap_usd, 'stopped_by_cap': stop.is_set(), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    json.dump(manifest, open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
