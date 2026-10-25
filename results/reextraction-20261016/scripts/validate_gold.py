"""Check a gold (or model) answer file against the v9 schema and local rules.

Usage: python validate_gold.py <sample.jsonl> <answers.jsonl> [--ids batch.jsonl]
Each answer line: {"record_id": ..., "platform": ..., "gold": {...}, "notes": [...]}. Prints one line per problem.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from pydantic import ValidationError
from weightloss.extraction.reextract import check_output, output_model


def problems_for(rec, answer):
    out = []
    try:
        parsed = output_model(rec['platform']).model_validate(answer)
    except ValidationError as e:
        return [f"schema: {err['loc']}: {err['msg']}" for err in e.errors(include_url=False)][:5]
    out += check_output(parsed, rec['sentences'])
    for d, drug in enumerate(parsed.drugs):
        if not drug.purposes and not drug.side_effects:
            out.append(f'drugs[{d}]: group without items')
    if rec['platform'] == 'reddit':
        w = parsed.writer
        if w.age_group is None and w.sex is None and w.evidence:
            out.append('writer: evidence without age or sex')
    return out


def main():
    sample = {(r['platform'], r['record_id']): r for r in map(json.loads, open(sys.argv[1]))}
    want = None
    if '--ids' in sys.argv:
        want = {(r['platform'], r['record_id']) for r in map(json.loads, open(sys.argv[sys.argv.index('--ids') + 1]))}
    seen, bad = set(), 0
    for n, line in enumerate(open(sys.argv[2]), 1):
        a = json.loads(line)
        key = (a['platform'], str(a['record_id']))
        if key not in sample:
            print(f'line {n}: unknown record {key}'); bad += 1; continue
        if key in seen:
            print(f'line {n}: duplicate {key}'); bad += 1
        seen.add(key)
        for p in problems_for(sample[key], a['gold']):
            print(f'{key[0]}:{key[1]}: {p}'); bad += 1
    if want is not None:
        for key in sorted(want - seen):
            print(f'missing {key[0]}:{key[1]}'); bad += 1
    print(f'checked {len(seen)} answers, {bad} problems')


if __name__ == '__main__':
    main()
