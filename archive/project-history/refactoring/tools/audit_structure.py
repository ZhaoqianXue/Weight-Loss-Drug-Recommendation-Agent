"""Check the maintained repository layout against the approved architecture."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = {'apps','archive','artifacts','configs','data','docs','experiments','notebooks','requirements','results','scripts','src','tests'}
actual = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')}
required = [
    'src/weightloss', 'apps/web', 'data/raw', 'data/external', 'data/interim', 'data/processed',
    'artifacts/embeddings', 'artifacts/indexes', 'artifacts/web',
    'tests/unit', 'tests/integration', 'tests/frontend', 'tests/fixtures',
    'docs/references', 'docs/refactoring/planning',
    'tests/unit/test_collection.py', 'tests/unit/test_annotations.py',
    'tests/integration/test_snapshot.py', 'tests/integration/test_architecture.py',
    'tests/integration/research/test_adapters.py',
]
retired = ['outputs','task_plan.md','findings.md','progress.md','tests/test_refresh.py','tests/test_architecture.py','tests/research','.venv-wheel']
issues = [f'Unexpected top-level directory: {n}' for n in sorted(actual-expected)]
issues += [f'Missing top-level directory: {n}' for n in sorted(expected-actual)]
issues += [f'Missing planned path: {n}' for n in required if not (root/n).exists()]
issues += [f'Retired path still exists: {n}' for n in retired if (root/n).exists()]
# New command wrappers or sys.path workarounds must not reintroduce the old layout.
for path in (root/'src/weightloss').rglob('*.py'):
    if 'sys.path.insert(' in path.read_text():
        issues.append(f'Import-path workaround: {path.relative_to(root)}')
print(json.dumps({'status':'passed' if not issues else 'failed','top_level_directories':sorted(actual),'issues':issues},indent=2))
if issues:
    raise SystemExit(1)
