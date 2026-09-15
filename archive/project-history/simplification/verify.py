"""One-time simplification acceptance; intentionally outside routine CI."""
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXPECTED = {'src','apps','configs','data','results','tests','requirements','docs','archive'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    mapping = json.loads((HERE/'migration-map.json').read_text())
    moves = json.loads((HERE/'moves.json').read_text())
    def relocated(path):
        for old, new in sorted(moves.items(), key=lambda item: -len(item[0])):
            if path == old or path.startswith(old+'/'):
                return new+path[len(old):]
        return path
    issues = []
    immutable = 0
    local_cache = []
    for old, entry in mapping.items():
        target = ROOT/entry['new_path']
        # Private ignored caches are verified when present, but not required in a clone.
        if entry['new_path'].startswith('.cache/') and not target.exists():
            local_cache.append(entry['new_path'])
            continue
        if not target.is_file():
            issues.append('Missing: '+entry['new_path'])
        elif entry['preserve_bytes']:
            if digest(target) != entry['sha256_before']:
                issues.append('Changed immutable file: '+entry['new_path'])
            else:
                immutable += 1
    original = json.loads((HERE.parent/'refactoring/migration-map.json').read_text())
    original_verified = 0
    for old, entry in original.items():
        targets = [entry['new_path'], *entry.get('additional_paths', [])]
        for target in targets:
            if not (ROOT/relocated(target)).is_file():
                issues.append('Missing original mapping: '+target)
        if entry['preserve_bytes']:
            target = ROOT/relocated(entry['new_path'])
            if target.is_file() and digest(target) == entry['sha256_before']:
                original_verified += 1
            else:
                issues.append('Original immutable hash mismatch: '+entry['new_path'])
    for name in ('offline.lock','research-macos-x86_64-py311.lock'):
        archived = ROOT/'archive/environments/pre-simplification'/name
        if digest(archived) != mapping['requirements/'+name]['sha256_before']:
            issues.append('Original lock changed: '+name)
    actual = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith('.')}
    if actual != EXPECTED:
        issues.append('Unexpected root layout: '+str(sorted(actual)))
    configs = []
    for path in (ROOT/'configs').rglob('*.json'):
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or 'web_build' not in config:
            continue
        run = config['run_id']
        for key, value in {'web_build':f'results/{run}/demo','indexes':f'data/indexes/{run}', 'terminology_embeddings':'data/external/terminology/embedded_ae.csv'}.items():
            if config[key] != value:
                issues.append(f'Obsolete {key}: {path}')
        configs.append(str(path.relative_to(ROOT)))
    extras = sorted(tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['optional-dependencies'])
    if extras != ['demo','dev','research']:
        issues.append('Unexpected extras: '+str(extras))
    result = {'status':'passed' if not issues else 'failed','top_level_directories':sorted(actual),'mapped_files':len(mapping),'immutable_files_verified':immutable,'original_mappings_verified':len(original),'original_immutable_files_verified':original_verified,'local_caches_absent':local_cache,'current_run_configs':configs,'extras':extras,'issues':issues}
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    verify()
