"""Verify each frozen migrated artifact against the pre-refactor baseline."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
mapping=json.loads((root/'docs/refactoring/migration-map.json').read_text())
missing=[];changed=[];verified=0
for old,record in mapping.items():
    for extra in record.get('additional_paths', []):
        if not (root/extra).is_file(): missing.append(extra)
    target=root/record['new_path']
    if not target.is_file():
        missing.append(str(target.relative_to(root)))
    elif record['preserve_bytes']:
        if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256_before']:
            changed.append(record['new_path'])
        else: verified+=1
report={'status':'passed' if not missing and not changed else 'failed','mapped_files':len(mapping),'immutable_files_verified':verified,'missing':missing,'changed_immutable':changed}
print(json.dumps(report,indent=2))
if missing or changed:raise SystemExit(1)
