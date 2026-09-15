# Results index

| Location | Status |
| --- | --- |
| `migrated-2026-09-14/manifest.json` | Provenance of the preserved baseline; not a new scientific experiment |
| `migrated-2026-09-14/operations/` | Validation/build operations performed after migration |
| `legacy/standardization/` | Original baseline samples, UMLS v2–v6, comparisons and evaluation reports |
| `legacy/extraction/` | Historical extraction samples |
| `legacy/catalog.json` | File hashes and original paths for historical result artifacts; unknown experiment metadata remain unknown |
| Future `<run_id>/` | Run initialization, operation manifests, metrics and optional frozen-state record |

No new paper-level performance claim is made by this refactor. Historical files are not current results. Formal paper figures/tables should link to the generating command, selected run configuration and output hash when they are produced.

Large regenerated artifacts and model response events are stored outside the maintained result metadata. Manifests and small reproducibility records are not blanket-ignored. Synthetic smoke outputs live in an explicitly selected local output directory, usually `.cache/`.
