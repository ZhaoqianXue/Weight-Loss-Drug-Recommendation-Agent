# Data dictionary and lifecycle

## Current selection

`configs/pipeline.json` selects `raw/webmd/2026-09-12/`, `interim/migrated-2026-09-14/`, and `processed/migrated-2026-09-14/`. The migration date names the imported run, not the original extraction date. Exact collection times remain in each row and the collection manifest. Raw snapshots and frozen runs must not be edited in place.

## Primary fields

| Field | Meaning |
| --- | --- |
| `Review ID` | Unique source review identifier; preserved across processing stages |
| `Drug Name` / `Brand Name` | Generic molecule / canonical brand from the maintained catalog |
| `Date` | Source review date; standardized output normalizes to YYYY-MM-DD |
| `User`, `Age`, `Gender`, `Patient Type` | Source-provided profile fields; not verified demographic measurements |
| `Medication Duration`, `Condition` | Source-provided duration and condition |
| `Overall Rating`, `Effectiveness`, `Ease of Use`, `Satisfaction` | Source ratings on a 1–5 scale |
| `Likes`, `Dislikes` | Source reaction counts |
| `Textual Review` | Review narrative, possibly empty for rating-only records |
| `Source URL`, `Source Page`, `Collected At`, `Source Visibility` | Collection provenance and rendered/embedded visibility |
| `Extraction Status` | `historical_reused`, `pending`, `completed`, or `failed`; pending/failed do not support absence claims |
| `Annotation Method` | Origin of reused or incrementally produced annotations |
| `structured_info`, `relations` | JSON-encoded extraction object / relation list |
| `Standardization Status` | Origin/completion state for standardized annotations |
| `standardized_info`, `standardized_relations` | JSON-encoded standardized object / relation list |

Raw, extracted and standardized files have different schemas and hashes. They preserve ordered review IDs and source fields, except the documented date normalization. A website build's standardized CSV is byte-identical to its selected processed CSV. Missing narrative, missing annotation and explicit no-symptom annotation are different states.

## External inputs

- `external/annotation_seeds/pre_2026_refresh/`: historical files and their original manifest. The two annotation CSVs are active refresh inputs; the original raw snapshot is retained for provenance.
- `external/terminology/ae.csv`: baseline terminology input; source/version completeness remains historical, see the embedding provenance record.
- `external/prescribing_information/`: historical PDFs, not an updated regulatory corpus.
- `external/terminology/embedded_ae.csv`: required baseline embedding input retained because full original reproduction metadata are incomplete.

No new source permissions are granted by moving these files. Review-text publication and terminology licensing remain separate decisions recorded in [the roadmap](../docs/roadmap.md). Do not delete tracked inputs or historical artifacts solely because a directory is now conventionally considered a cache.

## Manifests

`collection_manifest.json` validates raw hashes and collection coverage. `dataset_manifest.json` validates standardized output and its raw source. `results/<run>/operations/*/manifest.json` adds code, environment and execution provenance. Original historical manifests are retained byte-for-byte even when they mention old paths; use the [current migration map](../archive/project-history/simplification/migration-map.json), chained with the [original map](../archive/project-history/refactoring/migration-map.json) to resolve them.

`indexes/<run_id>/` stores generated retrieval indexes, guarded by dataset and model fingerprints and ignored by Git. The selected run's generated CSV copy lives in `results/<run_id>/demo/static/`.
