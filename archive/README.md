# Historical archive

| Directory | Preserved material |
| --- | --- |
| [experiments](experiments/README.md) | Legacy extraction, UMLS, evaluation and PI embedding scripts |
| [notebooks](notebooks/README.md) | Two historical exploratory notebooks |
| [results](results/) | Original experiment samples, reports and catalog hashes |
| [environments](environments/) | Historical requirements and pre-simplification locks |
| [project-history](project-history/) | Dated notes, prior architecture plans, migration maps and old operation logs |
| `indexes/root/`, `indexes/website/` | Stale historical FAISS indexes |
| `data_backup/`, `website/data/`, `collection/` | Earlier data, excluded reviews, duplicated site data and captured pages |

These files are historical evidence. They do not become current experiments when moved. Rebuild retrieval indexes under `data/indexes/<run_id>/`; old indexes remain rejected by fingerprint checks. Active annotation seeds and terminology embeddings live under `data/external/`.

The [simplification migration map](project-history/simplification/migration-map.json) maps the previous layout to the current one and records immutable hashes. Chain it with [the original migration map](project-history/refactoring/migration-map.json) for earlier paths. Frozen JSON manifests retain their original bytes. Historical Markdown links were repaired, while dated descriptions and scientific claims retain their original scope.

Archived [audit tools](project-history/refactoring/tools/) apply to the pre-simplification checkout; they are not daily CI commands. The new one-time verification is described in [the implementation report](../docs/simplification.md). Local historical model caches remain under `.cache/legacy/` and are not added to Git.
