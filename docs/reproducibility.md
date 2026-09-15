# Reproducibility and operations

## Configuration

`configs/pipeline.json` selects the migrated, frozen baseline. Paths are resolved relative to the explicit research root, not the terminal working directory. `--root` / `WEIGHTLOSS_ROOT` chooses the checkout; `--config` / `WEIGHTLOSS_CONFIG` chooses the run. An external config may alternatively set `project_root` relative to its own location. The wheel includes the Python package, not study data or application assets.

`new-run --run-id NAME` clones the selected extraction and standardized files and records their hashes, then creates `configs/runs/NAME.json`. It does not switch the default. `--raw-dir PATH` may select a new validated collection; run `refresh` before validation because the copied annotations initially describe the parent snapshot.

## Lifecycle

1. Collect into a new raw directory or select an existing frozen collection.
2. Create a new working run and refresh its rows using matching annotations and the historical seed.
3. Run incremental extraction and standardization explicitly. Successful per-record checkpoints remain available after interruption. Missing credentials leave data unchanged.
4. Validate the processed result. Pending annotations may remain; coverage is explicit and never interpreted as zero side effects.
5. Build the website, compute metrics or build/import retrieval artifacts from that selected result.
6. Freeze the run after validation. Additional extraction requires another new run.

The active dataset is identified by configuration plus digest. The migrated run name indicates the migration date, not an invented historical experiment date. Collection time remains the timestamp in `collection_manifest.json`.

## Provenance records

Processing, collection, evaluation, retrieval and freeze commands record start/end time, status, run ID, Git commit and dirty flag, Python source hashes, an environment reference, configuration, input/output hashes, annotation coverage, model identifiers and parameters. `new-run` writes an initialization manifest; `reproduce-fixture` records its synthetic replay. The read-only `validate` command prints its check result, and `serve` starts the local server; neither creates an operation manifest. Source hashes supplement Git commit when the checkout has uncommitted changes. Historical model revisions are unknown; they are never reconstructed from filenames. Synthetic replay explicitly records that no model was used.

Environment inventories (Python, platform, installed packages and dependency-lock hashes) live once per fingerprint under `results/<run_id>/environments/`. Each stage references that file relative to the run directory and records its digest. A changed environment produces another inventory; a corrupted inventory is rejected. Stage source hashes include the extraction schema and prompts. Input hashes include terminology and the active terminology embeddings.

Ordinary `build-web` creates only `results/<run_id>/demo/build_manifest.json`, without another operation directory. That manifest includes the selected config, dataset digest, source hashes and hashes of the run/freeze records when present. Build the final demo after freezing if the demo must reference the final frozen record. Existing historical operation records are preserved under `archive/project-history/operations/`.

Operations and web builds use the same exclusive per-run lock. If a process is forcibly killed, verify that its recorded PID is no longer running before removing `.operation.lock`. Reuse the working run to recover checkpoints; do not clear another process's lock. An operation marked completed means the command completed, not that all pending annotations were processed.

Model request metadata and parsed responses are logged when supplied by the provider. These `events.jsonl` payloads can contain study text and are ignored by Git. Run manifests reference local input paths and contain hashes; do not put tokens in run configuration. External model behavior can change, so recorded response replay is distinguished from a new model experiment.

## Environment and tests

See [requirements](../requirements/README.md). CI is configured for Python 3.11 on Ubuntu and macOS with Node.js 22. The local implementation report distinguishes checks actually executed from CI jobs merely configured.

The synthetic fixture contains eight entirely fabricated reviews, two fixed annotations and six pending records. It exercises the actual annotation-reuse pipeline, the expected count/mention assertions, dataset validation, static build and freeze. It does not test LLM quality, medical correctness or a live Neo4j instance.

`make test` checks the selected data, Python behavior, frontend assertions and local Markdown links. `make check` also runs explicit data validation. Migration audits are archived one-time checks, outside daily CI. Old website data duplicates remain preserved in the migration provenance; the maintained application uses only generated build resources.

## Historical experiments

[Legacy experiment notes](../archive/experiments/README.md) describe the preserved scripts and notebook limits. Files referring to missing FAERS spreadsheets or unavailable original environments are not claimed reproducible. Dated documents in `archive/project-history/history/` retain their original descriptions; their links are updated when possible and old names resolve through the [migration map](../archive/project-history/refactoring/migration-map.json).

## Advanced commands

## Work on a new run

The migrated baseline is frozen. Create a separate run before processing:

```sh
.venv/bin/weightloss new-run --run-id my-experiment
.venv/bin/weightloss --config configs/runs/my-experiment.json refresh
```

Install the [research environment](../requirements/README.md), then explicitly invoke model steps:

```sh
.venv-research/bin/weightloss --config configs/runs/my-experiment.json extract --limit 10
.venv-research/bin/weightloss --config configs/runs/my-experiment.json standardize --limit 10
.venv-research/bin/weightloss --config configs/runs/my-experiment.json validate
.venv-research/bin/weightloss --config configs/runs/my-experiment.json stats
.venv-research/bin/weightloss --config configs/runs/my-experiment.json freeze
.venv-research/bin/weightloss --config configs/runs/my-experiment.json build-web
```

Set `OPENAI_API_KEY` in the environment before model calls. `.env.example` documents variable names; `.env` is not automatically loaded. `WEIGHTLOSS_CONFIG` can select a configuration for the CLI; set `WEIGHTLOSS_ROOT` as well when using an external configuration without `project_root`.

Processing, collection, evaluation, retrieval and freeze commands write manifests under the selected run's `results/.../operations/`, including code and input hashes, configuration, environment references, status and coverage. `new-run` records initialization separately; `validate` and `serve` do not create operation manifests. Provider event payloads are kept locally and ignored by Git. A failed model step retains its checkpoint; refresh the manifest after recovery by completing standardization. Validation rejects an inconsistent checkpoint before web publication. Frozen configurations reject refresh, extraction and standardization.

The historical annotation seed is an active input under `data/external/annotation_seeds/pre_2026_refresh/`; do not remove it as a backup. See [data dictionary and lifecycle](../data/README.md).

## Collect a new snapshot

```sh
.venv/bin/weightloss collect --output-dir data/raw/webmd/NEW_SNAPSHOT --run-dir .cache/webmd/NEW_COLLECTION
.venv/bin/weightloss new-run --run-id new-review-run --raw-dir data/raw/webmd/NEW_SNAPSHOT
.venv/bin/weightloss --config configs/runs/new-review-run.json refresh
```

Choose new snapshot and run names. Existing snapshot directories are rejected. The collector validates page identity, pagination, IDs and totals before publishing a staged snapshot. Resume an interrupted collection using the same checkpoint directory; start a fresh checkpoint directory for a new collection date. Collection does not switch the active research configuration automatically.

## Research retrieval and backend

```sh
.venv-research/bin/weightloss --config configs/runs/my-experiment.json build-index
.venv-research/bin/weightloss --config configs/runs/my-experiment.json import-graph
.venv-research/bin/weightloss --config configs/runs/my-experiment.json chatbot
.venv-research/bin/weightloss --config configs/runs/my-experiment.json serve --backend
```

Graph import requires a configured Neo4j server (`NEO4J_URI`, `NEO4J_PASSWORD`) and retains the empty-database/snapshot guard. Index use retains the dataset/model fingerprint guard. Historical FAISS indexes are archived as stale and are never relabeled current. Flask initializes the research chatbot only when `/chat` is requested. Complete live model, index-building and database workflows were not executed as part of the structural migration.

TableRAG, GraphRAG and the current scientific methods remain in place. Proposed changes to terminology, graph schema, retrieval strategy and Reddit ingestion remain undecided in [roadmap](roadmap.md). Reddit ingestion has not been implemented. The prescribing-information PDFs and UMLS v2–v6 results are historical; no complete current FDA retrieval integration is claimed.
