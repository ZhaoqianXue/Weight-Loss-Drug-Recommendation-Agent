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

Each CLI operation records start/end time, status, run ID, Git commit and dirty flag, Python source hashes, installed package versions, lockfile hashes, configuration, input/output hashes, annotation coverage, model identifiers and parameters. Source hashes supplement Git commit when the checkout has uncommitted changes. Historical model revisions are unknown; they are never reconstructed from filenames. Synthetic replay explicitly records that no model was used.

Operations use an exclusive per-run lock. If a process is forcibly killed, verify that its recorded PID is no longer running before removing `.operation.lock`. Reuse the working run to recover checkpoints; do not clear another process's lock. An operation marked completed means the command completed, not that all pending annotations were processed.

Model request metadata and parsed responses are logged when supplied by the provider. These `events.jsonl` payloads can contain study text and are ignored by Git. Run manifests reference local input paths and contain hashes; do not put tokens in run configuration. External model behavior can change, so recorded response replay is distinguished from a new model experiment.

## Environment and tests

See [requirements](../requirements/README.md). CI is configured for Python 3.11 on Ubuntu and macOS with Node.js 22. The local implementation report distinguishes checks actually executed from CI jobs merely configured.

The synthetic fixture contains eight entirely fabricated reviews, two fixed annotations and six pending records. It exercises the actual annotation-reuse pipeline, the expected count/mention assertions, dataset validation, static build and freeze. It does not test LLM quality, medical correctness or a live Neo4j instance.

`make check` also verifies the migrated current dataset and the historical byte audit. Old website data duplicates remain preserved in the migration provenance; the maintained application uses only generated build resources.

## Historical experiments

[Legacy experiment notes](../experiments/README.md) describe the preserved scripts and notebook limits. Files referring to missing FAERS spreadsheets or unavailable original environments are not claimed reproducible. Dated documents in `docs/history/` retain their original descriptions; their links are updated when possible and old names resolve through the [migration map](refactoring/migration-map.json).
