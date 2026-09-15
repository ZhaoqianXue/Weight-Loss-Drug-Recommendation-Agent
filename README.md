# Weight Loss Drug Recommendation Agent

A research prototype for structured extraction and adverse-event standardization of WebMD reviews, with TableRAG, a Neo4j knowledge graph, and a deterministic website assistant.

The current frozen dataset contains **2,727 reviews**, covering **4 generic names and 8 brands**; 2,681 reviews have text. Annotations are historically reused for 2,344 records; 383 remain pending (381 with text, 2 without). Pending means unknown, not absence of side effects. Historical annotations have not been clinically revalidated. See [project scope](docs/project-scope.md).

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/weightloss/` | Installable research package: ingestion, extraction, standardization, embeddings, retrieval, pipeline, evaluation |
| `configs/pipeline.json` | Explicit selection of the frozen snapshot, annotation seeds, input/output paths |
| `configs/drugs.json` | Canonical medication catalog and aliases |
| `apps/web/` | Flask adapter, HTML template, frontend source |
| `data/raw/webmd/2026-09-12/` | Frozen collection snapshot; original timestamps and bytes preserved |
| `data/external/` | Annotation seeds, terminology and historical prescribing-information PDFs |
| `data/interim/migrated-2026-09-14/` | Frozen extraction results |
| `data/processed/migrated-2026-09-14/` | Frozen standardized results and dataset manifest |
| `artifacts/embeddings/legacy/` | Preserved terminology embeddings still required by the baseline |
| `artifacts/web/`, `artifacts/indexes/` | Generated website and new retrieval indexes |
| `experiments/legacy/`, `notebooks/legacy/` | Historical experiments; see their execution limits before use |
| `results/` | Run metadata, metrics and preserved historical results |
| `archive/` | Historical backups, captured pages and stale indexes |
| `tests/` | Offline regression tests and synthetic fixtures |
| `docs/` | Scope, provenance, history, references and reproducibility guides |

All maintained Python commands use the installed `weightloss` CLI. Temporary legacy wrappers have been removed; canonical code lives under `src/weightloss/`. Tests are divided into `tests/unit/`, `tests/integration/`, `tests/frontend/` and `tests/fixtures/`. The optional model-adapter tests are under `tests/integration/research/` and run explicitly with `make test-research`.

Local `.venv/`, `.venv-research/`, `.cache/` and Git metadata are runtime/tool directories, excluded from the source layout. `scripts/` contains the migration and structure auditors. Completed planning records are under `docs/refactoring/planning/`, and the historical local QA note is under `docs/history/`.

## Install the verified offline environment

Python 3.11 and Node.js 22 are the tested local runtime versions. From the repository root:

```sh
uv venv --python 3.11 .venv
uv pip sync requirements/offline.lock --python .venv/bin/python
uv pip install --python .venv/bin/python --no-deps -e .
make check
```

The offline environment supports collection parsing, data validation, synthetic replay, the static site and Flask. No model credentials are needed for these checks. The CLI has no mandatory third-party dependencies; optional dependency groups are declared in `pyproject.toml`. See [environment details](requirements/README.md) for research dependencies and platform constraints.

An editable install locates this checkout without relying on the current working directory. A wheel installation requires `WEIGHTLOSS_ROOT=/absolute/path/to/WeightLoss`, or `weightloss --root /absolute/path/to/WeightLoss ...`. Data and web assets are intentionally outside the Python wheel.

## Validate and preview

```sh
.venv/bin/weightloss validate
.venv/bin/weightloss build-web
.venv/bin/weightloss serve
```

Open [the local preview](http://127.0.0.1:5001). The built site is self-contained relative to its output directory; visualization libraries still load from external CDNs. Its Medication Experience Assistant computes summaries, ratings, reported side-effect counts and review browsing directly from the selected CSV. It does not call the LLM, `/chat`, TableRAG or GraphRAG.

`build-web` validates one processed dataset, copies its CSV/catalog/manifest into the build, and replaces the prior build only after staging succeeds. Generated web files are not independent research inputs. Original pre-refactor copies remain traceable in the [migration map](docs/refactoring/migration-map.json).

## Reproduce the offline synthetic example

```sh
.venv/bin/weightloss reproduce-fixture --output-dir .cache/my-fixture
```

Use a fresh destination each time. This replays synthetic reviews and frozen annotation responses through refresh, validation, summary metrics, static build and run freezing. It checks hand-specified expected metrics and calls no external model. Results appear under `.cache/my-fixture/results/fixture/`. This exercises infrastructure; it is not an evaluation of extraction accuracy.

## Work on a new run

The migrated baseline is frozen. Create a separate run before processing:

```sh
.venv/bin/weightloss new-run --run-id my-experiment
.venv/bin/weightloss --config configs/runs/my-experiment.json refresh
```

Install the [research environment](requirements/README.md), then explicitly invoke model steps:

```sh
.venv-research/bin/weightloss --config configs/runs/my-experiment.json extract --limit 10
.venv-research/bin/weightloss --config configs/runs/my-experiment.json standardize --limit 10
.venv-research/bin/weightloss --config configs/runs/my-experiment.json validate
.venv-research/bin/weightloss --config configs/runs/my-experiment.json stats
.venv-research/bin/weightloss --config configs/runs/my-experiment.json build-web
.venv-research/bin/weightloss --config configs/runs/my-experiment.json freeze
```

Set `OPENAI_API_KEY` in the environment before model calls. `.env.example` documents variable names; `.env` is not automatically loaded. `WEIGHTLOSS_CONFIG` can select a configuration for the CLI; set `WEIGHTLOSS_ROOT` as well when using an external configuration without `project_root`.

Processing, collection, build, evaluation, retrieval and freeze commands write manifests under the selected run's `results/.../operations/`, including code and input hashes, configuration, installed packages, status and coverage. `new-run` records initialization separately; `validate` and `serve` do not create operation manifests. Provider event payloads are kept locally and ignored by Git. A failed model step retains its checkpoint; refresh the manifest after recovery by completing standardization. Validation rejects an inconsistent checkpoint before web publication. Frozen configurations reject refresh, extraction and standardization.

The historical annotation seed is an active input under `data/external/annotation_seeds/pre_2026_refresh/`; do not remove it as a backup. See [data dictionary and lifecycle](data/README.md).

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

TableRAG, GraphRAG and the current scientific methods remain in place. Proposed changes to terminology, graph schema, retrieval strategy and Reddit ingestion remain undecided in [TODO](TODO.md). Reddit ingestion has not been implemented. The prescribing-information PDFs and UMLS v2–v6 results are historical; no complete current FDA retrieval integration is claimed.

## Documentation and provenance

- [Reproducibility and operation guide](docs/reproducibility.md)
- [Data dictionary and lifecycle](data/README.md)
- [Contribution guide](CONTRIBUTING.md)
- [Project scope](docs/project-scope.md), [Reddit sources](docs/reddit-data-sources.md), [assistant validation](docs/chatbot-validation.md)
- [Results index](results/README.md), [architecture plan](docs/architecture-refactoring-plan.md), [implementation report](docs/refactoring/implementation.md)
- [Migration map](docs/refactoring/migration-map.json), [pre-refactor baseline](docs/refactoring/baseline.json), [historical execution records](docs/history/execution-records.md)

Historical records retain the meaning of their original dates; their old path names can be resolved through the migration map. No Git history was rewritten. Software authorship/license selection and data release permissions remain separate release decisions; this reorganization grants no new license and does not change the release scope of review text.
