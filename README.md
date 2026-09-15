# Weight Loss Drug Recommendation Agent

A research prototype for structured extraction and adverse-event standardization of WebMD reviews, with TableRAG, a Neo4j knowledge graph, and a deterministic website assistant.

The frozen baseline contains **2,727 reviews, four generic names and eight brands**. There are 2,681 reviews with text, 2,344 historically reused annotations and 383 pending records (381 with text). Pending annotations mean unknown. Historical annotations have not been clinically revalidated. See [project scope](docs/project-scope.md).

## Get started

With Python 3.11, Node.js 22 and uv available, run from this checkout:

```sh
make install
make test
make reproduce
make demo
```

Open [the local demo](http://127.0.0.1:5001); stop it with Ctrl-C. `make demo PORT=5002` selects another port. These commands need no model credentials. Visualization libraries load from external CDNs.

`make reproduce` runs eight synthetic reviews and fixed annotation responses through refresh, validation, metrics, web build and freezing. It writes a fresh `.cache/fixture-<timestamp>/results/fixture/` directory. Use `make reproduce OUTPUT=.cache/my-fixture` to choose a fresh destination. This tests reproducibility of the pipeline, not model accuracy or full paper results.

The website computes summaries and evidence browsing from the selected CSV. Its assistant does not invoke the research chatbot or graph retrieval. Live research operations have separate [advanced commands](docs/reproducibility.md#advanced-commands).

## Repository structure

```text
src/weightloss/       # Core package and one CLI
apps/web/            # Website and Flask source
configs/             # Catalog, default selection and run configs
data/                # raw, external, interim, processed, indexes
results/<run_id>/    # Run records, metrics and generated demo/
tests/               # unit, integration, frontend, fixtures
requirements/        # Offline and platform-specific research locks
docs/                # Current scope, operations, roadmap, references
archive/             # Historical experiments, notebooks, results and audits
```

These are the nine visible root directories. Hidden environments, Git metadata and `.cache/` are local tooling. Optional metrics and figures are created only by actual runs. Historical PI embedding scripts are archived; the baseline's active terminology embeddings remain in `data/external/terminology/`.

The default [configuration](configs/pipeline.json) selects the frozen migrated baseline. Its generated website lives in `results/migrated-2026-09-14/demo/`; rebuilding validates the data and stages an atomic replacement. New runs use `results/<run_id>/demo/` and `data/indexes/<run_id>/`. Create a new run before changing derived data.

## Documentation

- [Operations and reproducibility](docs/reproducibility.md): collection, incremental processing, checkpoints, freezing, retrieval and provenance.
- [Environment setup](requirements/README.md): three extras (`demo`, `research`, `dev`) and verified platform limits.
- [Data dictionary](data/README.md), [results](results/README.md), [contributing](CONTRIBUTING.md).
- [Project scope](docs/project-scope.md), [roadmap](docs/roadmap.md), [assistant validation](docs/chatbot-validation.md), [Reddit sources](docs/reddit-data-sources.md).
- [Simplification verification](docs/simplification.md), [historical archive](archive/README.md).

An editable install supports the CLI outside the checkout. A wheel requires `weightloss --root /absolute/path/to/WeightLoss ...`; data and application assets remain in the checkout. Scientific methods, drug scope, authorship and release permissions are unchanged by the reorganization.
