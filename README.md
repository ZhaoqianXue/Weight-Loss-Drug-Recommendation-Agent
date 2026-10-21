# Weight Loss Drug Recommendation Agent

A research prototype for structured extraction and adverse-event standardization of WebMD reviews, with a Neo4j knowledge graph and a deterministic website assistant.

The frozen baseline contains **2,727 reviews, four generic names and eight brands**. There are 2,681 reviews with text, 2,344 historically reused annotations and 383 pending records (381 with text). Pending annotations mean unknown. Historical annotations have not been clinically revalidated. See [project scope](docs/project-scope.md).

## Get started

With Python 3.11, Node.js 22, pnpm and uv available, run from this checkout:

```sh
make install
make web-install
make test
make reproduce
make demo
```

Open [the local demo](http://127.0.0.1:5001); stop it with Ctrl-C. `make demo PORT=5002` selects another port. The site reads the local Neo4j knowledge graph (`python -m weightloss.graph.neo4j_db start`); the other commands need no model credentials, and the page loads no external resources.

`make reproduce` runs eight synthetic reviews and fixed annotation responses through refresh, validation, metrics, web build and freezing. It writes a fresh `.cache/fixture-<timestamp>/results/fixture/` directory. Use `make reproduce OUTPUT=.cache/my-fixture` to choose a fresh destination. This tests reproducibility of the pipeline, not model accuracy or full paper results.

## Run the AskGLP1 website on your own computer

Use this for the user study. You need macOS with [Homebrew](https://brew.sh), and a key for the OpenAI API.

1. Install the tools once: `brew install python@3.11 node pnpm uv neo4j`. Neo4j brings Java 21.
2. Download the site data bundle (`askglp1-site-data-*.zip`, about 80 MB) from the project's Google Drive folder. It holds the knowledge graph and the frozen FAERS tables. These are not in this public repository, so never commit or share them.
3. In this folder, create a file named `.env` with one line, `OPENAI_API_KEY=<your key>`. The file is git-ignored.
4. Run `make site-setup BUNDLE=<path to the zip>`. It installs the Python and frontend packages, unpacks the data, imports the graph into a Neo4j database inside this folder, builds the analysis database and checks the key with OpenAI. It takes about three minutes. You can run it again safely; finished steps are skipped.
5. Run `make demo` and open http://127.0.0.1:5001. Stop the site with Ctrl-C. Later sessions need only `make demo`, which also starts the database.

Each question costs about one US cent, and a server process stops answering after $5 (`WEIGHTLOSS_ASK_BUDGET_USD=20 make demo` raises the cap). An answer takes 30 to 120 seconds.

The website, AskGLP1, has a FastAPI backend (`src/weightloss/web/`) and a React frontend (`apps/web/`, see its [README](apps/web/README.md) for development commands). It reads the knowledge graph through the agent's graph tools, so its counts follow the same rules as the agent's, and it runs the agent (`weightloss.agent`) for questions, which needs `OPENAI_API_KEY` and is capped at $5 of model spend per server process (`WEIGHTLOSS_ASK_BUDGET_USD`). Live research operations have separate [advanced commands](docs/reproducibility.md#advanced-commands).

## Repository structure

```text
src/weightloss/       # Core package and one CLI
apps/web/            # Website frontend (Vite, React, TypeScript)
configs/             # Catalog, default selection and run configs
data/                # raw, external, interim, processed, indexes
results/<run_id>/    # Run records, metrics and generated demo/
tests/               # unit, integration, fixtures
requirements/        # Offline and platform-specific research locks
docs/                # Current scope, operations, roadmap, references
archive/             # Historical experiments, notebooks, results and audits
```

These are the nine visible root directories. Hidden environments, Git metadata and `.cache/` are local tooling. Optional metrics and figures are created only by actual runs. Historical PI embedding scripts are archived; the baseline's active terminology embeddings remain in `data/external/terminology/`.

The default [configuration](configs/pipeline.json) selects the frozen migrated baseline. Its packaged website data live in `results/migrated-2026-09-14/demo/`; rebuilding validates the data and stages an atomic replacement. New runs use `results/<run_id>/demo/` and `data/indexes/<run_id>/`. Create a new run before changing derived data.

## Documentation

- [Operations and reproducibility](docs/reproducibility.md): collection, incremental processing, checkpoints, freezing, retrieval and provenance.
- [Environment setup](requirements/README.md): three extras (`demo`, `research`, `dev`) and verified platform limits.
- [Data dictionary](data/README.md), [results](results/README.md), [contributing](CONTRIBUTING.md).
- [Project scope](docs/project-scope.md), [roadmap](docs/roadmap.md), [assistant validation](docs/chatbot-validation.md), [Reddit sources](docs/reddit-data-sources.md).
- [Simplification verification](docs/simplification.md), [historical archive](archive/README.md).

An editable install supports the CLI outside the checkout. A wheel requires `weightloss --root /absolute/path/to/WeightLoss ...`; data and application assets remain in the checkout. Scientific methods, drug scope, authorship and release permissions are unchanged by the reorganization.

