# Weight Loss Drug Recommendation Agent

A research prototype that extracts structured information from WebMD patient reviews, standardizes adverse-event terminology, and answers questions with TableRAG and a Neo4j knowledge graph.

## Repository layout

This root directory is the canonical repository. It combines the later outer working directory with the original Git repository history. There is no nested repository.

| Directory | Contents |
| --- | --- |
| `code_scraping/` | WebMD review collection |
| `code_extraction/` | LLM extraction prompts, schema, and batch processing |
| `code_standardization/` | Baseline and UMLS standardization experiments and evaluation |
| `code_embedding/` | Adverse-event embeddings and experimental prescribing-information extraction |
| `code_chatbot/` | Canonical chatbot, TableRAG index builder, and Neo4j loader |
| `code_website/` | Flask entry point and interactive graph interface |
| `data_webmd/` | Validated current reviews for four generic names and eight brands |
| `data_extracted/` | Structured extraction results |
| `data_standardized/` | Baseline and versioned UMLS outputs and reports |
| `data_embedded/` | Adverse-event terminology and embeddings |
| `data_prescribing_information/` | Seven historical prescribing-information PDFs; not an updated regulatory corpus |
| `database_table/` | Historical FAISS indexes; rebuild before use with the current dataset |
| `data_literature/` | Reference papers |
| `data_backup/` | Earlier data and graph artifacts retained for reference |

The current raw, extraction, baseline standardized, and website datasets each contain **2,727 review records**, covering **4 generic names and 8 brands**. Of these, 2,681 contain review text. Historical annotations were reused for 2,344 matching records; **383 records carry the pending status: 381 have a nonempty text field, while 2 are rating-only records with no text to extract**. The reused historical outputs also include 44 empty-text records; see [the historical empty-text audit](docs/empty-review-audit-2026-09-12.md). Pending annotations are unknown, not evidence of no side effects. See [the current scope and counts](docs/project-scope.md), [the dataset manifest](data_standardized/dataset_manifest.json) for per-brand coverage, and [the historical refresh report](docs/data-refresh-2026-09-11.md) for provenance and outstanding model/index work.

Wegovy HD is an alias of **Wegovy**; Victoza 2-Pak and Victoza 3-Pak are aliases of **Victoza**, not extra datasets. Generic names are represented in `Drug Name`; brands are represented in `Brand Name`. Shared generic/brand review pages are collected once. The maintained catalog is [config/drugs.json](config/drugs.json). **Combination products are excluded regardless of review availability**, including Soliqua (insulin glargine + lixisenatide) and Xultophy (insulin degludec + liraglutide). See [the project scope](docs/project-scope.md) for exclusions and [Reddit data sources](docs/reddit-data-sources.md) for the eight retained brands.

UMLS versions, top-10 samples, embeddings, prescribing-information documents, notebooks, and previous evaluation reports remain historical experiments. They were not relabeled as current results. The pre-refresh canonical CSVs are preserved in `data_backup/pre_2026_refresh/`.

## View the interactive graph

From the repository root, run:

```sh
python3 -m http.server 5001 --bind 127.0.0.1 --directory code_website
```

Open `http://127.0.0.1:5001/templates/knowledge_graph.html`. This preview needs no LLM credentials or Neo4j. The page loads JavaScript libraries from external CDNs. Its Medication Experience Assistant computes answers directly from the loaded CSV: medication summaries, rating comparisons, reported side-effect counts, source-review browsing, and graph actions. It uses deterministic guided dialogue and supported text intents, not a general-purpose LLM or the backend API.

## Refresh review data

Install the small collection dependency set with `python3 -m pip install -r code_scraping/requirements.txt`, then run from this directory:

```sh
python3 code_scraping/batch_scraper.py
python3 code_pipeline/refresh_data.py
python3 -m unittest discover -s tests -p 'test_*.py'
node tests/review-assistant.test.js
```

The collector validates all eight brands' review-page identities, pagination, unique review IDs and headline totals before publishing. Page checkpoints are under `.cache/webmd/`; `--run-dir PATH` resumes an interrupted snapshot. Start a new run directory for a new collection date. The refresh command aligns annotations by content, updates all canonical CSV copies and manifests, and requires no model credentials.

## Backend and research workflows

The existing dependency snapshot is `code_website/requirements.txt`; additional standardization dependencies are recorded in `code_standardization/`. These are historical research environments, not a verified cross-platform installation lockfile.

Configure `OPENAI_API_KEY`, `UMLS_API_KEY` when needed, and `NEO4J_PASSWORD` in the shell environment. `.env.example` lists their names. The application does not automatically load a `.env` file.

Run scripts from the repository root so their relative data paths resolve:

```sh
python3 code_extraction/extract_all.py
python3 code_standardization/standardize_all.py
python3 code_chatbot/table_loader.py
python3 code_chatbot/graph_loader.py
python3 code_chatbot/chatbot.py
# Alternatively, start the Flask backend:
python3 code_website/flask_app.py
```

Extraction and standardization process pending reviews incrementally, checkpoint results, and support `--limit N`. They require model dependencies and `OPENAI_API_KEY`; missing credentials leave data unchanged. Standardization publishes updated website copies. FAISS indexes require rebuilding; missing or stale dataset fingerprints are rejected. Import the current snapshot into an empty Neo4j database before GraphRAG use; the loader refuses to mix a new snapshot into an existing graph. No external database was modified during this refresh. The Flask backend provides `/chat` as a separate research workflow. The page uses the local dataset-backed assistant instead; it does not invoke TableRAG, GraphRAG, or `/chat`. Run `node tests/review-assistant.test.js` from the repository root to validate the assistant (Node.js and Python 3 required). See `docs/chatbot-validation.md` for scope and results.

Prescribing-information processing scripts and FDA-related standardization prompts are present. A complete prescribing-information/FDA retrieval integration in the chatbot has not been established. Some historical notebooks reference FAERS Excel files that are not included in this repository.

## Version and data policy

The outer working copy takes precedence for conflicting files. In particular, extraction assigns side-effect relations to the primary review drug; the older associated-drug implementation remains available in Git history. See `docs/consolidation.md` for the migration decisions.

Notebook outputs are cleared. Slides and presentation build files are excluded. Large regenerable embedding caches are ignored, while versioned data and reports are retained. API credentials are read from environment variables. Original Git history is preserved without rewriting historical source or credentials.
