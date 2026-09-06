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
| `data_webmd/` | Original reviews for seven brands |
| `data_extracted/` | Structured extraction results |
| `data_standardized/` | Baseline and versioned UMLS outputs and reports |
| `data_embedded/` | Adverse-event terminology and embeddings |
| `data_prescribing_information/` | Seven prescribing-information PDFs |
| `database_table/` | FAISS schema and cell indexes |
| `data_literature/` | Reference papers |
| `data_backup/` | Earlier data and graph artifacts retained for reference |

The primary original, extracted, and baseline standardized datasets each contain 2,373 review records. The chatbot currently uses the baseline standardized dataset. UMLS v6 outputs are retained as experiments, not silently substituted into the chatbot.

## View the interactive graph

From the repository root, run:

```sh
python3 -m http.server 5001 --bind 127.0.0.1 --directory code_website
```

Open `http://127.0.0.1:5001/templates/knowledge_graph.html`. This preview needs no LLM credentials or Neo4j. The page loads JavaScript libraries from external CDNs. Its chat box uses demonstration responses and is not connected to the backend API.

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

These commands are separate workflows. Extraction and standardization make model calls and write output files. Neo4j must be available at the configured local URI with the graph imported before GraphRAG can answer questions. The Flask backend provides `/chat`, but connecting the page's demonstration chat to that endpoint remains future work.

Prescribing-information processing scripts and FDA-related standardization prompts are present. A complete prescribing-information/FDA retrieval integration in the chatbot has not been established. Some historical notebooks reference FAERS Excel files that are not included in this repository.

## Version and data policy

The outer working copy takes precedence for conflicting files. In particular, extraction assigns side-effect relations to the primary review drug; the older associated-drug implementation remains available in Git history. See `docs/consolidation.md` for the migration decisions.

Notebook outputs are cleared. Slides and presentation build files are excluded. Large regenerable embedding caches are ignored, while versioned data and reports are retained. API credentials are read from environment variables. Original Git history is preserved without rewriting historical source or credentials.
