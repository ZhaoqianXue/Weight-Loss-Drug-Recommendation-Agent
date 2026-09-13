# TODO

Recorded 2026-09-12 while planning the WWW 2027 submission (WebMD + Reddit, 4 generics / 8 brands). Items are ordered by dependency: 2 blocks 1, and 1 blocks 3.

## 1. Rebuild the knowledge graph schema as an evidence-level, provenance-aware graph

Current state: code_chatbot/graph_loader.py creates only Drug, Condition and SideEffect nodes with TREATS, CAUSES and REVIEWED_FOR edges. Counts, source platform, time, polarity ("no side effects"), formulation and switching are lost at load time, so the graph cannot support cross-platform comparison or a provenance-aware agent.

Target: one Evidence node per extracted assertion (hashed source ID, platform, community or page, month, self-report flag, polarity, severity, dose, duration, resolved flag) linked to Molecule, Brand (including an "Unspecified" brand per molecule), Indication, Symptom (MedDRA PT with SOC hierarchy, or UMLS CUI with semantic-type constraints), Outcome, Theme (cost, insurance, shortage, compounding, dose titration) and Source (Platform to Community). Add SWITCHED_FROM / SWITCHED_TO between Evidence and Brand. Materialise per-platform, per-month Brand-Symptom aggregates with numerators and denominators. Keep the dataset fingerprint guard and the empty-database import rule.

## 2. Rebuild the standardization layer

Current state: the standardized vocabulary in data_standardized/standardized_reviews_all.csv has 545 terms and contains systematic mapping errors, for example "gas gangrene" (60 occurrences, from "gas"), "irritable bowel syndrome" (70) and "burning sensation" (79). These would be visible in any published graph.

Target: retrieval-constrained LLM normalization. Retrieve candidate terms from MedDRA (licensed) or UMLS synonym tables, restrict UMLS candidates to sign/symptom, finding and disease semantic types, let the model choose among candidates or return "no match", and log the candidate list. Validate on 300 annotated side-effect mentions per platform (WebMD and Reddit) with two annotators; report precision, recall and F1 against the current UMLS v6 baseline. Both prior GLP-1 papers (AMIA 2025, Nature Health 2026) normalize to MedDRA PT, so the same target is required for comparability.

## 3. Remove TableRAG and the FAISS row index; consolidate on Neo4j

Current state: code_chatbot/chatbot.py runs a TableRAG ReAct loop over pandas backed by FAISS schema and cell indexes (database_table/, code_chatbot/table_loader.py) alongside a separate GraphRAG module. The FAISS indexes are already stale and fingerprint-guarded.

Target: a single Neo4j database. Structured questions (counts, comparisons, time windows, per-platform estimates) run as parameterised Cypher aggregation templates over the evidence graph; evidence snippets are retrieved through a Neo4j vector index. The agent keeps four tools: graph aggregate, evidence search, conflict check between platform-level estimates, and coverage check. Delete table_loader.py, the TableRAG prompts and the FAISS loading paths once the graph tools cover the validated question set in docs/chatbot-validation.md; update code_website/flask_app.py and the tests accordingly.

## Also flagged, not yet decided

- code_website/static/standardized_reviews_all.csv and code_website/data/standardized_reviews_all.csv ship WebMD review full text in a public repository. For any released artifact, publish hashed IDs and derived fields only.
- Reddit ingestion pipeline (Academic Torrents dumps plus Arctic Shift for 2026, self-report classifier, community-type tagging) does not exist yet; sources are listed in docs/reddit-data-sources.md.
