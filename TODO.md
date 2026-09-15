# TODO and decision status

Reorganized on 2026-09-14. Observations and counts below were recorded on 2026-09-12, not remeasured during this documentation task. Earlier wording, with a subsequent editorial update noted, is retained in the [execution archive](docs/history/execution-records.md#original-12).

## Confirmed project decisions

- Use WebMD and Reddit; retain four molecules and eight brands, with exclusions governed by [project scope](docs/project-scope.md).
- Implement a knowledge graph and a graph-based question-answering agent. The specific schema, stack and allocation of work remain open.
- The rejected research-question memo has been deleted. No research questions are currently adopted from that memo.

## Recorded problems and proposals awaiting decisions

The descriptions of existing limitations are recorded findings. The solutions below are retained as proposals, not implementation commitments. The proposed dependency order is standardization → evidence graph → retrieval consolidation, conditional on adopting those designs. No deletion of TableRAG or schema change is authorized merely by this list.

### Evidence-level graph schema — proposed

Recorded problem: src/weightloss/retrieval/graph_loader.py creates only Drug, Condition and SideEffect nodes with TREATS, CAUSES and REVIEWED_FOR edges. Counts, source platform, time, polarity ("no side effects"), formulation and switching are lost at load time, so the graph cannot support cross-platform comparison or a provenance-aware agent.

Proposal: one Evidence node per extracted assertion (hashed source ID, platform, community or page, month, self-report flag, polarity, severity, dose, duration, resolved flag) linked to Molecule, Brand (including an "Unspecified" brand per molecule), Indication, Symptom (MedDRA PT with SOC hierarchy, or UMLS CUI with semantic-type constraints), Outcome, Theme (cost, insurance, shortage, compounding, dose titration) and Source (Platform to Community). Add SWITCHED_FROM / SWITCHED_TO between Evidence and Brand. Materialise per-platform, per-month Brand-Symptom aggregates with numerators and denominators. Keep the dataset fingerprint guard and the empty-database import rule.

### Standardization repair — proposed approach

Recorded problem: the standardized vocabulary in data/processed/migrated-2026-09-14/standardized_reviews_all.csv has 545 terms and contains systematic mapping errors, for example "gas gangrene" (60 occurrences, from "gas"), "irritable bowel syndrome" (70) and "burning sensation" (79). These would be visible in any published graph.

Proposal: retrieval-constrained LLM normalization. Retrieve candidate terms from MedDRA (licensed) or UMLS synonym tables, restrict UMLS candidates to sign/symptom, finding and disease semantic types, let the model choose among candidates or return "no match", and log the candidate list. Validate on 300 annotated side-effect mentions per platform (WebMD and Reddit) with two annotators; report precision, recall and F1 against the current UMLS v6 baseline. Vocabulary choice and licensing remain undecided.

### Neo4j-only retrieval and TableRAG removal — undecided

Recorded problem: src/weightloss/retrieval/chatbot.py runs a TableRAG ReAct loop over pandas backed by FAISS schema and cell indexes (archive/indexes/root/, src/weightloss/retrieval/table_loader.py) alongside a separate GraphRAG module. The FAISS indexes are already stale and fingerprint-guarded.

Proposal: a single Neo4j database. Structured questions (counts, comparisons, time windows, per-platform estimates) run as parameterised Cypher aggregation templates over the evidence graph; evidence snippets are retrieved through a Neo4j vector index. The agent keeps four tools: graph aggregate, evidence search, conflict check between platform-level estimates, and coverage check. If this architecture is selected, remove table_loader.py, the TableRAG prompts and the FAISS loading paths only after the graph tools cover the validated question set in docs/chatbot-validation.md; update apps/web/flask_app.py and the tests accordingly.

## Additional open items

- The tracked research datasets and archive/website/data/standardized_reviews_all.csv retain WebMD review full text; artifacts/web/static/standardized_reviews_all.csv is now a generated, Git-ignored copy. The recorded release proposal is to publish hashed IDs and derived fields only; review the release policy before publication.
- Reddit ingestion pipeline (Academic Torrents dumps plus Arctic Shift for 2026, self-report classifier, community-type tagging) does not exist yet; sources are listed in [Reddit data sources](docs/reddit-data-sources.md).
