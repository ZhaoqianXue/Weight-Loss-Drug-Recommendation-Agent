# Historical execution records

Consolidated on 2026-09-14. This is an archive, not an active task list. The original records below contain intermediate states, proposals and superseded instructions; they must not be executed as current directions. Current scope is in [project scope](../project-scope.md), current implementation and commands in [README](../../README.md), and decision status in [TODO](../../TODO.md).

## Migration plan and preservation policy

The approved migration consolidates ten process files into this record, archives four formal reports, retains five current-facing documents, and updates navigation. Every removed process file and every edited current-facing document has a pre-migration snapshot below. Entry 12 was subsequently edited at user request; all other snapshots remain verbatim. Fenced snapshots retain original paths and links as historical text, not live navigation. Each snapshot has a byte count and SHA-256; entry 12 records its retained edited text. Formal reports retain their original body with only relative Markdown link rebasing and an explicit historical notice. Audit JSON, manifests, code and data are unchanged. The Reddit source document received a subsequent editorial update on 2026-09-14.

## Repository consolidation

The outer working copy became the canonical repository while preserving original Git history. Keep the implementation choices and external recovery location in [consolidation decisions](consolidation.md). The source logs below additionally preserve validation coverage, cache names, whitespace handling, the reported task-budget overrun and intermediate publication status; none is a statement about today's Git status.

## WebMD source audit

The [source audit](webmd-review-audit-2026-09-11.md) preserves page counts, legacy redirects and unknown-versus-zero distinctions. The original audit logs retain directory expansion steps, observed neighboring names and browser limitations. These observations describe the dated audit and do not update source availability.

## Five-generic data refresh

The [refresh report](data-refresh-2026-09-11.md) describes the historical ten-brand snapshot; the [empty-text audit](empty-review-audit-2026-09-12.md) explains its missing narratives. The logs preserve matching details (2,360 candidate matches, 16 rejected, 2,344 reused), the embedded-only Mounjaro record, demographic parsing, filter fixes, and all recorded model/index limitations. Historical review ID 2902 belongs to the excluded Byetta source and contains literal `None`.

## Eight-brand scope change

The [scope document](../project-scope.md) governs four molecules and eight brands. The removed task prompt is retained verbatim for provenance, with two explicit corrections: different CSV schemas need matching ordered IDs and source content, not identical byte hashes; excluded-brand mentions in retained narratives and historical artifacts remain valid. Exact hashes are required only among equivalent copies. The original prompt's broad grep/hash criteria were superseded by these distinctions. Its instruction to leave history untouched applied to that earlier task; this documentation migration was separately approved.

The detailed changed-file list and validation results remain in the pre-migration scope snapshot below. The root logs retain the 50-file preservation check and Victoza pack-alias regression fix.

## File migration map

| Original file | Destination | SHA-256 (original unless marked edited) |
| --- | --- | --- |
| `task_plan.md` | [Original 1](#original-01) | `6111da4f8bcdd6a368e488a5f56c750ab0edc800208f0200b6fb72f018867ab9` |
| `findings.md` | [Original 2](#original-02) | `d72eed8c5f745752dbbc97276f4f9464ce1a73023669276f2eeb32db6c78bcfd` |
| `progress.md` | [Original 3](#original-03) | `10c99c9d934f35607db521ce4973d00126f3cd3e72b5d47ba2394bfadd16ff35` |
| `.planning/five-generic-refresh/task_plan.md` | [Original 4](#original-04) | `34ee29d6d50eb1e5ee801cc51d3a2e2f303f73df46ebd91c85343b7e63eadd52` |
| `.planning/five-generic-refresh/findings.md` | [Original 5](#original-05) | `e4ea4656c7ec8e2d628bbe79ea586c642e69633a00428e19479e395e91f4c640` |
| `.planning/five-generic-refresh/progress.md` | [Original 6](#original-06) | `c6314991ce81de6db5563d6bd8a980ae2337fadfb4ebb36eb8d07032f9c5a7f2` |
| `.planning/webmd-review-audit/task_plan.md` | [Original 7](#original-07) | `d852e72fcac1ca65a8d0bc729268aca1c8754b487df316ab2d77a6738a49308c` |
| `.planning/webmd-review-audit/findings.md` | [Original 8](#original-08) | `ee4edc2d90d434bc6cad4e2021a529720f94d4ffa351b8e5d3b7a3609c4e068c` |
| `.planning/webmd-review-audit/progress.md` | [Original 9](#original-09) | `16a51af783b0b1e203fb32fb8e0cd5a14de65d0f1918255cf0bfd0c76df701ac` |
| `.planning/codex-prompt-8-brand-scope.md` | [Original 10](#original-10) | `b5916a23a9c3cb606a54d0fd872ba52ed3e178fd763638e44745644d5e208f71` |
| `README.md` | [Original 11](#original-11) | `bcfdd283cf0b6b42cc09d6c38ffb889b20df6e13ad0d0031f2e88164d97fd644` |
| `TODO.md` | [Edited snapshot 12](#original-12) | `f76a39dc91a42fc69bd0fe54512dbd4a04bf172ea70d51fdbf31543a2f7e1c8a` |
| `docs/project-scope.md` | [Original 13](#original-13) | `9fe36023289b2832850458da943222a0405a691267905feacff8278987173c7b` |
| `docs/chatbot-validation.md` | [Original 14](#original-14) | `296658e8c40d605d8e8110971cc1167d1adedad662466b1b1777917a171cc0b6` |
| `docs/consolidation.md` | [consolidation.md](consolidation.md) (notice + rebased links) | `120cb01a78a782b58f900b6487c2ee995535d9891f3821b0c5786bcba9199f64` |
| `docs/webmd-review-audit-2026-09-11.md` | [webmd-review-audit-2026-09-11.md](webmd-review-audit-2026-09-11.md) (notice + rebased links) | `c017032bf87e0a7fcc946b8717c143375691c9ad176a2b8fed8b32bc8ccc4603` |
| `docs/data-refresh-2026-09-11.md` | [data-refresh-2026-09-11.md](data-refresh-2026-09-11.md) (notice + rebased links) | `d98f50e7053636f2f31f73de80595895ef6c6015fe64a2c494a83c5e81386120` |
| `docs/empty-review-audit-2026-09-12.md` | [empty-review-audit-2026-09-12.md](empty-review-audit-2026-09-12.md) (notice + rebased links) | `782be01e892e5bfb2c798d3c859c9d9fc17c14296c0ca0dd460854068ee393f2` |
| `docs/reddit-data-sources.md` | [Edited 2026-09-14](../reddit-data-sources.md) | `d7f8c6bda5c65bbf7dc0fd1bc22b75e0458df2fa9044597df47a597292f4d90a` |

## Archived records

The first ten entries replace the removed process files. The final four preserve current-facing documents before their editorial reorganization. Original headings, instructions, URLs and intermediate numbers are preserved inside the code fences, except for the explicitly marked editorial update to entry 12.

<a id="original-01"></a>

### Original 01: task_plan.md

Original bytes: 1813; SHA-256: `6111da4f8bcdd6a368e488a5f56c750ab0edc800208f0200b6fb72f018867ab9`.

<!-- snapshot-start:task_plan.md -->
```text
# Repository consolidation

## Objective
Use the outer working directory as the canonical repository, preserve the nested repository history, remove slide artifacts, and make maintained project content English-only.

## Phases
1. Inventory differences, Chinese content, credentials, and dependencies (complete).
2. Back up conflicting files outside the project, consolidate Git metadata, and resolve differences (complete).
3. Translate Chinese content, remove slides, and document the canonical layout (complete).
4. Validate syntax, data preservation, Git integrity, and English-only content (complete).

## Decisions
Preserve existing Git history without rewriting historical blobs. Publish the consolidated version to GitHub as subsequently authorized by the user. Preserve outer extraction behavior and document its difference from the historical implementation unless evidence requires another choice.

## Validation notes
The initial diff check found two trailing-whitespace lines carried into the integration. Both were corrected. The configured 4,000-token task budget was exceeded and disclosed during execution.

## Publication
The remote main branch has four subsequent commits deleting .DS_Store files. Incorporate them before a normal push, without force-pushing.

## Scope update — 2026-09-12
1. Inspect catalog, raw and derived schema, tests and docs (complete).
2. Archive the two excluded brand files and rebuild the eight-brand snapshot offline (complete).
3. Synchronize active code, tests and scope documentation (complete).
4. Run Python, JavaScript, data integrity and preservation checks; record results (complete; literal grep/hash exceptions documented in project scope).
No scraping, model calls, index/database changes, or commits. Existing historical planning files stay unchanged.
```
<!-- snapshot-end:task_plan.md -->

<a id="original-02"></a>

### Original 02: findings.md

Original bytes: 1787; SHA-256: `d72eed8c5f745752dbbc97276f4f9464ce1a73023669276f2eeb32db6c78bcfd`.

<!-- snapshot-start:findings.md -->
```text
# Findings

The outer directory contains later website and UMLS work. The nested repository retains Git history. The outer extraction schema intentionally assigns side effects to the primary review drug, while the historical version supports associated_drug. Outer prescribing-information scripts were deleted from the historical repository but remain relevant experimental scripts.

Reviewed all shared code files: nine identical files and four differing files (three Python files and one notebook). Outer source takes precedence. Historical-only README preserved before replacement. Chinese content appears in evaluation output, chart labels, and comments. No Chinese found in decoded notebook source during initial scan. Oversized caches are biobert_embeddings_v5.json and embeddings_cache_v6.pkl. Backup stored outside the repository.

## Scope update — 2026-09-12
The current working tree has pre-existing modifications. Raw, extracted and standardized CSV schemas differ; only the three standardized copies should share byte hashes, while all five must share ordered review IDs. Some retained source narratives and historical data mention excluded medications; preserve their content pending user clarification. Collector has no offline rebuild mode; rebuild from local per-brand CSVs with its existing writer and preserve collection timestamps.

Final integrity: all five current datasets have 2,727 ordered review IDs and matching raw source content; three standardized copies have identical byte hashes. Catalog and dataset-manifest pairs are identical. Thirty-eight retained narratives mention the excluded brands; active catalog, code, tests and primary medication fields are clear. Historical datasets and narratives were not rewritten to satisfy unrestricted text search.
```
<!-- snapshot-end:findings.md -->

<a id="original-03"></a>

### Original 03: progress.md

Original bytes: 2163; SHA-256: `10c99c9d934f35607db521ce4973d00126f3cd3e72b5d47ba2394bfadd16ff35`.

<!-- snapshot-start:progress.md -->
```text
# Progress

Inventory started. No source files moved or deleted yet.

Created external backup, moved original .git to the root, removed the nested working tree, removed slide artifacts and generated bytecode. Translating maintained code next.

Verification: Git fsck passed. All Python files parse. Both notebook outputs are empty. Twelve PDF text layers and 82 text files plus five pickle string streams have no Chinese. All 60 data hashes are unchanged except the intentionally regenerated English chart. Static preview HTML and CSV requests passed. Slide preview server stopped. Imported raw data whitespace is preserved through Git attributes; source whitespace was normalized only in newly integrated files. An initial warning-summary command hit a Unicode decoding error on an unclassified binary index; explicit binary Git attributes address this.

The user authorized committing and pushing the consolidated version. Fetched origin/main and identified four remote cleanup commits to retain.

## Scope update — 2026-09-12
Read the eight required files before edits. Saved targeted pre-edit files and historical/per-brand hashes outside the repository. Historical planning directories and existing backups remain untouched.

Offline snapshot rebuilt: 2,727 records; 2,344 reused annotations, 383 pending (381 with text, 2 rating-only). Python: 15 tests passed. Initial JS run exposed unsupported existing Victoza pack aliases; both intent matchers now accept the catalog aliases, with direct and comparison regression coverage. Data validation now also checks retained raw fields, catalog copies and manifest copies.

Final validation: 15 Python tests, 3,963 JavaScript assertions, 8 brands and 28 pairs passed. Data validator, Python syntax parsing, JS syntax checks and git diff --check passed. Checked the 50-file preservation baseline (allowing the rebuilt combined raw CSV and verifying moved files at their archive destinations); protected history, eight per-brand raw files and Reddit source documentation are unchanged. Original dated refresh report is unchanged below its addendum. No scraping, model API calls, database/index updates or commit.
```
<!-- snapshot-end:progress.md -->

<a id="original-04"></a>

### Original 04: .planning/five-generic-refresh/task_plan.md

Original bytes: 861; SHA-256: `34ee29d6d50eb1e5ee801cc51d3a2e2f303f73df46ebd91c85343b7e63eadd52`.

<!-- snapshot-start:.planning/five-generic-refresh/task_plan.md -->
```text
# Five-generic refresh

1. Inspect pipeline and preserve historical canonical CSVs — completed.
2. Implement catalog and validated collector; collect ten brands — completed: 3,318 IDs, 172 pages.
3. Refresh derived data, website and backend code — completed for available data; 2,344 annotations reused and 974 explicitly pending (933 with text). New model calls require an OPENAI_API_KEY configured outside chat. External FAISS/Neo4j rebuilds were not run; stale indexes are rejected.
4. Validate collection, naming, UI, assistant and reproducible documentation — completed: 15 Python tests, 4,565 JS assertions, independent CSV integrity audit and five-generic browser checks.

No model credentials were supplied in response to the asynchronous environment-path question. Do not claim all model annotation or external backend integration is complete.
```
<!-- snapshot-end:.planning/five-generic-refresh/task_plan.md -->

<a id="original-05"></a>

### Original 05: .planning/five-generic-refresh/findings.md

Original bytes: 2272; SHA-256: `e4ea4656c7ec8e2d628bbe79ea586c642e69633a00428e19479e395e91f4c640`.

<!-- snapshot-start:.planning/five-generic-refresh/findings.md -->
```text
# Findings

- User scope: Semaglutide, Tirzepatide, Liraglutide, Dulaglutide, Exenatide; ten brands. Canonical Wegovy and Bydureon only; aliases do not create rows.
- Collected 3,318 unique review IDs, including 3,233 narratives / 85 rating-only records; all ten headline totals match.
- Mounjaro has 661 rendered reviews plus one embedded-only record. Source Visibility preserves this distinction.
- Demographic enum labels in embedded JSON conflict with displayed labels. Parse visible demographics, including 75 or over; retain masked email display names.
- Historical text expansion joined words across spans. Reuse requires unique brand/date/author/full non-whitespace text match. Of 2,360 matching annotations, 16 were rejected for attribution to a different medication, leaving 2,344 reused.
- 974 pending records include 933 with text and 41 without text. No model API key is configured; no new model calls occurred.
- Default dependent filters originally hid new drugs lacking side-effect annotations. Medication catalog filters now remain available; reviewed_for edges represent source condition metadata.
- Raw collection, current canonical datasets, both website CSVs, manifests, collector/extraction/standardization/frontend/backend code, docs and tests are updated. Historical experiments remain labeled historical.

## 2026-09-12 follow-up clarification

- User explicitly excludes all combination products, including Soliqua and Xultophy regardless of review counts. Recorded in docs/project-scope.md and README.
- Archived raw/extracted/standardized datasets each contain 44 blank text records (42 Victoza, 1 Ozempic, 1 Saxenda), with side_effects=[] in all historical outputs. Old code retained empty reviews, default pandas parsing produced NaN, and extraction did not skip missing text. No API logs were available to reconstruct individual historical calls.
- Current 85 blank records = 44 historical reused + 41 pending (34 Byetta, 5 Bydureon, 2 Trulicity). All 41 pending blank IDs were rechecked live across 25 pages; source text is empty and ratings are present.
- 933 pending nonempty fields include literal None in Byetta review ID 2902. Nonempty counts are not substantive narrative counts. Documented without changing dataset statuses or code.
```
<!-- snapshot-end:.planning/five-generic-refresh/findings.md -->

<a id="original-06"></a>

### Original 06: .planning/five-generic-refresh/progress.md

Original bytes: 586; SHA-256: `c6314991ce81de6db5563d6bd8a980ae2337fadfb4ebb36eb8d07032f9c5a7f2`.

<!-- snapshot-start:.planning/five-generic-refresh/progress.md -->
```text
# Progress

Completed local raw/data/code/documentation refresh. Validated 15 Python tests, 4,565 JavaScript assertions across 45 pairs, all five generic filters in the browser, source counts/IDs and matching CSV hashes. Model runners fail clearly without credentials and preserve files. No commits, remote publication or external graph database writes were performed.

Remaining: new model extraction and standardization require a configured environment/key; FAISS rebuild and fresh Neo4j import require the research runtime/services. Detailed report: docs/data-refresh-2026-09-11.md.
```
<!-- snapshot-end:.planning/five-generic-refresh/progress.md -->

<a id="original-07"></a>

### Original 07: .planning/webmd-review-audit/task_plan.md

Original bytes: 379; SHA-256: `d852e72fcac1ca65a8d0bc729268aca1c8754b487df316ab2d77a6738a49308c`.

<!-- snapshot-start:.planning/webmd-review-audit/task_plan.md -->
```text
# WebMD review inventory audit

1. Discover eight ingredients, brands and legacy package pages (complete).
2. Verify available counts and redirects with live browser (complete).
3. Reconcile canonical pages, preserve unknowns, deliver sourced report (complete).

Limit: historical unlisted brand pages could not be confirmed; report does not claim worldwide exhaustive coverage.
```
<!-- snapshot-end:.planning/webmd-review-audit/task_plan.md -->

<a id="original-08"></a>

### Original 08: .planning/webmd-review-audit/findings.md

Original bytes: 2309; SHA-256: `ee4edc2d90d434bc6cad4e2021a529720f94d4ffa351b8e5d3b7a3609c4e068c`.

<!-- snapshot-start:.planning/webmd-review-audit/findings.md -->
```text
# Findings

Previous answer: ten nonzero main review pages total 3318; Foundayo zero verified via live browser. Possible omissions: generic pages and package-specific records; discontinued brands not fully checked.

Live browser directory: Ad prefix lists Adlarity then Adquey, no Adlyxin; Al prefix lists Albendazole, Albizia, Albuterol, no albiglutide. Directory lists first 20 and View More, but absent terms fall within observed alphabetical span.

WebMDRx exposes legacy entries: Victoza 2 Pak ID 163054 (51), exenatide ID 93223 (4), semaglutide ID 178018 (4), Ozempic 1 Mg Dose ID 181644 (4). These counts may be stale or merged; follow live review links before adding.

Live redirect verification: Victoza 2-Pak (163054) -> liraglutide-victoza, 440. Exenatide (93223) -> exenatide-byetta, 421. Semaglutide (178018) -> rybelsus-semaglutide, 81. Legacy counts must not be added. Web search cannot fetch these redirects; live browser succeeds.

Unexpected live redirect: Ozempic 1 Mg Dose legacy ID 181644 -> wegovy-semaglutide, 491 (do not interpret as product equivalence; preserve original name and link). Bydureon ID 159405 -> bydureon-bcise-exenatide, 170.

Live main counts: Ozempic 512 (condition subtotals sum 512). Mounjaro 662, while displayed condition subtotals 333+328=661; preserve page total and flag 1-record classification discrepancy.

Live totals: Zepbound 48, Saxenda 222. Both condition subtotals reconcile.

Live Trulicity 271; live Foundayo explicitly displays Be the first to share your experience with this treatment (0 reviews). Found additional combination products: Soliqua 6 from WebMD search; Xultophy requires live zero verification.

Fully expanded live L/li directory has Liraglutide (Victoza), no Lixisenatide. Full L/ly directory has no Lyxumia. Keep unavailable counts as null, not zero.

Fully expanded live T/ta directory: no Tanzeum (Tansy Ragwort followed by Tapioca).

Live E/ep directory complete: no Eperzan (Epclusa followed by Ephedra). All four previously unconfirmed brands absent from corresponding current directories.

Live Soliqua 6; live Xultophy explicitly zero. Main single-ingredient-product page sum 3318; combination-inclusive known page sum 3324. These are displayed rating/review records, not unique patients or guaranteed nonempty text reviews.
```
<!-- snapshot-end:.planning/webmd-review-audit/findings.md -->

<a id="original-09"></a>

### Original 09: .planning/webmd-review-audit/progress.md

Original bytes: 543; SHA-256: `16a51af783b0b1e203fb32fb8e0cd5a14de65d0f1918255cf0bfd0c76df701ac`.

<!-- snapshot-start:.planning/webmd-review-audit/progress.md -->
```text
# Progress

Started full inventory audit. No dataset changes authorized or made.

Directory letter click did not navigate (remained /a/al). Switching to observed href direct navigation.

L/li directory initially limited to 20 entries, ending at Lincomycin; must expand View More before absence assessment.

Completed 13 unique live review pages, five legacy redirects, and four unresolved brand directory checks. Report and JSON inventory saved in docs/. Validated unique page URLs and both arithmetic totals. No source dataset modifications.
```
<!-- snapshot-end:.planning/webmd-review-audit/progress.md -->

<a id="original-10"></a>

### Original 10: .planning/codex-prompt-8-brand-scope.md

Original bytes: 4777; SHA-256: `b5916a23a9c3cb606a54d0fd872ba52ed3e178fd763638e44745644d5e208f71`.

<!-- snapshot-start:.planning/codex-prompt-8-brand-scope.md -->
```text
# Codex task: reduce project scope to 8 brands (remove exenatide)

Repository: this directory is the canonical repo. Read README.md, docs/project-scope.md, drug_catalog.py, config/drugs.json, code_pipeline/refresh_data.py, code_pipeline/validate_data.py, tests/test_refresh.py and tests/review-assistant.test.js before editing anything.

## Decision to implement

On 2026-09-12 the project dropped the exenatide brands Byetta and Bydureon (alias "Bydureon BCise"). Scope is now 4 generic names and 8 brands: Semaglutide (Wegovy, Ozempic, Rybelsus), Tirzepatide (Mounjaro, Zepbound), Liraglutide (Victoza, Saxenda), Dulaglutide (Trulicity). Rationale to record: no Reddit community exists for either brand (no subreddit with 200+ subscribers; about 70-95 posts and about 280 comments per brand across 18 diabetes/GLP-1 subreddits, 2018-2026) and AstraZeneca discontinued both in the US on 2024-10-25 and 2024-10-28. The per-brand Reddit sources for the remaining 8 brands are already documented in docs/reddit-data-sources.md; do not rewrite that file, only link to it.

## Steps

1. Catalog: remove the two Exenatide entries from config/drugs.json. drug_catalog.py derives DRUGS and ALIASES from it; code_website/static/drugs.json is a copy produced by refresh_data.py, keep them identical.
2. Raw data: move data_webmd/webmd_byetta_reviews.csv and data_webmd/webmd_bydureon_reviews.csv to data_backup/excluded_exenatide_2026-09-12/ (move, do not delete). Rebuild data_webmd/webmd_all_reviews.csv and data_webmd/collection_manifest.json from the eight remaining per-brand CSVs, or filter the existing combined file if the collector cannot rebuild without a network run. Expected total: 2,727 records (491+512+81+662+48+440+222+271). Do not re-scrape and do not call any model API.
3. Derived data: run code_pipeline/refresh_data.py so that data_extracted/extracted_reviews_all.csv, data_standardized/standardized_reviews_all.csv, code_website/data/standardized_reviews_all.csv, code_website/static/standardized_reviews_all.csv, data_standardized/dataset_manifest.json and code_website/static/dataset_manifest.json all contain exactly the same 2,727 records with no Byetta or Bydureon rows and identical SHA-256 hashes across copies. Recompute pending/reused annotation counts from the data; do not hand-edit numbers.
4. Code: remove Byetta/Bydureon from the hard-coded drug lists and alias notes in code_chatbot/chatbot.py (around lines 907, 1325, 1494), the Bydureon example in the extraction prompt in code_extraction/schema_extraction.py (line 99, keep the Wegovy example), the Bydureon limitation string in code_pipeline/refresh_data.py, the color map entries in code_website/templates/knowledge_graph.html (line 1137), and any brand list in code_website/static/review-assistant.js. Leave FAISS, Neo4j and model code paths otherwise untouched.
5. Tests: update tests/test_refresh.py (10 brands/5 generics becomes 8/4; replace the 'bydureon bcise' alias assertion with 'victoza 2-pak' or 'wegovy hd'; replace 'Byetta' in the conservative-matching fixture with an out-of-scope name) and tests/review-assistant.test.js (the ['Trulicity','Byetta','Bydureon'] loop and the 'Bydureon BCise' alias check). Expect 8 brands and 28 pairs.
6. Docs: in docs/project-scope.md change the tables to 4 generics/8 brands and add a section "Explicit exclusion: exenatide products" with the decision date, the rationale above, the archive location of the WebMD records, and a link to docs/reddit-data-sources.md. Update README.md counts and brand lists (2,727 records, 4 generics, 8 brands, recomputed pending counts) and link docs/reddit-data-sources.md. Update the counts in docs/chatbot-validation.md after the tests run. Add a short dated addendum at the top of docs/data-refresh-2026-09-11.md pointing to the scope change instead of rewriting that report. Leave .planning/*, docs/webmd-review-audit-*, docs/empty-review-audit-*, data_standardized/cui_mappings_*.json, data_standardized/standardization_report_*.json and data_backup/ untouched as historical records.

## Acceptance

- `python3 -m unittest discover -s tests -p 'test_*.py'` and `node tests/review-assistant.test.js` pass; the JS summary reports brands=8.
- `python3 code_pipeline/validate_data.py` passes.
- All five canonical CSV copies have 2,727 records and identical hashes; `grep -ri "byetta\|bydureon" --include=*.csv data_webmd data_extracted data_standardized code_website` returns nothing.
- `grep -ril "byetta\|bydureon\|exenatide" --exclude-dir=.git --exclude-dir=data_backup .` returns only the historical documents listed above plus the new exclusion note and docs/reddit-data-sources.md.
- Write a short change summary at the end of docs/project-scope.md (files touched, before/after counts). Do not commit.
```
<!-- snapshot-end:.planning/codex-prompt-8-brand-scope.md -->

<a id="original-11"></a>

### Original 11: README.md

Original bytes: 7364; SHA-256: `bcfdd283cf0b6b42cc09d6c38ffb889b20df6e13ad0d0031f2e88164d97fd644`.

<!-- snapshot-start:README.md -->
````text
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

The current raw, extraction, baseline standardized, and website datasets each contain **2,727 review records**, covering **4 generic names and 8 brands**. Of these, 2,681 contain review text. Historical annotations were reused for 2,344 matching records; **383 records carry the pending status: 381 have a nonempty text field, while 2 are rating-only records with no text to extract**. The reused historical outputs also include 44 empty-text records; see [the historical empty-text audit](empty-review-audit-2026-09-12.md). Pending annotations are unknown, not evidence of no side effects. See [the current scope and counts](../project-scope.md), [the dataset manifest](../../data/processed/migrated-2026-09-14/dataset_manifest.json) for per-brand coverage, and [the historical refresh report](data-refresh-2026-09-11.md) for provenance and outstanding model/index work.

Wegovy HD is an alias of **Wegovy**; Victoza 2-Pak and Victoza 3-Pak are aliases of **Victoza**, not extra datasets. Generic names are represented in `Drug Name`; brands are represented in `Brand Name`. Shared generic/brand review pages are collected once. The maintained catalog is [config/drugs.json](../../configs/drugs.json). **Combination products are excluded regardless of review availability**, including Soliqua (insulin glargine + lixisenatide) and Xultophy (insulin degludec + liraglutide). See [the project scope](../project-scope.md) for exclusions and [Reddit data sources](../reddit-data-sources.md) for the eight retained brands.

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
````
<!-- snapshot-end:README.md -->

<a id="original-12"></a>

### Original 12: TODO.md

Edited on 2026-09-14 at user request; this entry is no longer a verbatim original. Metadata describes the retained text.

Retained bytes: 3477; SHA-256: `f76a39dc91a42fc69bd0fe54512dbd4a04bf172ea70d51fdbf31543a2f7e1c8a`.

<!-- snapshot-start:TODO.md -->
```text
# TODO

Recorded 2026-09-12 while planning the WWW 2027 submission (WebMD + Reddit, 4 generics / 8 brands). Items are ordered by dependency: 2 blocks 1, and 1 blocks 3.

## 1. Rebuild the knowledge graph schema as an evidence-level, provenance-aware graph

Current state: code_chatbot/graph_loader.py creates only Drug, Condition and SideEffect nodes with TREATS, CAUSES and REVIEWED_FOR edges. Counts, source platform, time, polarity ("no side effects"), formulation and switching are lost at load time, so the graph cannot support cross-platform comparison or a provenance-aware agent.

Target: one Evidence node per extracted assertion (hashed source ID, platform, community or page, month, self-report flag, polarity, severity, dose, duration, resolved flag) linked to Molecule, Brand (including an "Unspecified" brand per molecule), Indication, Symptom (MedDRA PT with SOC hierarchy, or UMLS CUI with semantic-type constraints), Outcome, Theme (cost, insurance, shortage, compounding, dose titration) and Source (Platform to Community). Add SWITCHED_FROM / SWITCHED_TO between Evidence and Brand. Materialise per-platform, per-month Brand-Symptom aggregates with numerators and denominators. Keep the dataset fingerprint guard and the empty-database import rule.

## 2. Rebuild the standardization layer

Current state: the standardized vocabulary in data_standardized/standardized_reviews_all.csv has 545 terms and contains systematic mapping errors, for example "gas gangrene" (60 occurrences, from "gas"), "irritable bowel syndrome" (70) and "burning sensation" (79). These would be visible in any published graph.

Target: retrieval-constrained LLM normalization. Retrieve candidate terms from MedDRA (licensed) or UMLS synonym tables, restrict UMLS candidates to sign/symptom, finding and disease semantic types, let the model choose among candidates or return "no match", and log the candidate list. Validate on 300 annotated side-effect mentions per platform (WebMD and Reddit) with two annotators; report precision, recall and F1 against the current UMLS v6 baseline.

## 3. Remove TableRAG and the FAISS row index; consolidate on Neo4j

Current state: code_chatbot/chatbot.py runs a TableRAG ReAct loop over pandas backed by FAISS schema and cell indexes (database_table/, code_chatbot/table_loader.py) alongside a separate GraphRAG module. The FAISS indexes are already stale and fingerprint-guarded.

Target: a single Neo4j database. Structured questions (counts, comparisons, time windows, per-platform estimates) run as parameterised Cypher aggregation templates over the evidence graph; evidence snippets are retrieved through a Neo4j vector index. The agent keeps four tools: graph aggregate, evidence search, conflict check between platform-level estimates, and coverage check. Delete table_loader.py, the TableRAG prompts and the FAISS loading paths once the graph tools cover the validated question set in docs/chatbot-validation.md; update code_website/flask_app.py and the tests accordingly.

## Also flagged, not yet decided

- code_website/static/standardized_reviews_all.csv and code_website/data/standardized_reviews_all.csv ship WebMD review full text in a public repository. For any released artifact, publish hashed IDs and derived fields only.
- Reddit ingestion pipeline (Academic Torrents dumps plus Arctic Shift for 2026, self-report classifier, community-type tagging) does not exist yet; sources are listed in docs/reddit-data-sources.md.
```
<!-- snapshot-end:TODO.md -->

<a id="original-13"></a>

### Original 13: docs/project-scope.md

Original bytes: 5623; SHA-256: `9fe36023289b2832850458da943222a0405a691267905feacff8278987173c7b`.

<!-- snapshot-start:docs/project-scope.md -->
```text
# Project medication scope

## Included medications

The maintained catalog is [`config/drugs.json`](../../configs/drugs.json). The current project includes these four generic names and their eight brands:

| Generic name | Canonical brands |
| --- | --- |
| Semaglutide | Wegovy, Ozempic, Rybelsus |
| Tirzepatide | Mounjaro, Zepbound |
| Liraglutide | Victoza, Saxenda |
| Dulaglutide | Trulicity |

Wegovy HD is an alias of Wegovy; Victoza 2-Pak and Victoza 3-Pak are aliases of Victoza. Aliases do not create additional datasets. The per-brand Reddit sources are documented in [Reddit data sources](../reddit-data-sources.md).

## Explicit exclusion: combination products

User-confirmed on 2026-09-12: **combination products are excluded from this project, regardless of whether WebMD has reviews.** In particular:

| Combination generic name | Brand | Project decision |
| --- | --- | --- |
| Insulin glargine + lixisenatide | [Soliqua](https://reviews.webmd.com/drugs/drugreview-soliqua-insulin-glargine-lixisenatide) | Excluded |
| Insulin degludec + liraglutide | [Xultophy](https://reviews.webmd.com/drugs/drugreview-xultophy-insulin-degludec-liraglutide) | Excluded |

These products must not enter collection targets, review totals, extracted datasets, model indexes, or website medication lists. Their constituent GLP-1 drugs do not make the combination products equivalent to the included single-ingredient brands. Historical screening documents may mention these products to document exclusion; such mentions do not imply inclusion.

## Explicit exclusion: exenatide products

Decision date: **2026-09-12**. **Byetta and Bydureon (alias Bydureon BCise), both exenatide products, are excluded from the current project.** The recorded scope decision rests on two considerations:

- Reddit coverage is insufficient for the intended brand-level analysis: neither brand has a subreddit with 200 or more subscribers. Across 18 diabetes/GLP-1 subreddits during 2018–2026, there are about 70–95 posts and about 280 comments per brand (approximately 70 posts/281 comments for Byetta and 95 posts/285 comments for Bydureon).
- AstraZeneca discontinued Byetta in the United States on **2024-10-25** and Bydureon BCise on **2024-10-28**.

These findings were supplied with the scope decision; no new Reddit collection or source investigation was performed in this update. The remaining eight brands' sources and the recorded exclusion evidence are in [Reddit data sources](../reddit-data-sources.md); that document is retained unchanged.

The **421 Byetta reviews and 170 Bydureon reviews (591 total)** were moved, not deleted, to [`data_backup/excluded_exenatide_2026-09-12/`](../../archive/data_backup/excluded_exenatide_2026-09-12/). The eight retained per-brand WebMD files are unchanged. The combined raw dataset and its collection manifest were rebuilt offline, retaining original collection timestamps; extracted and standardized data and website copies were then regenerated without scraping or model API calls.

Scope exclusion applies to the primary `Brand Name` and `Drug Name` fields, catalog entries, and active medication lists. It does not erase mentions of previous or alternative treatments from source narratives: 38 retained-brand reviews mention Byetta or Bydureon. Historical UMLS datasets, mapping/cache files, audits, planning records and existing backups are also preserved. Consequently, an unrestricted text search can still find these names in source narratives and historical artifacts.

Raw, extracted and standardized CSVs have different schemas (and standardized dates are normalized). All five canonical CSVs share the same ordered review IDs and source content; the three standardized copies have identical SHA-256 hashes. Raw and extracted byte hashes differ by design. Catalog copies and dataset-manifest copies are byte-identical within each pair.

## Change summary — 2026-09-12

| Measure | Before | After |
| --- | ---: | ---: |
| Generic names | 5 | 4 |
| Brands | 10 | 8 |
| Brand pairs | 45 | 28 |
| Current review records | 3,318 | 2,727 |
| Reviews with text | 3,233 | 2,681 |
| Reused historical annotations | 2,344 | 2,344 |
| Pending records | 974 | 383 |
| Pending with text | 933 | 381 |
| Pending without text | 41 | 2 |

Files touched:

- Catalog: `config/drugs.json` and its generated copy `code_website/static/drugs.json`; `drug_catalog.py` continues to derive brands and aliases without modification.
- Raw data: the two archived per-brand files, `data_webmd/webmd_all_reviews.csv`, and `data_webmd/collection_manifest.json`.
- Derived data: `data_extracted/extracted_reviews_all.csv`, `data_standardized/standardized_reviews_all.csv`, both website CSV copies, and both dataset manifests.
- Active code: `code_chatbot/chatbot.py`, `code_extraction/schema_extraction.py`, `code_pipeline/refresh_data.py`, `code_pipeline/validate_data.py`, `code_website/templates/knowledge_graph.html`, and `code_website/static/review-assistant.js`. The existing Victoza pack aliases now work in direct and comparison questions. Model, FAISS and Neo4j paths otherwise remain unchanged.
- Tests: `tests/test_refresh.py` and `tests/review-assistant.test.js`.
- Documentation: `README.md`, this scope document, `docs/chatbot-validation.md`, and a dated addendum to `docs/data-refresh-2026-09-11.md`; root `task_plan.md`, `findings.md`, and `progress.md` record execution. Historical `.planning/` records and `docs/reddit-data-sources.md` are unchanged.

Validation: 15 Python tests passed; 3,963 JavaScript assertions passed across 8 brands and 28 pairs; the data validator passed. No commit was made.
```
<!-- snapshot-end:docs/project-scope.md -->

<a id="original-14"></a>

### Original 14: docs/chatbot-validation.md

Original bytes: 6973; SHA-256: `296658e8c40d605d8e8110971cc1167d1adedad662466b1b1777917a171cc0b6`.

<!-- snapshot-start:docs/chatbot-validation.md -->
```text
# Medication Experience Assistant validation

Current automated validation: 2026-09-12; original interface validation: 2026-09-09. Status: implemented and tested dataset-backed interactive assistant; no conference acceptance or independent human answer-quality certification is claimed.

## Implemented scope

- Medication summaries with review counts and effectiveness, ease-of-use, and satisfaction means on the 1–5 scale, each with its observed denominator.
- Two-medication comparisons, rating-only and side-effect-only views, and multi-turn follow-ups that retain medication and symptom context.
- Reported side-effect exploration across all eight medications or the current selection. Chinese symptom aliases and common English follow-ups are supported; arbitrary natural-language interpretation is not promised.
- Evidence explanations covering provenance, graph weights, sample-size imbalance, association versus causation, and unsupported questions.
- Original review excerpts with expandable full text, data-row IDs, brand, date, source CSV link, and pagination. Comparison evidence interleaves medications while retaining CSV order within each medication. It is not a representative sample or a generated summary.
- Graph actions for one medication, a pair, or a symptom; reset restores all drugs and the 1% top-side-effect setting. A graph scope badge makes chat-driven filtering visible. Statistics always use the full dataset, independent of graph filters.
- Loading/error messaging, safe text rendering, keyboard-accessible choices, a live conversation region, and internally scrollable tables on narrow screens.

## Source and counting rules

The current website and assistant use `code_website/static/standardized_reviews_all.csv`, **2,727 rows covering eight brands and four generic names**. SHA-256: `e4bb15982d884905efb723b9c0d68be6dc60e1e4ea56bfb4ebf0bb69607840d8`. The September refresh reused 2,344 historical annotations and explicitly marks 383 rows pending (381 with text and 2 rating-only). See [the current project scope](../project-scope.md) and machine-readable manifests for the current snapshot; [the original refresh report](data-refresh-2026-09-11.md) is historical.

Review IDs R0001 onward refer to original data-row order, excluding the CSV header. They are stable for this snapshot, not globally stable identifiers. 46 records lack review text; they remain in statistical denominators but are excluded from readable evidence. Of these, 44 have reused historical empty-side-effect outputs and 2 carry the pending status. The historical reuse count therefore is not a count of analyzed narratives; see [the historical empty-text audit](empty-review-audit-2026-09-12.md). WebMD source IDs are additionally preserved in the CSV Review ID column. Structured side-effect terms are trimmed and case-folded; each term is counted once per review. Synonyms are not automatically merged. Percentages divide by all records for the medication. Invalid extraction is treated as unknown, not evidence that no symptom occurred; coverage is displayed. Ratings use finite numeric values in [1, 5], excluding blank or invalid values.

Graph relation weights count extracted relation occurrences. Chat counts deduplicate terms within reviews. These measures can differ and are explicitly explained. Extraction outputs are not clinically adjudicated; the tests verify calculations against this dataset, not the medical correctness of the extraction.

## Automated validation

Command: `node tests/review-assistant.test.js`.

Current result: **3,963 checks passed** over all eight medications, all 28 medication pairs, and the 2,727-record source. All four generic mappings, retained brands and canonical aliases are covered. Python independently calculates expected group counts, mean ratings, and review-level side-effect counts. Node tests exercise the shipped JavaScript against those expectations, verify every available source review against its original row, and cover pagination, context retention, English and Chinese follow-ups, unsupported questions, sample-size explanations, graph action payloads, duplicate terms, malformed/missing extraction, empty ratings, out-of-range ratings, no matches, and empty datasets.

These checks are assertions, not 3,963 independently annotated questions or human ratings. JavaScript syntax checks and `git diff --check` also passed.

## Browser validation and iterations

The 2026-09-12 automated scope checks cover four generic names and eight brands, all 28 brand pairs, generic-name questions, and explicit unknown results when no side effects have been extracted. Browser and mobile checks were not rerun for this scope update. The following preserves the earlier interface validation.

Checked all four entry flows, Wegovy summary, Wegovy–Ozempic comparison, nausea follow-up, seven-medication symptom table, source expansion, pagination, evidence explanation, and graph filtering. A nausea-focused Wegovy–Ozempic action displayed exactly the two medications plus nausea. Desktop and 390-pixel-wide layouts were checked. The chat panel had no horizontal overflow; narrow tables scroll internally. Browser error logs were empty in the tested sessions.

Iterations corrected: the evidence test initially assumed every record had text; independent data inspection identified the missing record and the test now uses actual text availability. The intent matcher initially matched “prove” inside “approve”; word boundaries corrected this and a regression case was added. Comparison evidence initially surfaced one medication first; it now alternates medication groups. Seven-medication tables were transposed for readability. A inherited full-width button style squeezed the input; explicit flex sizing corrected it. Long-answer scroll positioning now shows the start of the result, and quantitative results precede interpretive notes.

## Relationship to WWW2027

The [official WWW2027 Demo call](https://www2027.thewebconf.org/demos/) requires an implemented and tested system and describes review criteria of originality, significance, quality, and clarity. It does not define a per-answer pass score. This work provides a functioning, inspectable demonstration and repeatable tests. It does not establish research novelty, clinical validity, unbiased comparative inference, broad natural-language coverage, user-study effectiveness, or acceptance-level quality. A paper-level evaluation would need a separately annotated question set, independent answer assessment, baseline comparisons, and appropriate evaluation of the research contribution. No such results have been fabricated.

The frontend intentionally uses deterministic calculations and routing; the repository's separate LLM/TableRAG/GraphRAG backend has not been activated or evaluated in this task. No new clinical recommendations, dose advice, current approval claims, or external medical facts are generated from the review data.
```
<!-- snapshot-end:docs/chatbot-validation.md -->

## Documentation migration — 2026-09-14

- Preservation snapshot and migration map: complete.
- Ten process files consolidated and four formal reports archived: complete.
- README navigation, scope boundaries and TODO decision states: complete.
- Verification at the end of the documentation migration, before subsequent editorial updates: 14 verbatim snapshots match their original SHA-256 values; four archived reports match the original bodies after documented notice/link changes; the Reddit source document and 120 tracked non-Markdown files are unchanged; 70 local links/anchors resolve. `git diff --check` passed.
- Inventory correction: an initial recursive count also found `outputs/chatbot-qa/task_plan.md`, a Git-ignored local QA note outside the maintained documentation. It remains untouched. The maintained Markdown set changed from 19 to 10 files; this count excludes generated/ignored outputs. The earlier inventory omitted that local note.
- Verification scope was corrected to the maintained README, TODO and docs tree after the first count included that ignored note. No content was deleted to satisfy the count.
- No code, dataset, manifest, audit JSON, reference PDF or prescribing-information PDF was changed. No application tests were rerun for this documentation-only migration; historical test results remain explicitly historical. No commit or push was performed.
