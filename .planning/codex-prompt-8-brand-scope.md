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
