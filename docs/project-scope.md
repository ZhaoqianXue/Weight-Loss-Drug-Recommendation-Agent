# Project medication scope

## Included medications

The maintained catalog is [`config/drugs.json`](../config/drugs.json). The current project includes these four generic names and their eight brands:

| Generic name | Canonical brands |
| --- | --- |
| Semaglutide | Wegovy, Ozempic, Rybelsus |
| Tirzepatide | Mounjaro, Zepbound |
| Liraglutide | Victoza, Saxenda |
| Dulaglutide | Trulicity |

Wegovy HD is an alias of Wegovy; Victoza 2-Pak and Victoza 3-Pak are aliases of Victoza. Aliases do not create additional datasets. The per-brand Reddit sources are documented in [Reddit data sources](reddit-data-sources.md).

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

These findings were supplied with the scope decision; no new Reddit collection or source investigation was performed in this update. The remaining eight brands' sources and the recorded exclusion evidence are in [Reddit data sources](reddit-data-sources.md); that document is retained unchanged.

The **421 Byetta reviews and 170 Bydureon reviews (591 total)** were moved, not deleted, to [`data_backup/excluded_exenatide_2026-09-12/`](../data_backup/excluded_exenatide_2026-09-12/). The eight retained per-brand WebMD files are unchanged. The combined raw dataset and its collection manifest were rebuilt offline, retaining original collection timestamps; extracted and standardized data and website copies were then regenerated without scraping or model API calls.

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
