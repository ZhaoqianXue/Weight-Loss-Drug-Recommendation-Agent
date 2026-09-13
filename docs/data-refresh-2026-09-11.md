> **Addendum — 2026-09-12:** The project now includes **4 generic names, 8 brands and 2,727 records**. See [the scope decision and current counts](project-scope.md#explicit-exclusion-exenatide-products). The report below preserves the original collection and validation results as a historical record.

# Five-generic WebMD dataset refresh — 2026-09-11 (America/Phoenix)

Collection publication completed at 2026-09-12T05:32:49.141428+00:00 (UTC). This report describes the local project snapshot, not a deployed external service.

## Scope and collected records

The current dataset contains **3,318 unique WebMD review IDs**, across **172 review pages**, **5 generic names**, and **10 brands**. **3,233 records contain narrative text; 85 are rating-only.** The source pages report the same per-brand totals. Generic names are grouping columns, not extra copies of the brand records.

| Generic name | Canonical brand / source | Records | With text | Reused annotation | Pending |
| --- | --- | ---: | ---: | ---: | ---: |
| Semaglutide | [Wegovy](https://reviews.webmd.com/drugs/drugreview-wegovy-semaglutide) | 491 | 491 | 477 | 14 |
| Semaglutide | [Ozempic](https://reviews.webmd.com/drugs/drugreview-ozempic-semaglutide) | 512 | 511 | 497 | 15 |
| Semaglutide | [Rybelsus](https://reviews.webmd.com/drugs/drugreview-rybelsus-semaglutide) | 81 | 81 | 75 | 6 |
| Tirzepatide | [Mounjaro](https://reviews.webmd.com/drugs/drugreview-mounjaro-tirzepatide) | 662 | 662 | 623 | 39 |
| Tirzepatide | [Zepbound](https://reviews.webmd.com/drugs/drugreview-zepbound-tirzepatide) | 48 | 48 | 22 | 26 |
| Liraglutide | [Victoza](https://reviews.webmd.com/drugs/drugreview-liraglutide-victoza) | 440 | 398 | 431 | 9 |
| Liraglutide | [Saxenda](https://reviews.webmd.com/drugs/drugreview-saxenda-liraglutide) | 222 | 221 | 219 | 3 |
| Dulaglutide | [Trulicity](https://reviews.webmd.com/drugs/drugreview-trulicity-dulaglutide) | 271 | 269 | 0 | 271 |
| Exenatide | [Byetta](https://reviews.webmd.com/drugs/drugreview-exenatide-byetta) | 421 | 387 | 0 | 421 |
| Exenatide | [Bydureon](https://reviews.webmd.com/drugs/drugreview-bydureon-bcise-exenatide) | 170 | 165 | 0 | 170 |

Wegovy HD is represented as **Wegovy**; Bydureon BCise is represented as **Bydureon**. These are naming aliases for shared review pages. Different formulations are not split into new brands, and formulation must not be inferred from the brand name alone. The generic/brand catalog is `config/drugs.json`; Foundayo/orforglipron and other generics are outside this requested scope. Combination products are explicitly excluded regardless of review availability, including Soliqua (insulin glargine + lixisenatide) and Xultophy (insulin degludec + liraglutide); see [the project scope](project-scope.md).

## Collection and provenance

- `data_webmd/collection_manifest.json` records source URLs, per-brand headline/collected counts, pages, publication time, and SHA-256. Every row includes `Review ID`, `Source URL`, `Source Page`, `Collected At`, and `Source Visibility`.
- The collector parses public embedded JSON for stable IDs, full narratives and numeric ratings. Demographics come from visible review cards because embedded gender/user-type labels disagreed with rendered labels. The `75 or over` age group is supported; affected pages were fetched again and their IDs and totals checked before publication.
- Exactly **3,317 records are rendered cards** and **1 Mounjaro record is embedded-only**. That record is included in the 662 headline total but has no rendered card or condition classification. It is retained with `Source Visibility=embedded_only`; unverified demographics are blank. Filter this column if an analysis requires rendered-card-only reviews.
- Rating-only records are preserved. Distinct review IDs with identical text are not collapsed. Email-masked display names remain masked.
- Pagination rejects repeated IDs, unexpected drug redirects, changing totals, unmatched visible cards, and incomplete final counts. All ten brands must finish validation before publishing. Individual files use temporary-file replacement. Page checkpoints are local, ignored caches; they are not separately added to record counts.

## Annotation handling and outstanding work

**Raw collection and website inclusion are complete. New model annotation is not complete.** The current environment has no `OPENAI_API_KEY`; this refresh made no new extraction or standardization model calls.

**2,344 historical annotations** were reused only when brand, date, author, and full non-whitespace review characters matched uniquely. Whitespace is ignored solely because the old scraper joined expanded text spans without spaces. No fuzzy paraphrase matching was used. Missing author and the displayed Anonymous label are normalized. Sixteen otherwise matching historical reviews had side effects or relations attributed to a different medication; those entire annotations were withheld and returned to pending. Archived original outputs are retained unchanged.

**974 records carry the pending status**, comprising **933 nonempty text fields** awaiting processing and **41 rating-only records** with no narrative to extract. The 2,344 reused historical outputs separately include 44 empty-text records, so the dataset contains 85 empty narratives overall. Nonempty text is not automatically substantive: one of the 933 contains the literal string `None`. See [the empty-text audit](empty-review-audit-2026-09-12.md) for exact counts and the old code path. Pending rows contain source drug/condition/duration metadata and no invented adverse effects. A `reviewed_for` relation represents the source condition field; it is distinct from an extracted `treats` relation and is not a claim of efficacy or approval. Unannotated rating-only records are skipped by the model extraction runner.

The existing adverse-event labels and historical FDA fields have not been clinically revalidated. New regulatory classification returns unknown because the checked-in prescribing-information material is historical; missing information is never converted into off-label use. Previous UMLS experiments, top-10 outputs, reports, embeddings and prescribing PDFs remain historical artifacts, not refreshed findings.

## Updated components

- `code_scraping/`: ten-brand catalog collection, retryable requests, validation, resumable page cache, provenance and canonical combined CSV.
- `code_pipeline/refresh_data.py`: conservative annotation matching, metadata fallback, canonical extraction/standardization CSVs, both website CSV copies, manifests and public catalog copy.
- `code_extraction/extract_all.py` and `code_standardization/standardize_all.py`: pending-only incremental runs, per-record checkpoints, optional `--limit`, credential checks, and website publication after standardization. Dosage form is no longer inferred solely from a brand name.
- Website: five generic filters, ten brand filters, generic-name questions, canonical aliases, descriptive ratings and source review browsing, visible extraction coverage, and unknown side-effect results for unprocessed drugs. Selected drugs remain visible even without extracted effects.
- Backend: expanded catalog and generic filtering guidance; dataset/model fingerprints prevent stale FAISS loading. Neo4j requires an empty-database snapshot import and verifies dataset provenance. The Flask static page can load without model initialization.

**FAISS and Neo4j were not rebuilt or connected in this environment.** Their historical files are not claimed as current. The new guards prevent treating them as current data. Backend live model/embedding/database integration remains untested here.

## Reproduce and finish model processing

```sh
python3 -m pip install -r code_scraping/requirements.txt
python3 code_scraping/batch_scraper.py
python3 code_pipeline/refresh_data.py
python3 code_pipeline/validate_data.py
# With the research dependencies installed and OPENAI_API_KEY set locally:
python3 code_extraction/extract_all.py --limit 50
python3 code_standardization/standardize_all.py --limit 50
# Repeat without --limit to process all eligible pending text.
# Rebuild retrieval indexes, and import into an empty configured Neo4j database:
python3 code_chatbot/table_loader.py
python3 code_chatbot/graph_loader.py
```

Start a new default collection run directory for a fresh crawl. To resume only an interrupted snapshot, pass its previous `--run-dir` path. The completed refresh used `.cache/webmd/refresh-20260912`. Model runners do not read credentials from chat or load `.env` automatically; configure them locally. The immutable pre-refresh CSVs and hashes are in `data_backup/pre_2026_refresh/`.

## Verification

- **15 Python tests passed**: source identity, visible demographics and oldest age group, Unicode, embedded-only alignment, masked names, malformed pages, rating mismatch, duplicate pagination, no publishing after collection failure, catalog/count/ID alignment, conservative annotation matching, and stale-index rejection.
- **4,565 JavaScript assertions passed**, covering all 10 brands, all 45 brand pairs, all 5 generic-name queries, aliases, source browsing, numeric statistics and unknown extraction handling. These are code assertions, not independent clinical evaluations.
- Independent CSV validation passed with the bundled Python runtime. All canonical outputs share the 3,318 review IDs; the three standardized website/backend CSV copies have identical hashes. Numeric ratings are in [1,5] and review narratives contain no replacement-character decoding artifacts.
- Browser checks verified all five generic filters and their corresponding brand lists; Dulaglutide shows Trulicity, Exenatide shows Byetta/Bydureon, and Trulicity nausea reports “Not available” with 0/271 processed. Source condition relations remain visible without implying extracted side effects.

Machine-readable integrity results: [`data-refresh-validation.json`](data-refresh-validation.json). Current annotation coverage and dataset hash: `data_standardized/dataset_manifest.json`.
