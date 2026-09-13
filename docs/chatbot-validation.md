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

The current website and assistant use `code_website/static/standardized_reviews_all.csv`, **2,727 rows covering eight brands and four generic names**. SHA-256: `e4bb15982d884905efb723b9c0d68be6dc60e1e4ea56bfb4ebf0bb69607840d8`. The September refresh reused 2,344 historical annotations and explicitly marks 383 rows pending (381 with text and 2 rating-only). See [the current project scope](project-scope.md) and machine-readable manifests for the current snapshot; [the original refresh report](data-refresh-2026-09-11.md) is historical.

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
