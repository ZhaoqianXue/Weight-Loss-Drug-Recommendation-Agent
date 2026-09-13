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
