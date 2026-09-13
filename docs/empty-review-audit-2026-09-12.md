# Empty review text: current counts and historical handling

## Why 974 pending records but 933 with text?

The current `Extraction Status` mixes records needing model processing with records lacking any narrative to process. The previous summary used “pending” too broadly. The exact breakdown is:

| Current annotation status | Nonempty text field | Empty text field | Total |
| --- | ---: | ---: | ---: |
| Historical annotation reused | 2,300 | 44 | 2,344 |
| Pending | 933 | 41 | 974 |
| Total | 3,233 | 85 | 3,318 |

Thus there are **85 empty-text records in the whole dataset**. Forty-four already existed in the pre-refresh dataset and had reusable historical outputs; the other 41 entered with the three newly included brands and have no historical output to reuse.

| Brand | Empty-text records | Current treatment |
| --- | ---: | --- |
| Ozempic | 1 | Historical result reused |
| Victoza | 42 | Historical result reused |
| Saxenda | 1 | Historical result reused |
| Trulicity | 2 | Pending; no narrative to extract |
| Byetta | 34 | Pending; no narrative to extract |
| Bydureon | 5 | Pending; no narrative to extract |

“Nonempty” means `Textual Review.strip()` is not empty, not that the text contains useful medical information. In particular, Byetta review ID `2902` contains the literal string `None`. It is included among the 933 nonempty pending records. This is source text, not a Python null value; it should not be described as a substantive narrative without further text-quality screening.

## What does an empty record contain?

These records have numerical ratings and may also have a review date, reviewer display name, age group, medication duration and condition, but the collected public source has no narrative text. They are retained as rating records so that per-brand counts and rating summaries reflect the source dataset. An empty narrative is not a statement that no adverse effects occurred.

The live recheck of the 41 newly added empty records is recorded in [`empty-review-source-check-2026-09-12.json`](empty-review-source-check-2026-09-12.json). It checks the original WebMD review IDs, public embedded `UserExperience` field and three rating fields across their source pages. The recheck found all 41 IDs across 25 pages: every source `UserExperience` was empty and all three numeric rating fields were present. It does not establish whether a person originally left the text blank, or whether text became unavailable later through the platform.

## What did the old code do?

The implementation inspected is Git revision `f1e5e50dfc330c98317a29f5c0e6a1286584800e`. The corresponding pre-refresh raw/extracted/standardized CSVs are preserved in [`data_backup/pre_2026_refresh/`](../data_backup/pre_2026_refresh/).

1. **Collection retained the row.** `code_scraping/scraper.py` attempted the long-review spans, short-review text and a description fallback. If no text was found, the text remained empty, but `page_data.append(data)` still added the complete rating record. There was no “skip empty narrative” condition.
2. **CSV loading converted blanks to missing values.** `code_extraction/extract_all.py` used default `pandas.read_csv`. Reading the archived CSV reproduces 44 missing (`NaN`) `Textual Review` values.
3. **Extraction did not skip missing text.** `process_reviews_structured` passed `row['Textual Review']` to `safe_extract_structured` without a missing/empty-text check. That function passed the value to the prompt chain. The code therefore allowed a `NaN` value to reach prompt formatting, rather than explicitly identifying a rating-only record. No historical API logs were inspected, so this is a finding about the code path, not a reconstruction of individual provider responses.
4. **Historical outputs exist for all 44 blank records.** Both archived extraction and standardization CSVs retain all 44 rows; every one has `side_effects: []`. Other drug/condition fields and relations may have been populated from CSV metadata or automatic rules. The empty list does not demonstrate that the reviewer reported no side effects.
5. **There was no dedicated status distinction.** The old outputs lacked `Extraction Status`, so missing narrative and an extracted empty side-effect list were not explicitly distinguished. The current reuse count of 2,344 still includes these 44 legacy empty-text outputs; it must not be presented as 2,344 successfully analyzed narratives.

To inspect the exact old implementation:

```sh
git show f1e5e50dfc330c98317a29f5c0e6a1286584800e:code_scraping/scraper.py
git show f1e5e50dfc330c98317a29f5c0e6a1286584800e:code_extraction/extract_all.py
git show f1e5e50dfc330c98317a29f5c0e6a1286584800e:code_extraction/schema_extraction.py
```

## Current behavior and interpretation

The current extraction runner selects pending/failed rows **only when their text field is nonempty**, so the 41 new empty-text rows are not sent to the model. It reports them separately in its command output, although their CSV status remains `pending`. They can contribute to rating summaries, but cannot supply narrative adverse-effect evidence.

A more precise reporting label is **933 records with nonempty text awaiting processing + 41 rating-only records with no text to extract**, rather than implying all 974 need the same model processing. A future status schema should distinguish `no_text` from `pending` consistently for all 85 empty records. This clarification documents the existing data and behavior; it does not silently change the dataset statuses or statistical denominators.
