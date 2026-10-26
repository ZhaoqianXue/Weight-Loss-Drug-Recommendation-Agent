# Extraction reference correction

The current validation table excludes three purpose labels from WebMD annotation sample 299 under the explicit-reason rule. The original annotations and pre-adjudication agreement remain unchanged. This is a reference correction, not a new model run or a full relabeling. Only text-derived purposes enter this score; profile conditions do not. No source review text is distributed.

| Quantity | Before | Current |
| --- | --- | --- |
| Reference purposes | 17 | 14 |
| Frozen predictions | 15 | 15 |
| True positives | 13 | 13 |
| False positives | 2 | 2 |
| False negatives | 4 | 1 |
| Precision, 95% interval | 0.87 [0.67, 1.00] | 0.87 [0.67, 1.00] |
| Recall, 95% interval | 0.76 [0.47, 1.00] | 0.93 [0.75, 1.00] |

Intervals use the existing post-level percentile bootstrap with 2,000 replicates and seed 1, scoring the 100 WebMD reviews separately. The matching rules and prediction set are unchanged. Original purpose agreement is still F1 = 0.21 on Reddit and 0.54 on WebMD; adjudication does not establish improved inter-annotator reliability.

[Aggregate audit JSON](reference-correction.json).
