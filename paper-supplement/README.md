# Paper supplementary materials

This review-v2 documentation accompanies the shortened submission and preserves the complete methods and extended results retained in the full manuscript. The research code, nine prompt files and complete 120-question set are unchanged. Table numbers in these documents follow the complete manuscript, before submission tables are renumbered.

| Material | Readable document | Machine-readable files |
| --- | --- | --- |
| Full paper prompt descriptions | [Prompts](prompts.md) and [frozen prompt index](../prompts/README.md) | Existing prompt files |
| Extraction fields, profile rules and grounding checks | [Extraction schema](extraction-schema.md) | [Reddit schema](schema-reddit.json), [WebMD schema](schema-webmd.json) |
| Complete validation, error diagnostics and rating sensitivity | [Validation details](validation-details.md) | [Reference correction](reference-correction.json) |
| Extended ingredient/year comparisons, original Tables 6 and 8 | [RQ1 extended results](rq1-extended.md) | [Ingredient comparisons](ingredient-comparisons.tsv), [Year subsets](year-subsets.tsv) |
| Collection, cleaning, length checks, FAERS processing and coding diagnostics | [Reproducibility](reproducibility.md) | [16 subreddits, original Table 11](subreddits.tsv) |
| Corrected text-only WebMD purpose reference | [Reference correction](reference-correction.md) | [Aggregate audit](reference-correction.json) |
| Complete 120 evaluation questions | [Question list and field definitions](../evaluation/questions/README.md) | [TSV](../evaluation/questions/questions.tsv) |

The submission retains the core methods, primary results, annotation/scoring rules, representative questions, fatigue counterexamples, RQ2 denominators and automatic-marker limitations, and an ethics statement. Detailed tables here do not replace its main-paper evidence. Annual/subset Wilson intervals remain pointwise; the reported p values use their stated Bonferroni families. Supplemental semaglutide alopecia results are not adjusted for baseline selection. Stopping/switching markers remain unvalidated automatic output. Length checks within retained Reddit posts do not validate excluded short posts.

The repository does not distribute source user posts/reviews, account identifiers, model credentials or licensed MedDRA vocabulary files. The synthetic offline check described in the root README does not reproduce the paper’s empirical results.
