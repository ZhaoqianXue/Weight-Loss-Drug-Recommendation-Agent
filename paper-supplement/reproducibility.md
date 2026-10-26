This document preserves the complete supplementary explanation from the full manuscript. Table numbers below follow that complete version and can differ from the shortened submission.

# Reproducibility Statement

**Subreddits.** Table 11 lists the 16 subreddits. The collection window starts at the earliest FDA approval among the brands assigned to the subreddit in the collection configuration. This assignment sets only the start date but does not label individual posts. All windows end at the snapshot of September 15, 2026.

| Subreddit | Type | Start date | Submissions | Comments |
| --- | --- | --- | --- | --- |
| mounjaro | B | 2014-09-18 | 104,709 | 1,532,780 |
| ozempic | B | 2014-09-18 | 64,624 | 930,335 |
| rybelsuspill | B | 2019-09-20 | 1,374 | 12,898 |
| saxendaforweightloss | B | 2014-12-23 | 7 | 94 |
| trulicity | B | 2014-09-18 | 1,166 | 10,292 |
| victoza | B | 2010-01-25 | 35 | 320 |
| wegovy | B | 2021-06-04 | 6,387 | 44,267 |
| wegovyweightloss | B | 2021-06-04 | 70,819 | 851,545 |
| zepbound | B | 2023-11-08 | 156,962 | 2,613,491 |
| liraglutide | I | 2010-01-25 | 10,785 | 97,888 |
| semaglutide | I | 2017-12-05 | 75,779 | 1,040,727 |
| tirzepatiderx | I | 2022-05-13 | 8,263 | 124,128 |
| glp1 | X | 2010-01-25 | 8,260 | 100,332 |
| saxendawegovymounjaro | X | 2014-12-23 | 516 | 3,454 |
| diabetes_t2 | T | 2010-01-25 | 33,286 | 493,971 |
| type2diabetes | T | 2010-01-25 | 8,722 | 87,788 |
| Total |  |  | 551,694 | 7,944,310 |

B = brand; I = ingredient; X = cross-drug; T = type 2 diabetes. Start dates follow the collection configuration and do not label individual posts.

[Download TSV](subreddits.tsv).

**Reddit data cleaning.** It has three stages. The first drops posts under 20 words, deleted or empty placeholders, link-only content, bot signatures, moderator templates, and text identified as non-English with high confidence, leaving 4,804,353 posts. The second removes near-duplicates, boilerplate, and posts dominated by a title, emoji, or copied sentences, leaving 4,660,528. The third, lossy by design, keeps posts with more than 40 distinct content words (words other than function words, URLs, and stock social phrases, outside sentences repeated across posts) and drops promotional and referral templates.

**Content-word threshold selection.** Requiring more than 40 distinct words within a post yields a retained corpus close to one million posts, the target set by the LLM budget. We fix this threshold before detection and information extraction. Rules on raw length (88 words) or on other word counts would select largely the same posts (Jaccard 0.84 to 0.95). The kept posts are 22.1% of cleaned posts but hold 49.9% of their text. Keep rates vary little by year (20.7% to 23.4% from 2022 to 2026) and are higher in the two diabetes subreddits (28%) than in drug subreddits (19% to 22%). Among kept posts, detection keeps a share that rises steadily with length, from 17.7% at the threshold to 49.6% above 200 content words. Longer side-effect posts also name more PTs, from 1.8 per post at 41 to 50 content words to 2.8 above 120. We therefore repeat the analyses of Section 4.1 and Section 4.2 within four bands of kept posts (41–50, 51–80, 81–120, and above 120 content words). All seven tirzepatide–semaglutide differences keep non-overlapping intervals in every band, the brand and yearly comparisons keep their direction, and Spearman’s ρ with FAERS stays between 0.21 and 0.22. In the shortest band, 5 of the 55 calls against FAERS change, all for PTs whose Reddit and FAERS shares differ by less than 1 percentage point. WebMD reviews are not filtered by length. Of reviews reporting a side effect, 46% fall below the Reddit threshold. These short reviews rank PTs much as Reddit posts do (ρ = 0.87), show the same tirzepatide–semaglutide gaps for nausea and vomiting, and keep every gap to FAERS named in Section 4.2.

**Drug identification.** Each relation records whether its ingredient comes from a named drug, the WebMD profile, or the subreddit. Among Reddit side-effect relations without a named drug, 98.7% receive an ingredient from the subreddit. A subreddit does not assign a brand, so brand-level counts are based on named brands only. In the narratives and in FAERS alike, compounded products and mentions of an ingredient alone count at the ingredient level with the brand unspecified.

**FAERS data processing.** Each report enters in its latest version, and study drugs are identified from the reporter’s product-name and active-substance fields. The openFDA harmonized fields incorrectly assign 99.8% of raw Ozempic entries to Rybelsus. Entries for insulin degludec/liraglutide, a fixed combination with insulin outside the eight study brands, are dropped, and blinded trial drugs are left out of the comparison. Duplicate cases submitted by different senders are kept. The directly identifiable ones make up less than 1% of the compared reports. Reported FAERS terms span MedDRA versions 16.0 to 29.0 and are mapped to 29.1 PTs, directly or through their LLT, leaving 4 of 1,061,384 reaction entries unmatched. An independent rebuild of the report, drug, and reaction tables from the raw API responses reproduces every processed row.

**RQ2 comparison data.** The comparison window runs from each brand’s FDA approval (or an ingredient’s first approved brand) through June 30, 2026, and excludes 171 FAERS reports and 7,627 narrative posts with side effects. The primary comparison keeps FAERS reports in which a study drug is suspect or interacting. Non-clinical terms, excluded from all three sources, are PTs whose primary MedDRA path lies in the official groups for product issues, medication errors, off-label use and intentional misuse, therapeutic response and drug interaction terms, overdose and underdose, or pregnancy and lactation exposure. Excluding them removes 18.5% of FAERS report–term pairs and 0.16% of narrative ones. Narrative denominators count posts or reviews with at least one retained side effect, while the FAERS denominator counts reports with at least one retained event. The narrative denominators include 3,246 Reddit posts and 20 WebMD reviews whose side effects are uncoded.

**Term coding.** The retrieval index contains 82,124 LLTs. Of the 385,790 relations, 93.8% receive a PT, with coverage of 95.4% among side-effect relations and 81.9% among purpose relations. The coding method is fixed on development samples without human labels. With 20 candidates instead of 10, the selector chooses one of the 10 added candidates for only 3.4% of 1,000 development relations, whereas two identical runs already disagree on 3.2%. A wording that names several concepts receives a single code.
