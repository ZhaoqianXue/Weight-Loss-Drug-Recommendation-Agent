# Reddit data sources for the eight in-scope brands

Decision date: 2026-09-12. Scope: 4 generic names, 8 brands (see [project scope](project-scope.md)). WebMD sources are already fixed in [`config/drugs.json`](../config/drugs.json) and are not repeated here.

## How these numbers were produced

- Counts come from the Arctic Shift Reddit archive API (`https://arctic-shift.photon-reddit.com`), queried on 2026-09-12 with monthly `created_utc` aggregations summed by year. Subscriber counts come from the same archive's subreddit metadata and are refreshed infrequently.
- "Posts" and "comments" for a subreddit are the archived volume of that subreddit. "Keyword" counts are posts whose title or body contain the brand name, or comments whose body contains it; they are not filtered for self-reported use. Sehgal et al. (Nature Health, 2026) found that about 42% of GLP-1 posts indicate personal use, so self-report volume is roughly 0.4 x the keyword count.
- Many queries on very active subreddits (r/Mounjaro, r/Ozempic, r/Zepbound, r/WegovyWeightLoss, r/diabetes, r/mounjarouk, r/Retatrutide) timed out for some years. Every number marked `>=` is a lower bound. The archive lags the live site by roughly four to six weeks, and 2026 is partial.
- Rybelsus has no exact-name subreddit; its community is r/RybelsusPill, found by prefix enumeration. Byetta, Bydureon, dulaglutide, and exenatide have no subreddit with 200 or more subscribers under any name prefix (checked 2026-09-12), which is the basis for excluding the exenatide brands.

## Collection route

1. Bulk history: Academic Torrents per-subreddit dumps (Pushshift/Watchful1 lineage, currently through 2025-12) for every subreddit listed below. Do not pull large subreddits through the Arctic Shift API; it times out.
2. Recent months: Arctic Shift API or its monthly Parquet releases for 2026.
3. In parallel, apply to the Reddit for Researchers program (BigQuery access, IRB letter and institutional sponsor required, one-year access, no redistribution). Reddit states that research through the Data API or third-party tools is not authorized; the PRAW route used in the AMIA 2025 paper should not be reused. The Nature Health 2026 paper's data statement (Pushshift and Arctic Shift, raw data not shared) is the precedent to follow in the ethics section.
4. Keep only hashed author IDs, never redistribute raw text, paraphrase any quoted passage, and exclude content from quarantined or private communities.

## Community types

Reddit communities specialise by purpose. Record the type on every ingested subreddit; it is a stratification variable, not noise.

| Type | Examples | Handling |
| --- | --- | --- |
| Brand community | r/Ozempic, r/WegovyWeightLoss, r/RybelsusPill, r/Mounjaro, r/Zepbound, r/trulicity | Primary brand-level evidence |
| Molecule community | r/Semaglutide, r/liraglutide, r/tirzepatidehelp, r/TirzepatideRX | Brand resolved from text; otherwise "Unspecified" brand under the molecule |
| Maintenance phase | r/MounjaroMaintenance, r/ozempicmaintenance, r/Zepbound_Maintenance, r/WegovyMaintenance | Primary evidence; tag phase = maintenance |
| Type 2 diabetes specific | r/Mounjaro_ForType2, r/ozempicT2D; general r/diabetes, r/diabetes_t2, r/type2diabetes, r/prediabetes | Keyword-filtered evidence; aligns with WebMD's diabetes-dominant condition mix |
| Regional | r/mounjarouk, r/WegovyUK, r/OzempicAustralia, r/WegovyGermany, r/Semaglutide_UK | Tag region; WebMD is US-only |
| Compounded or grey market | r/tirzepatidecompound, r/CompoundedSemaglutide, r/compoundedtirzepatide, r/GLP1Sourcing, r/glp1peptides | Molecule-level only, product = compounded; never counted as brand evidence |
| Off-label practice | r/GLP1microdosing, r/TirzepatidePCOS, r/PCOS | Tag; keyword-filtered |
| Out of scope | r/Retatrutide and other unapproved agents; r/loseit and general weight-loss communities unless keyword-filtered | Exclude or keyword-filter only |

## Per-brand sources

Volumes are archived posts / comments for the whole subreddit unless the row says "keyword".

### Semaglutide

| Brand | Primary subreddits | Secondary and keyword sources | Notes |
| --- | --- | --- | --- |
| Ozempic | r/Ozempic (111,787 subscribers; 58,259 posts / 848,584 comments, 2019-2025); r/OzempicForWeightLoss (32,975; 14,257 / 143,995); r/ozempicmaintenance (4,248; 457 / 2,540, 2023-2026); r/ozempicT2D (412; 83 / 364) | r/Semaglutide (129,464; 66,066 / 911,391) as the molecule community; keyword in r/type2diabetes: 387 posts / 1,503 comments; keyword in r/diabetes and r/diabetes_t2 not yet measured; r/Ozempic_ (9,939), r/OzempicAustralia (876) unmeasured | Largest structured-review counterpart on WebMD (512 reviews) |
| Wegovy | r/WegovyWeightLoss (84,184; 59,833 / 709,459, 2022-2025); r/WegovyPillWeightLoss (7,474; 5,052 posts in 2026, oral Wegovy approved 2025-12-22); r/Wegovy (4,130; volume unmeasured); r/WegovyMaintenance (440) | r/Semaglutide (shared); r/WegovyUK (2,810), r/WegovyGermany (759) regional; r/zepbound2wegovy (438) switching | Oral pill community is new in 2026 and should be tagged formulation = oral |
| Rybelsus | r/RybelsusPill (3,563; 1,374 posts / 12,897 comments, 2023-2026) | keyword: r/Semaglutide >=2,350 posts / >=4,598 comments; r/Ozempic >=290 / >=1,348; r/diabetes_t2 153 / 723; r/diabetes >=173 / >=212; r/type2diabetes 48 / 142; r/GLP1 35 / 184; r/pharmacy 24 / 131 | Adequate for brand-level analysis; oral formulation |

### Tirzepatide

| Brand | Primary subreddits | Secondary and keyword sources | Notes |
| --- | --- | --- | --- |
| Mounjaro | r/Mounjaro (115,163; 86,947 / 1,281,781, 2022-2025); r/MounjaroMaintenance (12,890; 2,847 / 40,809, 2023-2026); r/Mounjaro_ForType2 (3,720; 2,451 posts, 2022-2026) | r/mounjarouk (19,183; unmeasured, regional); molecule: r/tirzepatidehelp (14,180; 1,784 / 46,542, 2024-2025), r/TirzepatideRX (12,869; 8,211 posts) | Compounded communities r/tirzepatidecompound (51,390; 58,498 / 1,109,329) and r/CompoundedTirzepatide (16,950; 9,117 / 136,514) are product = compounded, not Mounjaro |
| Zepbound | r/Zepbound (99,126; 115,858 / 1,811,851, 2023-2025); r/zepbound_support (2,833); r/zepboundathletes (4,017); r/zepboundRX (3,463); r/Zepbound_Maintenance (621) | r/zepbound2wegovy (438) switching; molecule communities as above | Largest Reddit community versus only 48 WebMD reviews; the strongest coverage inversion in the dataset |

### Liraglutide

| Brand | Primary subreddits | Secondary and keyword sources | Notes |
| --- | --- | --- | --- |
| Victoza | r/liraglutide (16,228; 10,754 posts / 97,803 comments, 2019-2026; shared Victoza and Saxenda community, brand resolved from text); r/Victoza (590, restricted; 34 / 319, 2022-2025) | keyword in r/diabetes, r/diabetes_t2, r/type2diabetes not yet measured | Diabetes indication; expect most Victoza evidence in r/liraglutide and diabetes communities |
| Saxenda | r/liraglutide (shared); r/saxendawegovymounjaro (3,761; 516 / 3,454, almost all 2023); r/saxendaforweightloss (626; 7 / 94) | keyword in weight-loss communities not yet measured | Daily injection for weight management; r/liraglutide activity peaked in 2023 and declined after tirzepatide |

### Dulaglutide

| Brand | Primary subreddits | Secondary and keyword sources | Notes |
| --- | --- | --- | --- |
| Trulicity | r/trulicity (2,370; 1,166 posts / 10,292 comments, 2022-2026) | keyword: r/diabetes >=613 posts / >=1,423 comments; r/diabetes_t2 >=438 / >=1,967; r/type2diabetes 111 / 359; r/Mounjaro >=482 / >=1,482 and r/Ozempic >=279 / >=779 (mostly switching narratives); r/PCOS 70 / 324; r/pharmacy 21 / 222 | Thinnest in-scope brand on Reddit; report with confidence intervals and use a SWITCHED_FROM relation for the r/Mounjaro and r/Ozempic mentions |

## Cross-brand and general communities

| Subreddit | Subscribers | Archived volume | Use |
| --- | --- | --- | --- |
| r/GLP1 | 6,561 | 8,249 posts, 2022-2026 | All molecules; brand resolved from text |
| r/GLP1_Ozempic_Weygovy, r/GLP1_Ozempic_Wegovy | 5,819; 2,285 | unmeasured | Semaglutide brands |
| r/GLP1microdosing | 2,027 | 6,360 posts / 69,427 comments, 2024-2026 | Off-label practice; tag |
| r/diabetes | 137,420 | very large; API times out | Keyword-filtered only |
| r/diabetes_t2 | 43,104 | large | Keyword-filtered only; subreddit rule "No Research Studies" targets recruitment, so no posting or contact |
| r/type2diabetes | 12,526 | moderate | Keyword-filtered only |
| r/prediabetes | 23,466 | moderate | Keyword-filtered only |

## Excluded exenatide brands, for the record

Byetta and Bydureon (exenatide) were removed from the project on 2026-09-12. Across 18 diabetes and GLP-1 subreddits from 2018 to 2026 the archive holds about 70 posts and 281 comments mentioning Byetta and about 95 posts and 285 comments mentioning Bydureon, roughly half of them in r/diabetes and r/diabetes_t2. No subreddit for either brand has 200 or more subscribers (r/bydureon holds one post). AstraZeneca discontinued Byetta on 2024-10-25 and Bydureon BCise on 2024-10-28 in the United States. The WebMD records (421 Byetta, 170 Bydureon) are archived, not deleted.
