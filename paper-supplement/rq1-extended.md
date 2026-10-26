This document preserves the complete supplementary explanation from the full manuscript. Table numbers below follow that complete version and can differ from the shortened submission.

# RQ1 Supplementary Results

**WebMD profile context.** Table 5 provides data context by listing the six original condition-treated labels from 2,129 WebMD reviews with a retained profile condition. Each review contributes one profile condition, and the three weight-management categories sum to 868 reviews. Reasons extracted from review text remain separate and are excluded from these counts.

### Table 5. WebMD profile conditions

| Condition treated | Reviews |
| --- | --- |
| Type 2 Diabetes Mellitus | 1,257 |
| Weight Loss Management for an Obese Person | 531 |
| Weight Loss Management for Overweight Person with BMI 27 to 29 and Weight-Related Comorbidity | 333 |
| weight loss management, obese adolescent (BMI 95th percentile or greater for age and sex) | 4 |
| cardiovascular event risk reduction in obesity | 3 |
| sleep apnea due to airway obstruction | 1 |
| Total | 2,129 |

**Comparison Scope.** For every pair of the four ingredients and of the eight brands, we compare each PT reported in at least 1% of either group’s side-effect posts and in at least 30 posts in either group. On Reddit, this gives 202 comparisons between ingredients and 925 between brands, of which 111 and 309 have non-overlapping intervals. On WebMD, it gives 76 and 196 comparisons, of which 14 and 28 do. Over time, we compare 2025, the last full year, with 2022 for semaglutide and with 2023, the first full year after Mounjaro’s approval, for tirzepatide. This covers the 28 and 29 PTs in at least 1% of each drug’s posts. We compare years within each drug because posts shift toward tirzepatide (Figure 3c). Liraglutide and dulaglutide, with at most 1,558 and 318 side-effect posts a year, are left out, whereas semaglutide and tirzepatide have at least 2,272 in every year from 2022. The [reproducibility documentation](reproducibility.md) repeats the comparisons of Section 4.1 within length bands of kept posts.

### Table 6. Ingredient comparisons, pooled shares and annual ratios

| PT | Semaglutide share (%) | Tirzepatide share (%) | Ratio 2022 | Ratio 2023 | Ratio 2024 | Ratio 2025 | Ratio 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vomiting | 15.0 | 8.2 | 0.55 | 0.58 | 0.51 | 0.57 | 0.57 |
| Malaise | 8.6 | 6.2 | 0.75 | 0.82 | 0.71 | 0.71 | 0.67 |
| Nausea | 38.3 | 29.1 | 0.82 | 0.74 | 0.78 | 0.80 | 0.75 |
| Decreased appetite | 5.0 | 7.1 | 1.75 | 1.28 | 1.37 | 1.55 | 1.40 |
| Pain | 2.1 | 3.4 | 2.29 | 2.00 | 1.68 | 1.30 | 1.70 |
| Insomnia | 1.1 | 2.2 | 4.07 | 2.82 | 2.09 | 1.99 | 1.40 |
| Feeling cold | 0.5 | 1.6 | 2.67 | 4.81 | 3.25 | 2.97 | 4.12 |
| Pruritus | 0.3 | 1.3 | 12.53 | 6.17 | 3.34 | 3.04 | 6.04 |

### Table 6 continued. Drug identification, brand pair and platform

| PT | Ratio named drug | Ratio inferred ingredient | Ratio Mounjaro/Ozempic | Ratio Zepbound/Wegovy | Ratio WebMD |
| --- | --- | --- | --- | --- | --- |
| Vomiting | 0.54 | 0.55 | 0.59 | 0.48 | 0.45 |
| Malaise | 0.69 | 0.75 | 0.77 | 0.58 | 0.65† |
| Nausea | 0.73 | 0.76 | 0.76 | 0.71 | 0.66 |
| Decreased appetite | 1.40 | 1.39 | 1.34 | 1.55 | 2.00 |
| Pain | 1.47 | 1.84 | 1.55 | 1.37 | 0.90† |
| Insomnia | 2.09 | 2.03 | 2.12 | 2.06 | 0.40† |
| Feeling cold | 3.07 | 3.99 | 3.37 | 2.75 | 0.67† |
| Pruritus | 2.77 | 6.86 | 2.42 | 3.42 | 1.80† |

Shares count each drug’s side-effect posts. Ratios divide tirzepatide shares by semaglutide shares. Named and inferred refer to the basis of drug identification. The two brand-pair columns compare brands with a shared approved indication. The persistence rule excludes WebMD. † marks interval overlap or a reversal in direction. Annual and subset Wilson intervals are pointwise and unadjusted. The TSV stores the WebMD ratio numerically and its dagger annotation in a separate Boolean column.

[Download TSV](ingredient-comparisons.tsv).

### Table 7. Fatigue comparisons by brand and year

| Brands | Reddit diabetes (%) | Reddit weight management (%) | Years | WebMD diabetes (%) | WebMD weight management (%) |
| --- | --- | --- | --- | --- | --- |
| Ozempic, Wegovy | 10.7 | 14.2 | 5 of 5 | 19.0† | 20.8† |
| Mounjaro, Zepbound | 11.3 | 14.1 | 3 of 4 | 13.3† | 21.2† |
| Victoza, Saxenda | 9.2 | 14.9 | 1 of 5 | 6.9 | 19.0 |

Years count non-overlapping pointwise 95% Wilson intervals in the pooled direction among years with posts about both brands, through June 2026. † marks overlapping intervals. Correction results and annual limitations are stated in the accompanying text.

### Table 8. Between-year comparisons across subsets

| Drug | PT | Baseline year | All side-effect posts (%) | Named-drug side-effect posts (%) | All purpose-or-side-effect posts (%) |
| --- | --- | --- | --- | --- | --- |
| Semaglutide | Dyspepsia | 2022 | 6.0 → 4.3 | 6.1 → 4.0 | 5.3 → 3.9 |
| Semaglutide | Abdominal distension | 2022 | 5.8 → 4.4 | 5.6 → 4.2 | 5.1 → 4.0 |
| Semaglutide | Nausea | 2022 | 44.2 → 36.0 | 40.9 → 33.1 | 39.0 → 32.3 |
| Semaglutide | Alopecia | 2023 | 1.3 → 2.2 | 1.7 → 2.6 | 1.2 → 1.9 |
| Tirzepatide | Fatigue | 2023 | 12.7 → 14.5 | 11.4 → 13.2 | 10.5 → 12.7 |
| Tirzepatide | Abdominal discomfort | 2023 | 1.9 → 2.7 | 1.6 → 2.5 | 1.6 → 2.3 |
| Tirzepatide | Alopecia | 2023 | 1.6 → 2.3 | 1.9 → 3.0 | 1.3 → 2.0 |

Each arrow runs from the baseline year to 2025. Semaglutide alopecia uses the supplemental 2023 baseline. Named-drug posts and the wider purpose-or-side-effect denominator are separate subset checks. These are descriptive/pointwise-interval comparisons, not an alternative multiplicity-corrected significance rule. The primary semaglutide alopecia comparison starts in 2022 and does not pass correction (p = 0.409). The supplemental 2023–2025 result is not adjusted for baseline selection. Tirzepatide alopecia does not pass correction (p = 0.055).

[Download TSV](year-subsets.tsv).

**Comparisons by Drug.** A difference between two drugs persists when the intervals stay apart, in the same direction, from 2022 to 2026. For semaglutide and tirzepatide, it must also hold between the brands with a shared approved indication, Ozempic and Mounjaro, and Wegovy and Zepbound. Of the 30 PTs compared for these two drugs, 19 differ in the pooled data and 8 persist (Table 6). Besides the seven in Section 4.1, the eighth is pain, the general MedDRA term, whose relations are worded “pain” or “body aches” in 2,115 of 4,061 cases. The other 11 PTs differ in only some years, and alopecia, anxiety, asthenia, constipation, dyspepsia, eructation, and headache reverse direction in at least one year. No other pair of ingredients has a difference that persists. Brands are never inferred from the subreddit, so for the 28 brand pairs only the yearly rule applies. Seven brand differences pass it. Six are nausea or vomiting between a semaglutide brand and a tirzepatide brand, and the seventh is fatigue between Ozempic and Wegovy.

**Comparisons by Brand.** Table 7 gives annual fatigue comparisons using 95% Wilson intervals. Ozempic and Wegovy have non-overlapping intervals in all 5 years. Mounjaro and Zepbound have non-overlapping intervals from 2024 to 2026 but overlapping intervals in 2023, which has 45 Zepbound posts. Victoza and Saxenda have non-overlapping intervals in 1 of 5 years, with 15 to 106 Victoza posts a year, and their point-estimate direction reverses in 2025. The pooled Reddit fatigue comparisons yield corrected p=1.74×10^-18 for Ozempic and Wegovy and p=2.81×10^-11 for Mounjaro and Zepbound. Victoza and Saxenda yield corrected p=1. For other PTs, Ozempic and Wegovy differ for headache and nausea (both corrected p<0.001). Mounjaro and Zepbound differ for diarrhea (corrected p=0.0132), eructation (p=0.0152), and malaise (p<0.001). On WebMD, the Victoza and Saxenda intervals separate, but neither brand has 30 fatigue-positive reviews. The other two pairs have overlapping intervals (both corrected p=1).

**Comparisons by Year.** A difference is retained across subsets when non-overlapping 95% Wilson intervals keep the same direction in posts naming the drug, under the wider denominator of posts linking the drug to a purpose or side effect, and under both restrictions. In the full sample, 5 semaglutide PTs (2022 vs. 2025) and 7 tirzepatide PTs (2023 vs. 2025) have non-overlapping intervals. Three of these between-year differences for each drug meet the subset rule (Table 8). Exceptions are gastrooesophageal reflux disease and abdominal discomfort for semaglutide, and abdominal pain upper, feeling cold, insomnia, and pain for tirzepatide. Insomnia point estimates remain lower in 2025 than in 2023 in the named-drug subsets, but the intervals overlap. Section 4.1 reports alopecia from a supplemental 2023 baseline for semaglutide and the 2023 baseline for tirzepatide. From the 2022 baseline, the semaglutide intervals overlap (1.5% and 2.2%, corrected p=0.409). The supplemental 2023–2025 comparison yields corrected p=9.28×10^-7 across the 28 PTs tested in that window, without adjustment for baseline selection.

**Adjustment for Multiple Comparisons.** We run two-sided two-proportion z-tests with Bonferroni correction within each family. All p values in this section are Bonferroni-adjusted, with a significance level of 0.05. Drug and brand comparisons are grouped by platform and comparison level. Between-year comparisons are grouped by drug and the years compared. All eight pooled Reddit semaglutide–tirzepatide differences remain significant (p<0.001). For the primary semaglutide between-year comparison, corrected p=0.00778 for abdominal distension and p<0.001 for dyspepsia and nausea. For the tirzepatide between-year comparison, corrected p=0.0421 for abdominal discomfort, p=0.0186 for fatigue, and p=0.00864 for insomnia. Tirzepatide alopecia does not pass correction (p=0.055). These tests use the full sample in each compared year. Table 8 also reports CI checks for named-drug posts and wider denominators. Wilson intervals for annual and subset comparisons are pointwise and unadjusted for multiple comparisons.
