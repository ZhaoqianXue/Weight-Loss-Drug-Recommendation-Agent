# WebMD review inventory audit

Checked September 11, 2026 (America/Phoenix); recorded September 12, 2026 05:03:50 UTC.

## Scope and method

Rechecked the eight ingredients discussed in this project. Opened each counted review page in a live browser. Counts use the unfiltered overall review total. Checked old generic and package links from WebMDRx for redirects. Insulin combinations are reported separately. This is an inventory of accessible WebMD pages, not an exhaustive worldwide brand registry.

## Verified pages

| Generic ingredient | Brand / page grouping | Displayed reviews | Scope |
|---|---|---:|---|
| Semaglutide | [Ozempic](https://reviews.webmd.com/drugs/drugreview-ozempic-semaglutide) | 512 | single_ingredient |
| Semaglutide | [Wegovy / Wegovy HD](https://reviews.webmd.com/drugs/drugreview-wegovy-semaglutide) | 491 | single_ingredient |
| Semaglutide | [Rybelsus](https://reviews.webmd.com/drugs/drugreview-rybelsus-semaglutide) | 81 | single_ingredient |
| Tirzepatide | [Mounjaro](https://reviews.webmd.com/drugs/drugreview-mounjaro-tirzepatide) | 662 | single_ingredient |
| Tirzepatide | [Zepbound](https://reviews.webmd.com/drugs/drugreview-zepbound-tirzepatide) | 48 | single_ingredient |
| Liraglutide | [Victoza](https://reviews.webmd.com/drugs/drugreview-liraglutide-victoza) | 440 | single_ingredient |
| Liraglutide | [Saxenda](https://reviews.webmd.com/drugs/drugreview-saxenda-liraglutide) | 222 | single_ingredient |
| Orforglipron | [Foundayo](https://reviews.webmd.com/drugs/drugreview-foundayo-orforglipron) | 0 | single_ingredient |
| Dulaglutide | [Trulicity](https://reviews.webmd.com/drugs/drugreview-trulicity-dulaglutide) | 271 | single_ingredient |
| Exenatide | [Byetta](https://reviews.webmd.com/drugs/drugreview-exenatide-byetta) | 421 | single_ingredient |
| Exenatide | [Bydureon / Bydureon BCise](https://reviews.webmd.com/drugs/drugreview-bydureon-bcise-exenatide) | 170 | single_ingredient |
| Insulin glargine + lixisenatide | [Soliqua](https://reviews.webmd.com/drugs/drugreview-soliqua-insulin-glargine-lixisenatide) | 6 | combination |
| Insulin degludec + liraglutide | [Xultophy](https://reviews.webmd.com/drugs/drugreview-xultophy-insulin-degludec-liraglutide) | 0 | combination |

Single-ingredient pages: **3,318** displayed records. Including the two insulin combinations: **3,324**. Foundayo and Xultophy explicitly display an invitation to submit the first review.

## Unconfirmed products

| Ingredient | Brand | Directory checked | Result |
|---|---|---|---|
| Lixisenatide | Adlyxin | [Current directory](https://www.webmd.com/drugs/2/alpha/a/ad) | No entry found; count unknown, not zero |
| Lixisenatide | Lyxumia | [Current directory](https://www.webmd.com/drugs/2/alpha/l/ly) | No entry found; count unknown, not zero |
| Albiglutide | Tanzeum | [Current directory](https://www.webmd.com/drugs/2/alpha/t/ta) | No entry found; count unknown, not zero |
| Albiglutide | Eperzan | [Current directory](https://www.webmd.com/drugs/2/alpha/e/ep) | No entry found; count unknown, not zero |

Also checked the Albiglutide alphabetical position in the [Al directory](https://www.webmd.com/drugs/2/alpha/a/al), and expanded the complete [Li directory](https://www.webmd.com/drugs/2/alpha/l/li) for Lixisenatide. Expanded the Ta directory beyond the first 20 entries. No matching current entries were found. This does not establish that no historical page ever existed.

## Legacy links and duplicate counting

Legacy counts below came from [WebMDRx](https://www.webmdrx.com/drug-classes/antihyperglycemic-glucagon-like-peptide-1-glp-1-receptor-agonists). Their live destinations were independently checked. Do not add these old counts to the current page totals.

| Legacy entry | Old directory count | Live destination |
|---|---:|---|
| [Victoza 2-Pak](https://www.webmd.com/drugs/drugreview-163054-Victoza-2-Pak) | 51 | [liraglutide-victoza](https://reviews.webmd.com/drugs/drugreview-liraglutide-victoza) |
| [Exenatide](https://www.webmd.com/drugs/drugreview-93223-Exenatide) | 4 | [exenatide-byetta](https://reviews.webmd.com/drugs/drugreview-exenatide-byetta) |
| [Semaglutide](https://www.webmd.com/drugs/drugreview-178018-Semaglutide) | 4 | [rybelsus-semaglutide](https://reviews.webmd.com/drugs/drugreview-rybelsus-semaglutide) |
| [Ozempic 1 Mg Dose](https://www.webmd.com/drugs/drugreview-181644-Ozempic-1-Mg-Dose) | 4 | [wegovy-semaglutide](https://reviews.webmd.com/drugs/drugreview-wegovy-semaglutide) |
| [Bydureon](https://reviews.webmd.com/drugs/drugreview-159405-bydureon-subcutaneous) | Not used | [bydureon-bcise-exenatide](https://reviews.webmd.com/drugs/drugreview-bydureon-bcise-exenatide) |

The Ozempic 1 Mg Dose redirect to Wegovy is unexpected. Preserve the original entry name, identifier and destination during any future collection; do not infer product equivalence from the redirect. Redirects alone cannot prove whether all historical reviews were migrated.

## Data-quality limitations

- Page totals count ratings/reviews, not verified unique patients or nonempty textual reviews.
- No complete pagination extraction or text deduplication performed.
- Missing directory entries are not proof of zero reviews or historical absence.
- Mounjaro shows 662 overall, but visible condition counts 333+328=661.
- Legacy Ozempic 1 Mg Dose ID 181644 redirects to Wegovy; website routing is not evidence of clinical brand equivalence.
- Shared URLs do not establish which historical reviews were retained or whether every record has the correct brand/formulation.

## Changes from the prior answer

- The original ten nonzero single-ingredient pages still total 3,318; their live counts agree with the earlier answer.
- Added the fixed-dose combination scope: Soliqua 6, Xultophy 0.
- Verified that four legacy generic/package entries are redirects, not additional current datasets.
- Replaced search-only absence checks with corresponding live directory checks. Unconfirmed counts remain null.
- Identified the Mounjaro one-record condition subtotal discrepancy.

Project review datasets and collection scripts were not changed.
