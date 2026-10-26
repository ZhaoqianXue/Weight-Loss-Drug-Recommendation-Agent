This document preserves the complete supplementary explanation from the full manuscript. Table numbers below follow that complete version and can differ from the shortened submission.

# Extraction Schema

The model returns one JSON object per post under a strict output schema. For Reddit, the object holds a `writer` and a list of `drugs`. For WebMD, it holds only the `drugs`, because the profile supplies the demographics.

- `writer`: the Reddit user’s `age_group` (13–18, 19–24, 25–34, 35–44, 45–54, 55–64, 65–74, 75 or over, or null), `sex` (female, male, other, or null), and `evidence`, a list of sentence numbers.

- `drugs`: `brand` (one of the eight brands, or null) and `ingredient` (one of the four ingredients, or null), which must agree, with lists of `purposes` and `side_effects`.

- `purposes`: `purpose`, in the user’s words, and `evidence`, at least one sentence number.

- `side_effects`: `effect`, in the user’s words; `dose` and `duration`, as written; `intensity` (mild, moderate, or severe); `impact` (daily life, medical visit, emergency visit, or hospitalization); `action_taken` (any of dose reduced, escalation paused, dose skipped, or stopped); and `evidence`.

**Grounding checks.** Code maps each cited sentence number to its character span in the post and checks that each purpose and side-effect wording occurs in the cited sentences, allowing a shared term (“nauseous” and “nausea”) or a slight misspelling of a long word. A post that fails the check is extracted once more, and relations that still fail are dropped (575 of 384,236, or 0.15%).

**WebMD profiles.** The reviewed drug comes from the profile, and the review is assigned to it unless the text names another study drug. Patient profiles, and profiles with the role left blank, also include the information for age group, sex/gender, and the condition treated as a purpose (unless marked “Other”). WebMD records age in 11 bands. The KG keeps the eight from 13–18 upward and leaves younger ages missing (13 reviews). The 25 caregiver profiles supply none of these fields, because their entries may describe either the caregiver or the patient, and the rules for the user’s own use also leave a caregiver’s account of the patient out of the review text.

## Machine-readable schemas

- [Reddit strict request schema](schema-reddit.json)
- [WebMD strict request schema](schema-webmd.json)

These JSON files are exported from the frozen extraction models using the same strict-schema builder as the research requests. Every object is closed and every property is required; optional values use null. String lengths, list sizes, brand–ingredient compatibility, sentence existence and lexical grounding are checked locally. They are not all enforced by the wire schema. The Reddit response contains writer and drugs. The WebMD response contains drugs, with demographics and condition-treated values supplied separately by the profile rules above.
