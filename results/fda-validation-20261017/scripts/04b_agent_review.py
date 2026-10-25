"""Step 2 (review verdicts): Claude's agent review of the 135 sampled spellings, recorded per sample_id.
This is an agent review, not a human gold standard. Default verdict for unlisted rows: rule assignment correct."""
from __future__ import annotations

import pandas as pd

from common import REVIEW, read_csv

BLINDED_NOTE = "blinded or placebo-controlled trial drug: the patient may not have received the study drug; " \
               "excluded from the primary analysis by design, kept in the broad-identity sensitivity analysis"
NOTES = {
    "S039": ("correct_rule_wording_odd", "reporter wrote 'insulin Saxenda'; the brand token is Saxenda, so the "
                                         "rule assigns Saxenda; Saxenda is not an insulin, wording is the reporter's"),
    "S050": ("correct_exposure_uncertain", "brand correct, but the reporter adds that the drug could not be "
                                           "obtained at the time; exposure may not have occurred"),
    "S090": ("correct", "compounded product named after a brand; counted as compounded semaglutide with brand "
                        "unspecified, as in the patient-side extraction rule"),
    "S092": ("uncertain_two_ingredients", "one entry lists cyanocobalamin, semaglutide and tirzepatide; counted "
                                          "for both ingredients as compounded"),
    "S099": ("uncertain_two_ingredients", "same entry as S092 seen from the tirzepatide stratum"),
    "S114": ("correct_brand_missed", "misspelled Victoza; ingredient assigned through the active-substance "
                                     "field, brand left unspecified (no fuzzy matching by design)"),
    "S120": ("correct", "trial formulation (PDS290 device) of semaglutide; ingredient correct, no brand"),
    "S122": ("correct", "reporter states the brand is not specified"),
    "S129": ("correct_formulation_unapproved", "ingredient correct, but sublingual tirzepatide is not an "
                                               "approved product; likely compounded, not flagged because no "
                                               "additive is listed"),
    "S132": ("correct_brand_missed", "misspelled Mounjaro; ingredient correct, brand left unspecified"),
    "S134": ("correct", "two brands of one ingredient in one entry; ingredient level only"),
    "S135": ("correct", "two brands of one ingredient in one entry; ingredient level only"),
}


def main() -> None:
    s = read_csv(REVIEW / "identity_sample.csv")
    verdicts, notes = [], []
    for sid, cls in zip(s.sample_id, s.identity_class):
        if sid in NOTES:
            v, n = NOTES[sid]
        elif cls == "blinded":
            v, n = "identity_uncertain_by_design", BLINDED_NOTE
        else:
            v, n = "correct", ""
        verdicts.append(v)
        notes.append(n)
    s["agent_verdict"] = verdicts
    s["agent_note"] = notes
    s["reviewer"] = "Claude (agent review, not human gold)"
    s["user_check"] = ""
    s.to_csv(REVIEW / "agent_identity_review.csv", index=False)
    print(s.agent_verdict.value_counts().to_string())


if __name__ == "__main__":
    main()
