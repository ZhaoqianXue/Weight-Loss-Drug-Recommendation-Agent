"""Step 1 (continued): compare the independent raw rebuild with the processed tables row by row, and list drug
entries whose raw fields contain a study-drug name only as a substring (missed by token matching)."""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, PROC, read_csv

INT = OUT / "integrity"


def compare(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> dict:
    m = a.merge(b, on=keys, how="outer", suffixes=("_raw", "_proc"), indicator=True)
    res = {"rows_raw": len(a), "rows_processed": len(b), "only_raw": int((m._merge == "left_only").sum()),
           "only_processed": int((m._merge == "right_only").sum()), "value_mismatches": {}}
    both = m[m._merge == "both"]
    for c in cols:
        res["value_mismatches"][c] = int((both[f"{c}_raw"] != both[f"{c}_proc"]).sum())
    res["identical"] = res["only_raw"] == res["only_processed"] == 0 and not any(res["value_mismatches"].values())
    return res


def main() -> None:
    out = {}
    rr = read_csv(INT / "raw_reports.csv.gz")
    pr = read_csv(PROC / "reports.csv.gz")
    cols = [c for c in rr.columns if c != "safetyreportid"]
    out["reports"] = compare(rr, pr[["safetyreportid"] + cols], ["safetyreportid"], cols)

    rx = read_csv(INT / "raw_reactions.csv.gz")
    px = read_csv(PROC / "reactions.csv.gz")
    k = ["safetyreportid", "reaction_index"]
    out["reactions"] = compare(rx, px, k, ["reactionmeddrapt", "reactionmeddraversionpt", "reactionoutcome"])

    ce = read_csv(INT / "raw_candidate_entries.csv.gz")
    te = read_csv(PROC / "target_drug_entries.csv.gz")
    tok = ce[ce.token_ingredients != ""].assign(ingredient=lambda d: d.token_ingredients.str.split(";")).explode(
        "ingredient")
    ecols = ["medicinalproduct", "activesubstancename", "drugcharacterization", "actiondrug", "drugindication",
             "drugadministrationroute"]
    k = ["safetyreportid", "drug_index", "ingredient"]
    out["target_entries"] = compare(tok[k + ecols], te[k + ecols], k, ecols)

    # Substring-only matches in the reporter's raw fields: entries the token rule did not treat as study drugs.
    ce["sub"] = ce.substring_raw_ingredients.str.split(";").map(lambda x: set(filter(None, x)))
    ce["tok"] = ce.token_ingredients.str.split(";").map(lambda x: set(filter(None, x)))
    extra = ce[[bool(s - t) for s, t in zip(ce["sub"], ce["tok"])]].copy()
    extra["substring_only_ingredients"] = [";".join(sorted(s - t)) for s, t in zip(extra["sub"], extra["tok"])]
    extra = extra.drop(columns=["sub", "tok"])
    extra.to_csv(INT / "substring_only_entries.csv", index=False)
    # Would any of them change a report's role scope? (only entry of that ingredient that is suspect/interacting)
    te_s = te.assign(prim=te.role.isin(["suspect", "interacting"]))
    prim_ing = set(zip(te_s.loc[te_s.prim, "safetyreportid"], te_s.loc[te_s.prim, "ingredient"]))
    any_ing = set(zip(te.safetyreportid, te.ingredient))
    ex = extra.assign(ingredient=[x.split(";") for x in extra.substring_only_ingredients]).explode("ingredient")
    ex["prim"] = ex.drugcharacterization.isin(["1", "3"])
    out["substring_only"] = {
        "entries": len(extra), "reports": int(extra.safetyreportid.nunique()),
        "top_names": extra.medicinalproduct.astype(str).str.upper().value_counts().head(25).to_dict(),
        "report_ingredient_pairs_not_otherwise_matched": int(sum((s, i) not in any_ing
                                                                 for s, i in zip(ex.safetyreportid, ex.ingredient))),
        "would_add_primary_role_to_pair": int(sum(p and (s, i) not in prim_ing
                                                  for s, i, p in zip(ex.safetyreportid, ex.ingredient, ex.prim))),
    }
    (INT / "compare_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
