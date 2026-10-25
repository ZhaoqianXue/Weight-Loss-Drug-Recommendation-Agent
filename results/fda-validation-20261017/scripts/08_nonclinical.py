"""RQ2 clinical-event restriction (user 2026-10-17): flag FAERS PTs that are not symptoms, using official MedDRA
groups on each PT's primary path, never hand-picked terms.

Groups on each PT's primary path (names checked against the 29.1 hierarchy files before use):
  product_issue        SOC  "Product issues"
  medication_error     HLGT "Medication errors and other product use errors and issues"
  off_label_misuse     HLGT "Off label uses and intentional product misuses/use issues"
  response_interaction HLT  "Therapeutic and nontherapeutic responses" (drug ineffective etc.) and HLT "Interactions"
                       (the sibling HLT "Withdrawal and rebound effects" is a clinical event and is kept)
  overdose_underdose   HLGT "Overdoses and underdoses NEC"
  pregnancy_exposure   HLT  "Exposures associated with pregnancy, delivery and lactation"
All six groups confirmed by the user on 2026-10-17. The same rule applies to the patient side.

  python 08_nonclinical.py inspect   check the files, list the groups and the PTs they would flag (no frozen writes)
  python 08_nonclinical.py apply     write the flags, clinical-only denominators and the clinical-only RQ2 join
"""
from __future__ import annotations

import importlib
import json
import sys

import pandas as pd

from common import FROZEN, MEDDRA, OUT, load_meddra, read_csv

GROUPS = [  # (group, hierarchy column, official name); first match wins
    ("product_issue", "soc_name", "Product issues"),
    ("medication_error", "hlgt_name", "Medication errors and other product use errors and issues"),
    ("off_label_misuse", "hlgt_name", "Off label uses and intentional product misuses/use issues"),
    ("response_interaction", "hlt_name", "Therapeutic and nontherapeutic responses"),
    ("response_interaction", "hlt_name", "Interactions"),
    ("overdose_underdose", "hlgt_name", "Overdoses and underdoses NEC"),
    ("pregnancy_exposure", "hlt_name", "Exposures associated with pregnancy, delivery and lactation"),
]
MDHIER = ["pt_code", "hlt_code", "hlgt_code", "soc_code", "pt_name", "hlt_name", "hlgt_name", "soc_name",
          "soc_abbrev", "null_field", "pt_soc_code", "primary_soc_fg"]


def load_hierarchy() -> pd.DataFrame:
    missing = [f for f in ["mdhier.asc", "hlgt.asc", "soc.asc"] if not (MEDDRA / f).exists()]
    if missing:
        sys.exit(f"STOP: missing MedDRA files in {MEDDRA}: {missing}")
    for f in ["mdhier.asc", "hlgt.asc", "soc.asc"]:
        raw = (MEDDRA / f).read_bytes()
        if b"\xef\xbf\xbd" in raw:
            sys.exit(f"STOP: {f} contains U+FFFD replacement bytes (re-encoded download)")
        raw.decode("cp1252")
    h = pd.read_csv(MEDDRA / "mdhier.asc", sep="$", header=None, dtype=str, keep_default_na=False, quoting=3,
                    encoding="cp1252").iloc[:, :12]
    h.columns = MDHIER
    prim = h[h.primary_soc_fg == "Y"]
    pt, _ = load_meddra()
    checks = {
        "mdhier_rows": len(h), "primary_rows": len(prim), "pts_in_pt_asc": len(pt),
        "pts_with_exactly_one_primary_path": int((prim.pt_code.value_counts() == 1).sum()),
        "primary_soc_agrees_with_pt_asc": int((prim.merge(pt, on="pt_code").soc_code
                                               == prim.merge(pt, on="pt_code").pt_primary_soc_code).sum()),
    }
    soc = pd.read_csv(MEDDRA / "soc.asc", sep="$", header=None, dtype=str, keep_default_na=False, quoting=3,
                      encoding="cp1252").iloc[:, :2].set_axis(["soc_code", "soc_name"], axis=1)
    hlgt = pd.read_csv(MEDDRA / "hlgt.asc", sep="$", header=None, dtype=str, keep_default_na=False, quoting=3,
                       encoding="cp1252").iloc[:, :2].set_axis(["hlgt_code", "hlgt_name"], axis=1)
    checks["soc_names_agree"] = int(prim.merge(soc, on="soc_code", suffixes=("", "_f"))
                                    .eval("soc_name == soc_name_f").sum())
    checks["hlgt_names_agree"] = int(prim.merge(hlgt, on="hlgt_code", suffixes=("", "_f"))
                                     .eval("hlgt_name == hlgt_name_f").sum())
    for g, col, name in GROUPS:
        checks[f"name_found:{name}"] = bool((prim[col] == name).any())
    print(json.dumps(checks, indent=1))
    ok = (checks["pts_with_exactly_one_primary_path"] == checks["pts_in_pt_asc"] == checks["primary_rows"]
          == checks["primary_soc_agrees_with_pt_asc"] == checks["soc_names_agree"] == checks["hlgt_names_agree"]
          and all(checks[f"name_found:{name}"] for _, _, name in GROUPS))
    if not ok:
        sys.exit("STOP: hierarchy checks failed; group names or files need review")
    prim = prim.copy()
    prim["nonclinical_group"] = ""
    for g, col, name in GROUPS:
        prim.loc[(prim[col] == name) & (prim.nonclinical_group == ""), "nonclinical_group"] = g
    return prim[["pt_code", "pt_name", "soc_code", "soc_name", "hlgt_code", "hlgt_name", "hlt_code", "hlt_name",
                 "nonclinical_group"]], checks


def main(mode: str) -> None:
    prim, checks = load_hierarchy()
    pc = read_csv(FROZEN / "faers_pt_counts.csv.gz")
    allp = pc[(pc.config == "primary") & (pc.level == "all")][["pt_code", "n_reports"]].astype({"n_reports": int})
    flagged = prim[prim.nonclinical_group != ""].merge(allp, on="pt_code", how="left").fillna({"n_reports": 0})
    flagged["n_reports"] = flagged.n_reports.astype(int)
    pairs = int(allp.n_reports.sum())
    summary = {
        "checks": checks,
        "flagged_pts_in_meddra": flagged.groupby("nonclinical_group").size().to_dict(),
        "flagged_pts_seen_in_faers_primary": flagged[flagged.n_reports > 0].groupby("nonclinical_group").size().to_dict(),
        "faers_primary_report_pt_pairs": pairs,
        "flagged_pairs_by_group": flagged.groupby("nonclinical_group").n_reports.sum().to_dict(),
    }
    summary["flagged_pairs_share"] = round(sum(summary["flagged_pairs_by_group"].values()) / pairs, 4)
    # What stays: real injuries that share SOC 10022117 with medication errors.
    inj = prim[(prim.soc_code == "10022117") & (prim.nonclinical_group == "")].merge(allp, on="pt_code")
    summary["kept_in_soc_10022117_top"] = inj.sort_values("n_reports", ascending=False).head(15)[
        ["pt_name", "hlgt_name", "n_reports"]].to_dict(orient="records")
    seen = flagged[flagged.n_reports > 0].sort_values("n_reports", ascending=False)
    seen[["nonclinical_group", "pt_code", "pt_name", "hlt_name", "hlgt_name", "soc_name", "n_reports"]].to_csv(
        OUT / "nonclinical_pt_candidates.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=1, ensure_ascii=False))
    (OUT / "nonclinical_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if mode != "apply":
        return

    # Flags on the PT-level frozen tables (no rows removed).
    prim.to_csv(FROZEN / "meddra_pt_hierarchy.csv", index=False)
    flag = prim[["pt_code", "nonclinical_group"]]
    for name in ["faers_pt_counts.csv.gz", "faers_rq3_pt_by_age_sex.csv.gz", "rq2_faers_vs_patient_primary.csv.gz"]:
        t = read_csv(FROZEN / name).drop(columns=["nonclinical_group"], errors="ignore")
        t.merge(flag, on="pt_code", how="left").fillna({"nonclinical_group": ""}).to_csv(FROZEN / name, index=False)

    # Clinical-only denominators: reports with at least one clinical PT in the stratum.
    rpt = read_csv(FROZEN / "faers_report_pts.csv.gz").merge(flag, on="pt_code", how="left")
    clin_ids = set(rpt.loc[rpt.nonclinical_group.fillna("") == "", "safetyreportid"])
    st = read_csv(FROZEN / "faers_report_strata.csv.gz")
    rows = []
    for cfg in ["primary", "sensitivity_any_role", "sensitivity_broad_identity"]:
        s = st[st[f"in_{cfg}"] == "True"]
        s = pd.concat([s, s.drop_duplicates("safetyreportid").assign(level="all", ingredient="all", brand="all")])
        g = s.assign(clin=s.safetyreportid.isin(clin_ids)).groupby(["level", "ingredient", "brand"])
        d = g.agg(n_reports=("safetyreportid", "size"), n_reports_with_clinical_pt=("clin", "sum")).reset_index()
        rows.append(d.assign(config=cfg))
    den = pd.concat(rows)[["config", "level", "ingredient", "brand", "n_reports", "n_reports_with_clinical_pt"]]
    den.to_csv(FROZEN / "faers_denominators_clinical.csv", index=False)
    # Patient side, same rule: records with at least one side effect that is not flagged (uncoded ones count).
    ps = importlib.import_module("05_patient_join").patient_strata()
    ps = ps[ps.in_window].merge(flag, on="pt_code", how="left")
    ps = ps[ps.nonclinical_group.fillna("") == ""]
    ps = pd.concat([ps, ps.assign(platform="combined")], ignore_index=True)
    k, rec = ["level", "s_ing", "s_brand"], ["platform", "record_id"]
    pden = (ps.drop_duplicates(k + rec).groupby(k + ["platform"]).size().rename("d").reset_index()
            .pivot_table(index=k, columns="platform", values="d", fill_value=0).astype(int)
            .add_suffix("_denominator").reset_index().rename(columns={"s_ing": "ingredient", "s_brand": "brand"}))
    pden.to_csv(FROZEN / "patient_denominators_clinical.csv", index=False)
    j = read_csv(FROZEN / "rq2_faers_vs_patient_primary.csv.gz")
    j = j[j.nonclinical_group == ""].drop(columns=["faers_denominator", "reddit_denominator", "webmd_denominator",
                                                   "combined_denominator"])
    dp = den[den.config == "primary"][["level", "ingredient", "brand", "n_reports_with_clinical_pt"]].rename(
        columns={"n_reports_with_clinical_pt": "faers_denominator"})
    j = j.merge(dp, on=["level", "ingredient", "brand"], how="left").merge(
        pden, on=["level", "ingredient", "brand"], how="left")
    for c in ["reddit_denominator", "webmd_denominator", "combined_denominator"]:
        j[c] = j[c].fillna(0).astype(int)
    j = j[["level", "ingredient", "brand", "pt_code", "pt_name", "pt_primary_soc_code", "faers_n_reports",
           "faers_denominator", "reddit_n_records", "reddit_denominator", "webmd_n_records", "webmd_denominator",
           "combined_n_records", "combined_denominator"]]
    j.to_csv(FROZEN / "rq2_faers_vs_patient_primary_clinical.csv.gz", index=False)
    print("applied:", len(j), "clinical rows;", len(clin_ids), "reports with a clinical PT")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "inspect")
