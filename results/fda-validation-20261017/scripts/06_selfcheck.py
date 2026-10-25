"""Self-check of the frozen tables: internal consistency, an independent recount of selected strata straight from
the processed tables (without the build code), and a reconciliation with the acquisition's stratum totals."""
from __future__ import annotations

import json

import pandas as pd

from common import APPROVAL, FROZEN, OUT, PROC, read_csv


def main() -> None:
    res = {}
    den = read_csv(FROZEN / "faers_denominators.csv")
    for c in ["n_reports", "n_reports_with_mapped_pt"]:
        den[c] = den[c].astype(int)
    pc = read_csv(FROZEN / "faers_pt_counts.csv.gz")
    pc["n_reports"] = pc.n_reports.astype(int)
    pc["stratum_reports"] = pc.stratum_reports.astype(int)
    res["pt_count_exceeds_denominator"] = int((pc.n_reports > pc.stratum_reports).sum())
    flow = read_csv(OUT / "flow_reports.csv")
    prim_all = int(den[(den.config == "primary") & (den.level == "all")].n_reports.iloc[0])
    res["flow_final_equals_primary_all_denominator"] = int(flow.reports.iloc[-1]) == prim_all
    res["flow_arithmetic_ok"] = all(int(a) - int(r) == int(b) for a, b, r in zip(
        flow.reports[:-1], flow.reports[1:], flow.removed[1:]))
    r3 = read_csv(FROZEN / "faers_rq3_denominators.csv")
    r3["n_reports"] = r3.n_reports.astype(int)
    s3 = r3.groupby(["level", "ingredient", "brand"]).n_reports.sum().reset_index().merge(
        den[den.config == "primary"], on=["level", "ingredient", "brand"])
    res["rq3_cells_sum_to_stratum"] = bool((s3.n_reports_x == s3.n_reports_y).all())
    strata = read_csv(FROZEN / "faers_report_strata.csv.gz")
    reps = read_csv(FROZEN / "faers_reports.csv.gz")
    res["report_strata_ids_subset_of_reports"] = bool(set(strata.safetyreportid) <= set(reps.safetyreportid))
    res["in_primary_flag_matches"] = int(reps.in_primary_analysis.eq("True").sum()) == prim_all

    # Independent recount from the processed tables: brand strata, primary configuration, Nausea.
    te = read_csv(PROC / "target_drug_entries.csv.gz")
    rp = read_csv(PROC / "reports.csv.gz", usecols=["safetyreportid", "receivedate"])
    rx = read_csv(PROC / "reactions.csv.gz")
    nausea = set(rx.loc[rx.reactionmeddrapt.str.lower() == "nausea", "safetyreportid"])
    combo = te.medicinalproduct.str.lower().str.contains("xultophy") | (
        (te.medicinalproduct + te.activesubstancename).str.lower().str.contains("degludec"))
    rec = []
    t = te[~combo].merge(rp)
    for b, d in APPROVAL.items():
        m = t[(t.raw_brands == b) & t.role.isin(["suspect", "interacting"]) & (t.receivedate >= d.replace("-", ""))
              & ~t.medicinalproduct.str.lower().str.contains("compound|blind|placebo")]
        ids = set(m.safetyreportid)
        f = den[(den.config == "primary") & (den.brand == b)].n_reports.iloc[0]
        fn = pc[(pc.config == "primary") & (pc.brand == b) & (pc.pt_name == "Nausea")].n_reports.iloc[0]
        rec.append({"brand": b, "independent_reports": len(ids), "frozen_reports": int(f),
                    "independent_nausea": len(ids & nausea), "frozen_nausea": int(fn)})
    rec = pd.DataFrame(rec)
    res["independent_brand_recount"] = rec.to_dict(orient="records")
    res["independent_brand_recount_all_equal"] = bool(
        ((rec.independent_reports == rec.frozen_reports) & (rec.independent_nausea == rec.frozen_nausea)).all())

    # Reconciliation with the acquisition's brand strata (extended window).
    acq = read_csv(PROC / "aggregates/stratum_report_totals.csv")
    acq = acq[(acq.window == "extended_receivedate_to_20260630") & (acq.stratum_type == "brand_raw_name")]
    fl = read_csv(OUT / "flow_by_stratum.csv")
    rows = []
    for b in APPROVAL:
        a_any = int(acq[(acq.stratum == b) & (acq.role_scope == "any_role")].n_reports.iloc[0])
        f = fl[fl.brand == b].iloc[0]
        rows.append({"brand": b, "acquisition_any_role": a_any, "validated_naming_brand": int(f.reports_naming_brand),
                     "difference": a_any - int(f.reports_naming_brand),
                     "before_approval_removed": int(f.before_approval_removed),
                     "validated_in_window_any_role": int(f.in_window_any_role)})
    res["acquisition_reconciliation"] = rows
    # Clinical-event restriction (08_nonclinical.py): denominators shrink, numerators never exceed them.
    dc = read_csv(FROZEN / "faers_denominators_clinical.csv")
    res["clinical_denominator_not_above_total"] = bool(
        (dc.n_reports_with_clinical_pt.astype(int) <= dc.n_reports.astype(int)).all())
    jc = read_csv(FROZEN / "rq2_faers_vs_patient_primary_clinical.csv.gz")
    res["clinical_join_has_no_flagged_pt"] = bool(
        not set(jc.pt_code) & set(pc.loc[pc.nonclinical_group != "", "pt_code"]))
    res["clinical_join_numerators_within_denominators"] = bool(all(
        (jc[n].astype(int) <= jc[d].astype(int)).all() for n, d in [
            ("faers_n_reports", "faers_denominator"), ("reddit_n_records", "reddit_denominator"),
            ("webmd_n_records", "webmd_denominator"), ("combined_n_records", "combined_denominator")]))
    (OUT / "selfcheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
