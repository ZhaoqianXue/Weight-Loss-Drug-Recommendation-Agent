"""RQ2 convenience join: FAERS primary-configuration PT counts next to patient-side PT counts from the v9 graph,
under the same aligned window. Patient counting follows project-document Section 3.6: a PT counts once per record
within a stratum; the denominator is the number of records with at least one side effect in that stratum.

The patient side is read from the graph as built (kg-v9-0.1.0); its MedDRA coding is not yet validated. Rerun
this script if the graph changes.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from common import APPROVAL, FROZEN, ING_START, KG, OUT, WINDOW_END, load_meddra, read_csv, sha256

ALL_START = min(APPROVAL.values())


def patient_strata() -> pd.DataFrame:
    """One row per (side-effect edge, stratum) with an in_window flag; levels all/ingredient/brand/unspecified."""
    con = sqlite3.connect(f"file:{KG}?mode=ro", uri=True)
    e = pd.read_sql("""select e.platform, e.record_id, e.date, d.ingredient, d.brand, s.pt_code
                       from edge_side_effect e join drug d on d.id = e.drug_id
                       join side_effect s on s.id = e.side_effect_id""", con)
    con.close()
    e["pt_code"] = e.pt_code.fillna("")
    e["brand"] = e.brand.str.lower()
    rows = [e.assign(level="all", s_ing="all", s_brand="all", start=ALL_START)]
    ing = e[e.ingredient != "unspecified"]
    rows.append(ing.assign(level="ingredient", s_ing=ing.ingredient, s_brand="",
                           start=ing.ingredient.map(ING_START)))
    br = ing[ing.brand != "unspecified"]
    rows.append(br.assign(level="brand", s_ing=br.ingredient, s_brand=br.brand, start=br.brand.map(APPROVAL)))
    un = ing[ing.brand == "unspecified"]
    rows.append(un.assign(level="unspecified", s_ing=un.ingredient, s_brand="unspecified",
                          start=un.ingredient.map(ING_START)))
    st = pd.concat(rows, ignore_index=True)
    st["in_window"] = (st.date >= st.start) & (st.date <= WINDOW_END)
    return st


def main() -> None:
    st = patient_strata()
    e = st[st.level == "all"]
    k = ["level", "s_ing", "s_brand"]
    rec = ["platform", "record_id"]
    excl = (st.drop_duplicates(k + rec).groupby(k + ["platform"])
            .agg(records=("in_window", "size"), records_in_window=("in_window", "sum")).reset_index())
    st = st[st.in_window]
    both = pd.concat([st, st.assign(platform="combined")], ignore_index=True)
    den = both.drop_duplicates(k + rec).groupby(k + ["platform"]).size().rename("denominator")
    num = (both[both.pt_code != ""].drop_duplicates(k + rec + ["pt_code"])
           .groupby(k + ["platform", "pt_code"]).size().rename("n_records"))
    pat = num.reset_index().merge(den.reset_index(), on=k + ["platform"])
    wide = pat.pivot_table(index=k + ["pt_code"], columns="platform", values=["n_records", "denominator"],
                           aggfunc="first")
    wide.columns = [f"{src}_{v}" for v, src in wide.columns]
    wide = wide.reset_index().rename(columns={"s_ing": "ingredient", "s_brand": "brand"})
    # Denominators must be present on every PT row of a stratum, including PTs one source never reports.
    dw = den.reset_index().pivot_table(index=k, columns="platform", values="denominator").add_suffix("_denominator")
    dw = dw.reset_index().rename(columns={"s_ing": "ingredient", "s_brand": "brand"})

    f = read_csv(FROZEN / "faers_pt_counts.csv.gz")
    f = f[f.config == "primary"]
    fd = read_csv(FROZEN / "faers_denominators.csv")
    fd = fd[fd.config == "primary"][["level", "ingredient", "brand", "n_reports_with_mapped_pt"]].rename(
        columns={"n_reports_with_mapped_pt": "faers_denominator"})
    f = f[["level", "ingredient", "brand", "pt_code", "n_reports"]].rename(columns={"n_reports": "faers_n_reports"})
    keys = ["level", "ingredient", "brand", "pt_code"]
    out = f.merge(wide.drop(columns=[c for c in wide.columns if c.endswith("_denominator")]), on=keys, how="outer")
    out = out.merge(fd, on=["level", "ingredient", "brand"], how="left").merge(
        dw, on=["level", "ingredient", "brand"], how="left")
    pt, _ = load_meddra()
    out = out.merge(pt, on="pt_code", how="left")
    for c in ["faers_n_reports", "reddit_n_records", "webmd_n_records", "combined_n_records"]:
        out[c] = pd.to_numeric(out[c]).fillna(0).astype(int)
    for c in ["faers_denominator", "reddit_denominator", "webmd_denominator", "combined_denominator"]:
        out[c] = pd.to_numeric(out[c]).fillna(0).astype(int)
    cols = keys + ["pt_name", "pt_primary_soc_code", "faers_n_reports", "faers_denominator", "reddit_n_records",
                   "reddit_denominator", "webmd_n_records", "webmd_denominator", "combined_n_records",
                   "combined_denominator"]
    out = out[cols].sort_values(keys)
    out.to_csv(FROZEN / "rq2_faers_vs_patient_primary.csv.gz", index=False)
    excl.rename(columns={"s_ing": "ingredient", "s_brand": "brand"}).to_csv(
        OUT / "patient_window_effect.csv", index=False)
    meta = {"kg": str(KG.relative_to(KG.parents[4])), "kg_sha256": sha256(KG), "window_end": WINDOW_END,
            "rows": len(out),
            "patient_records_with_side_effect": int(e.drop_duplicates(rec).shape[0]),
            "patient_records_in_window_all_level": int(st[st.level == "all"].drop_duplicates(rec).shape[0])}
    (OUT / "patient_join_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
