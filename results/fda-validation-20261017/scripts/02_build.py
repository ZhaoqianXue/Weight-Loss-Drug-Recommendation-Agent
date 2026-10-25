"""Steps 2-8: drug identity, combination exclusion, MedDRA 29.1, duplicates, roles, time window, population
fields, hospitalization and action taken. Builds the frozen FAERS comparison tables and the exclusion flow.

Inputs are the processed tables (checked against the raw pages in step 1) and the MedDRA 29.1 files.
Nothing upstream is modified.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from common import (ACTION_LABEL, AGE_UNIT_YEARS, APPROVAL, BLINDED, BRAND_TO_ING, COMBINATION, COMPOUNDED,
                    FROZEN, ING_START, INGREDIENTS, MAIN_CLASSES, OUT, PRIMARY_ROLES, PROC, SENSITIVITY_CLASSES,
                    TOKEN, WINDOW_END, age_group, load_meddra, map_terms, read_csv)

CONFIGS = {  # name -> (entry identity classes, roles or None for any role)
    "primary": (MAIN_CLASSES, PRIMARY_ROLES),
    "sensitivity_any_role": (MAIN_CLASSES, None),
    "sensitivity_broad_identity": (MAIN_CLASSES + SENSITIVITY_CLASSES, PRIMARY_ROLES),
}


def classify_entries(te: pd.DataFrame) -> pd.DataFrame:
    mp = te.medicinalproduct.str.lower()
    raw_text = mp + " " + te.activesubstancename.str.lower()
    mp_tok = mp.map(lambda x: set(TOKEN.findall(x)))
    raw_tok = raw_text.map(lambda x: set(TOKEN.findall(x)))
    cls, brand = [], []
    for ing, text, mt, rt in zip(te.ingredient, raw_text, mp_tok, raw_tok):
        brands = sorted(mt & set(INGREDIENTS[ing]))
        if COMBINATION.search(text):
            c = "combination"
        elif BLINDED.search(text):
            c = "blinded"
        elif COMPOUNDED.search(text):
            c = "compounded"
        elif len(brands) == 1:
            c = "brand"
        elif len(brands) > 1:
            c = "multiple_brands"
        elif ing in rt or rt & set(INGREDIENTS[ing]):
            c = "ingredient_name"
        else:
            c = "harmonized_only"
        cls.append(c)
        brand.append(brands[0] if c == "brand" else "")
    te = te.assign(identity_class=cls, brand=brand)
    te["primary_role"] = te.role.isin(PRIMARY_ROLES)
    return te


def strata_rows(te: pd.DataFrame) -> pd.DataFrame:
    """One row per (entry, stratum): ingredient level, brand level, and the brand-unspecified level (ingredient
    present, but no included entry of that ingredient names a brand anywhere in the report)."""
    inc = te[te.identity_class != "combination"]
    ing = inc.assign(level="ingredient", stratum_brand="")
    br = inc[inc.identity_class == "brand"].assign(level="brand", stratum_brand=lambda d: d.brand)
    branded = set(zip(br.safetyreportid, br.ingredient))
    unspec_mask = [(s, i) not in branded for s, i in zip(inc.safetyreportid, inc.ingredient)]
    un = inc[unspec_mask].assign(level="unspecified", stratum_brand="unspecified")
    return pd.concat([ing, br, un], ignore_index=True)


def window_start(level: str, ingredient: str, brand: str) -> str:
    return (APPROVAL[brand] if level == "brand" else ING_START[ingredient]).replace("-", "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    rep = read_csv(PROC / "reports.csv.gz")
    te = read_csv(PROC / "target_drug_entries.csv.gz")
    rx = read_csv(PROC / "reactions.csv.gz")
    pt, llt = load_meddra()
    flow, facts = [], {}

    # ---------------- Step 2: drug identity from raw fields ----------------
    te = classify_entries(te)
    facts["entries_total"] = len(te)
    facts["entries_by_identity_class"] = te.identity_class.value_counts().to_dict()
    facts["entries_by_role"] = te.role.value_counts().to_dict()
    facts["reports_by_identity_class"] = te.groupby("identity_class").safetyreportid.nunique().to_dict()
    oz = te[te.brand == "ozempic"]
    facts["ozempic_raw_entries_harmonized_to_NDA213051_oral"] = int(
        (oz.openfda_application_number.str.contains("NDA213051") & oz.openfda_route.str.contains("ORAL")).sum())
    facts["ozempic_raw_entries"] = len(oz)
    route = te[te.identity_class == "brand"].groupby("brand").drugadministrationroute.value_counts().unstack(fill_value=0)
    facts["brand_entries_by_raw_route_code"] = {b: {k: int(v) for k, v in r.items() if v} for b, r in route.iterrows()}
    te.drop(columns=[c for c in te.columns if c.startswith("openfda_")]).to_csv(
        OUT / "entries_classified.csv.gz", index=False)

    # Report-level exclusion flow.
    ids_all = set(rep.safetyreportid)
    flow.append({"step": "0_unique_reports", "reports": len(ids_all), "removed": 0,
                 "reason": "unique safetyreportid in openFDA release 2026-07-30, receivedate <= 2026-06-30"})
    non_combo = set(te.loc[te.identity_class != "combination", "safetyreportid"])
    combo_ids = set(te.loc[te.identity_class == "combination", "safetyreportid"])
    s1 = ids_all & non_combo
    facts["combination_reports_any"] = len(combo_ids)
    facts["combination_reports_only_combination"] = len(combo_ids - non_combo)
    facts["combination_reports_with_other_study_drug"] = len(combo_ids & non_combo)
    flow.append({"step": "1_combination_only_removed", "reports": len(s1), "removed": len(ids_all - s1),
                 "reason": "every study-drug entry is the insulin degludec/liraglutide combination (Xultophy); "
                           "in reports with another study drug only the Xultophy entry is dropped"})
    main_ids = set(te.loc[te.identity_class.isin(MAIN_CLASSES), "safetyreportid"])
    s2 = s1 & main_ids
    flow.append({"step": "2_identity_unconfirmed_removed", "reports": len(s2), "removed": len(s1 - s2),
                 "reason": "remaining study-drug entries are blinded trial drugs or matched only through openFDA "
                           "harmonized fields (kept for the broad-identity sensitivity analysis)"})

    # ---------------- Steps 5-6: strata, roles and window ----------------
    st = strata_rows(te).merge(rep[["safetyreportid", "receivedate"]], on="safetyreportid")
    keys = ["safetyreportid", "level", "ingredient", "stratum_brand"]
    st["start"] = [window_start(l, i, b) for l, i, b in zip(st.level, st.ingredient, st.stratum_brand)]
    st["in_window"] = (st.receivedate >= st.start) & (st.receivedate <= WINDOW_END.replace("-", ""))
    for name, (classes, roles) in CONFIGS.items():
        ok = st.identity_class.isin(classes)
        if roles is not None:
            ok &= st.primary_role
        st[f"entry_{name}"] = ok
    st["is_compounded"] = st.identity_class == "compounded"
    agg = st.groupby(keys).agg(
        in_window=("in_window", "first"), compounded=("is_compounded", "any"),
        **{f"in_{n}": (f"entry_{n}", "any") for n in CONFIGS},
    ).reset_index()
    for n in CONFIGS:
        agg[f"in_{n}"] &= agg.in_window

    def actions(mask_col):
        sub = st[st[mask_col]]
        return sub.groupby(keys).actiondrug.agg(lambda s: ";".join(sorted(set(s)))).rename(mask_col)
    act = pd.concat([actions("entry_primary"), actions("entry_sensitivity_any_role")], axis=1).reset_index()
    act.columns = keys + ["actiondrug_primary", "actiondrug_any_role"]
    ind = (st[st.entry_primary].groupby(keys).drugindication
           .agg(lambda s: "|".join(sorted({x for x in s if x}))).rename("drugindication_primary").reset_index())
    agg = agg.merge(act, on=keys, how="left").merge(ind, on=keys, how="left").fillna(
        {"actiondrug_primary": "", "actiondrug_any_role": "", "drugindication_primary": ""})

    # Window step in the flow: reports none of whose main-identity strata fall in the window.
    ing_rows = agg[agg.level == "ingredient"]
    main_ing = st[(st.level == "ingredient") & st.identity_class.isin(MAIN_CLASSES)][keys].drop_duplicates()
    main_ing = main_ing.merge(ing_rows[keys + ["in_window"]], on=keys)
    s3 = s2 & set(main_ing.loc[main_ing.in_window, "safetyreportid"])
    flow.append({"step": "3_outside_time_window_removed", "reports": len(s3), "removed": len(s2 - s3),
                 "reason": "receivedate before the first FDA approval of every study ingredient in the report "
                           "(upper bound 2026-06-30 removes none)"})
    s4 = s3 & set(ing_rows.loc[ing_rows.in_primary, "safetyreportid"])
    flow.append({"step": "4_no_suspect_or_interacting_role_removed", "reports": len(s4), "removed": len(s3 - s4),
                 "reason": "study drug reported only as concomitant (or role missing); kept in the any-role "
                           "sensitivity analysis"})
    facts["reports_any_role_in_window"] = len(s3)
    facts["reports_broad_identity_primary"] = int(ing_rows.loc[ing_rows.in_sensitivity_broad_identity,
                                                               "safetyreportid"].nunique())

    # ---------------- Step 3: MedDRA 29.1 ----------------
    mapping = map_terms(rx.reactionmeddrapt, pt, llt)
    rx = rx.merge(mapping[["raw_term", "pt_code", "map_method"]], left_on="reactionmeddrapt", right_on="raw_term")
    mm = rx.map_method.value_counts()
    facts["reactions_total"] = len(rx)
    facts["reactions_by_map_method"] = mm.to_dict()
    facts["reactions_unmatched_terms"] = rx.loc[rx.map_method == "unmatched", "reactionmeddrapt"].value_counts().to_dict()
    facts["reactions_by_meddra_version"] = rx.reactionmeddraversionpt.replace("", "missing").value_counts().sort_index().to_dict()
    cnt = rx.groupby("reactionmeddrapt").size().rename("n_reactions").reset_index()
    mapping.merge(cnt, left_on="raw_term", right_on="reactionmeddrapt").drop(columns="reactionmeddrapt").sort_values(
        "n_reactions", ascending=False).to_csv(FROZEN / "meddra_term_mapping.csv", index=False)
    rpt = (rx[rx.pt_code != ""].groupby(["safetyreportid", "pt_code"])
           .map_method.agg(lambda s: ";".join(sorted(set(s)))).reset_index()
           .merge(pt, on="pt_code"))
    facts["report_pt_pairs"] = len(rpt)
    facts["report_pt_pairs_collapsed_by_mapping"] = int(
        rx[rx.pt_code != ""].groupby("safetyreportid").reactionmeddrapt.apply(lambda s: s.str.lower().nunique()).sum()
        - len(rpt))
    n_pt = rpt.groupby("safetyreportid").size()
    facts["reports_without_mapped_pt"] = int(len(ids_all - set(n_pt.index)))
    rpt[["safetyreportid", "pt_code", "pt_name", "pt_primary_soc_code", "map_method"]].to_csv(
        FROZEN / "faers_report_pts.csv.gz", index=False)

    # ---------------- Step 7: population fields ----------------
    val = pd.to_numeric(rep.patientonsetage, errors="coerce")
    unit = rep.patientonsetageunit
    yrs = val * unit.map(AGE_UNIT_YEARS)
    groups = [age_group(y if pd.notna(y) else None, u, v if pd.notna(v) else None)
              for y, u, v in zip(yrs, unit, val)]
    rep["age_years"] = yrs.round(2)
    rep["age_group"] = [g for g, _ in groups]
    rep["age_group_status"] = [r for _, r in groups]
    rep["sex"] = rep.patientsex.map({"1": "male", "2": "female"}).fillna("")
    rep["hospitalization"] = np.where(rep.seriousnesshospitalization == "1", "yes", "no")
    crit = ["seriousnessdeath", "seriousnesslifethreatening", "seriousnesshospitalization", "seriousnessdisabling",
            "seriousnesscongenitalanomali", "seriousnessother"]
    rep["serious_no_criterion_recorded"] = (rep.serious == "1") & ~(rep[crit] == "1").any(axis=1)

    # ---------------- Step 4: possible duplicates (quantified, not removed) ----------------
    ptset = rpt.groupby("safetyreportid").pt_code.agg(lambda s: ",".join(sorted(s)))
    ingset = (te[te.identity_class.isin(MAIN_CLASSES)].groupby("safetyreportid").ingredient
              .agg(lambda s: ",".join(sorted(set(s)))))
    rep["pt_set"] = rep.safetyreportid.map(ptset).fillna("")
    rep["ingredient_set"] = rep.safetyreportid.map(ingset).fillna("")
    w = pd.to_numeric(rep.patientweight, errors="coerce").round(1)
    rep["_w"] = w.astype(str).where(w.notna(), "")
    rep["_a"] = rep.age_years.astype(str).where(rep.age_years.notna(), "")
    base = (rep.sex != "") & (rep._a != "") & (rep.occurcountry != "") & (rep.pt_set != "") & (rep.ingredient_set != "")
    loose = ["sex", "_a", "occurcountry", "pt_set", "ingredient_set"]
    keydefs = {"strict": (loose + ["_w"], base & (rep._w != "")),
               "loose": (loose, base),
               "loose_3pt": (loose, base & (rep.pt_set.str.count(",") >= 2))}
    dup = {}
    for kname, (cols, elig) in keydefs.items():
        k = rep.loc[elig, cols].astype(str).agg("|".join, axis=1)
        size = k.map(k.value_counts())
        gid = k.where(size > 1)
        rep[f"possible_duplicate_group_{kname}"] = ""
        rep.loc[gid.dropna().index, f"possible_duplicate_group_{kname}"] = (
            pd.factorize(gid.dropna())[0] + 1).astype(str)
        for scope, ids in {"all_reports": ids_all, "primary_analysis": s4}.items():
            m = rep.safetyreportid.isin(ids)
            kk = k[m.loc[k.index]]
            vc = kk.value_counts()
            in_groups = int(vc[vc > 1].sum())
            dup[f"{kname}_{scope}"] = {
                "eligible_reports": int(len(kk)), "reports_in_groups": in_groups,
                "groups": int((vc > 1).sum()), "excess_reports": int(in_groups - (vc > 1).sum()),
                "excess_share_of_scope": round((in_groups - (vc > 1).sum()) / len(ids), 4),
                "excess_share_of_eligible": round((in_groups - (vc > 1).sum()) / max(len(kk), 1), 4)}
    comp = rep.loc[rep.companynumb != "", "companynumb"]
    dup["companynumb_shared_by_multiple_reports"] = int((comp.value_counts() > 1).sum())
    owner = rep.loc[rep.companynumb != ""].groupby("companynumb").safetyreportid.agg(set).to_dict()
    linked = [sid for sid, nums in zip(rep.safetyreportid, rep.reportduplicate_numb.str.split("|"))
              if any(x and (owner.get(x, set()) - {sid}) for x in nums)]
    dup["reports_listing_a_duplicate_number_equal_to_another_reports_companynumb"] = len(linked)
    dup["of_which_in_primary_analysis"] = len(set(linked) & s4)
    dup["duplicate_field_eq_1"] = int((rep.duplicate == "1").sum())
    facts["duplicates"] = dup

    # ---------------- Frozen report-level tables ----------------
    rep_cols = ["safetyreportid", "safetyreportversion", "receivedate", "receiptdate", "reporttype", "serious",
                "hospitalization", "serious_no_criterion_recorded", "sex", "patientsex", "age_years", "age_group",
                "age_group_status", "patientonsetage", "patientonsetageunit", "occurcountry",
                "primarysource_qualification", "sender_type", "duplicate", "possible_duplicate_group_strict",
                "possible_duplicate_group_loose_3pt"]
    rep["n_mapped_pts"] = rep.safetyreportid.map(n_pt).fillna(0).astype(int)
    rep["in_primary_analysis"] = rep.safetyreportid.isin(s4)
    rep[rep_cols + ["n_mapped_pts", "in_primary_analysis"]].to_csv(FROZEN / "faers_reports.csv.gz", index=False)
    agg = agg.rename(columns={"stratum_brand": "brand"})
    agg.drop(columns=["in_window"]).assign(window_start=lambda d: [
        window_start(l, i, b) for l, i, b in zip(d.level, d.ingredient, d.brand)]).to_csv(
        FROZEN / "faers_report_strata.csv.gz", index=False)

    # ---------------- Aggregates ----------------
    has_pt = set(n_pt.index)
    all_rows = (agg.groupby("safetyreportid")[[f"in_{n}" for n in CONFIGS]].any().reset_index()
                .assign(level="all", ingredient="all", brand="all"))
    strata = pd.concat([agg[["safetyreportid", "level", "ingredient", "brand"] + [f"in_{n}" for n in CONFIGS]],
                        all_rows], ignore_index=True)
    strata = strata.merge(rep[["safetyreportid", "sex", "age_group", "hospitalization"]], on="safetyreportid")
    skey = ["config", "level", "ingredient", "brand"]
    long = pd.concat([strata[strata[f"in_{n}"]].assign(config=n) for n in CONFIGS], ignore_index=True)
    long = long.drop(columns=[f"in_{n}" for n in CONFIGS])
    long["has_mapped_pt"] = long.safetyreportid.isin(has_pt)
    den = long.groupby(skey).agg(n_reports=("safetyreportid", "size"),
                                 n_reports_with_mapped_pt=("has_mapped_pt", "sum")).reset_index()
    den.to_csv(FROZEN / "faers_denominators.csv", index=False)
    pc = (long.merge(rpt[["safetyreportid", "pt_code"]], on="safetyreportid")
          .groupby(skey + ["pt_code"]).size().rename("n_reports").reset_index()
          .merge(pt, on="pt_code").merge(den.rename(columns={"n_reports": "stratum_reports"}), on=skey))
    pc["share"] = (pc.n_reports / pc.n_reports_with_mapped_pt).round(6)
    pc = pc.sort_values(skey + ["n_reports"], ascending=[True] * 4 + [False])
    pc.to_csv(FROZEN / "faers_pt_counts.csv.gz", index=False)

    # RQ3: primary configuration by age group and sex.
    prim = long[long.config == "primary"].copy()
    prim["age_group"] = prim.age_group.replace("", "missing")
    prim["sex"] = prim.sex.replace("", "missing")
    k3 = ["level", "ingredient", "brand", "age_group", "sex"]
    den3 = prim.groupby(k3).agg(n_reports=("safetyreportid", "size"),
                                n_reports_with_mapped_pt=("has_mapped_pt", "sum")).reset_index()
    den3.to_csv(FROZEN / "faers_rq3_denominators.csv", index=False)
    pc3 = (prim.merge(rpt[["safetyreportid", "pt_code"]], on="safetyreportid").groupby(k3 + ["pt_code"]).size()
           .rename("n_reports").reset_index().merge(pt[["pt_code", "pt_name"]], on="pt_code"))
    pc3.to_csv(FROZEN / "faers_rq3_pt_by_age_sex.csv.gz", index=False)
    miss = {}
    for n in CONFIGS:
        ids = long.loc[long.config == n, "safetyreportid"].unique()
        r = rep[rep.safetyreportid.isin(ids)]
        miss[n] = {"reports": len(r), "sex_known": int((r.sex != "").sum()),
                   "age_known": int(r.age_years.notna().sum()), "age_group_known": int((r.age_group != "").sum()),
                   "age_group_under_13": int((r.age_group == "under 13").sum()),
                   "age_group_status": r.age_group_status.value_counts().to_dict(),
                   "both_known": int(((r.sex != "") & (r.age_group != "") & (r.age_group != "under 13")).sum())}
    facts["population_fields"] = miss
    facts["age_unit_counts_all_reports"] = rep.patientonsetageunit.replace("", "missing").value_counts().to_dict()

    # Step 8: hospitalization and action taken with the drug, per configuration and stratum.
    act_cols = {"primary": "actiondrug_primary", "sensitivity_any_role": "actiondrug_any_role",
                "sensitivity_broad_identity": "actiondrug_primary"}
    rows = []
    for n in CONFIGS:
        # The all-drug level takes the union of action codes over the report's ingredient rows in this config.
        ing_ok = agg[(agg.level == "ingredient") & agg[f"in_{n}"]]
        all_act = (ing_ok.groupby("safetyreportid")[act_cols[n]]
                   .agg(lambda s: ";".join(sorted(set(";".join(s).split(";"))))).reset_index()
                   .assign(level="all", ingredient="all", brand="all"))
        act_rows = pd.concat([agg[["safetyreportid", "level", "ingredient", "brand", act_cols[n]]], all_act])
        sub = long[long.config == n].merge(act_rows, how="left", on=["safetyreportid", "level", "ingredient", "brand"])
        # The broad-identity configuration reuses the primary action codes (blinded entries are not added).
        sub = sub.assign(acts=sub[act_cols[n]].fillna("").str.split(";"))
        for k, g in sub.groupby(["level", "ingredient", "brand"]):
            acts = g.acts.map(set)
            known = acts.map(lambda s: bool(s & {"1", "2", "3", "4"}))
            row = dict(zip(["level", "ingredient", "brand"], k), config=n, n_reports=len(g),
                       hospitalization_yes=int((g.hospitalization == "yes").sum()),
                       action_known_reports=int(known.sum()))
            for code, label in ACTION_LABEL.items():
                row[f"action_any_{label}"] = int(acts.map(lambda s: code in s).sum())
            rows.append(row)
    pd.DataFrame(rows).sort_values(skey).to_csv(FROZEN / "faers_rq2_hospitalization_action.csv", index=False)
    entry_actions = te[te.identity_class.isin(MAIN_CLASSES)].actiondrug.map(ACTION_LABEL).value_counts().to_dict()
    facts["actiondrug_main_entries"] = entry_actions
    r4 = rep[rep.safetyreportid.isin(s4)]
    facts["hospitalization_primary"] = {
        "yes": int((r4.hospitalization == "yes").sum()), "reports": len(r4),
        "seriousnesshospitalization_values": r4.seriousnesshospitalization.replace("", "absent").value_counts().to_dict(),
        "serious_no_criterion_recorded": int(r4.serious_no_criterion_recorded.sum()),
        "serious_values": r4.serious.replace("", "absent").value_counts().to_dict()}
    facts["reporttype_primary"] = r4.reporttype.replace("", "absent").value_counts().to_dict()

    # Brand-level flow.
    bflow = []
    for b, i in BRAND_TO_ING.items():
        e = te[(te.brand == b) & (te.identity_class == "brand")]
        ids = set(e.safetyreportid)
        a = agg[(agg.level == "brand") & (agg.brand == b)]
        inwin = set(a.loc[a.in_window, "safetyreportid"])
        bflow.append({"brand": b, "ingredient": i, "approval": APPROVAL[b], "reports_naming_brand": len(ids),
                      "before_approval_removed": len(ids - inwin), "in_window_any_role": len(ids & inwin),
                      "primary": int(a.in_primary.sum())})
    for i in INGREDIENTS:
        for lvl in ["ingredient", "unspecified"]:
            a = agg[(agg.level == lvl) & (agg.ingredient == i)]
            bflow.append({"brand": "(all products)" if lvl == "ingredient" else "unspecified", "ingredient": i,
                          "approval": ING_START[i], "reports_naming_brand": int(len(a)),
                          "before_approval_removed": int((~a.in_window).sum()),
                          "in_window_any_role": int(a.in_sensitivity_any_role.sum()),
                          "primary": int(a.in_primary.sum())})
    pd.DataFrame(bflow).to_csv(OUT / "flow_by_stratum.csv", index=False)
    pd.DataFrame(flow).to_csv(OUT / "flow_reports.csv", index=False)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1, default=int))
    print(pd.DataFrame(flow).to_string())
    print(json.dumps(facts, indent=1, default=int)[:6000])


if __name__ == "__main__":
    main()
