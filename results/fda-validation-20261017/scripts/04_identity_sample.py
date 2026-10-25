"""Step 2 (review sample): stratified sample of distinct raw product-name spellings for an agent review of the
identity rule. Strata: identity class x brand (brand class) or x ingredient (other classes). Each stratum gives
its 3 most frequent spellings plus up to 5 drawn at random (seed 20261017) from the remaining spellings."""
from __future__ import annotations

import pandas as pd

from common import OUT, REVIEW, read_csv

SEED = 20261017


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    e = read_csv(OUT / "entries_classified.csv.gz")
    e["name"] = e.medicinalproduct.str.upper().str.strip()
    e["substance"] = e.activesubstancename.str.upper().str.strip()
    e["stratum"] = e.identity_class + ":" + e.brand.where(e.identity_class == "brand", e.ingredient)
    g = (e.groupby(["stratum", "name", "substance", "ingredient", "identity_class", "brand"])
         .agg(n_entries=("safetyreportid", "size"), n_reports=("safetyreportid", "nunique")).reset_index())
    rows = []
    for st, d in g.groupby("stratum"):
        d = d.sort_values(["n_entries", "name"], ascending=[False, True])
        top = d.head(3).assign(pick="top3")
        rest = d.iloc[3:]
        rnd = rest.sample(min(5, len(rest)), random_state=SEED).assign(pick="random") if len(rest) else rest
        rows.append(pd.concat([top, rnd]))
        print(st, "distinct spellings", len(d))
    s = pd.concat(rows, ignore_index=True)
    s.insert(0, "sample_id", [f"S{i + 1:03d}" for i in range(len(s))])
    s.to_csv(REVIEW / "identity_sample.csv", index=False)
    g.groupby("stratum").agg(distinct_spellings=("name", "size"), entries=("n_entries", "sum")).to_csv(
        REVIEW / "identity_spelling_strata.csv")
    print(len(s), "sampled spellings")


if __name__ == "__main__":
    main()
