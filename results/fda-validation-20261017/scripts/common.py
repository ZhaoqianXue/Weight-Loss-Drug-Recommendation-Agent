"""Shared constants and loaders for the FAERS validation (release 2026-07-30)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "results/fda-validation-20261017"
PROC = ROOT / "data/fda/processed/openfda-drug-event-release-2026-07-30"
MEDDRA = ROOT / ".cache/terminology/meddra/29.1/English/api-20261004"
KG = ROOT / ".cache/reextraction-20261016/kg/full-v7/kg.sqlite"
OUT = STAGE / "outputs"
FROZEN = STAGE / "frozen"
REVIEW = STAGE / "review"

INGREDIENTS = {
    "semaglutide": ["wegovy", "ozempic", "rybelsus"],
    "tirzepatide": ["mounjaro", "zepbound"],
    "liraglutide": ["victoza", "saxenda"],
    "dulaglutide": ["trulicity"],
}
BRAND_TO_ING = {b: i for i, bs in INGREDIENTS.items() for b in bs}
TERMS = {t for i, bs in INGREDIENTS.items() for t in [i, *bs]}
TOKEN = re.compile(r"[a-z0-9]+")

# FDA approval dates, the same dates that set the Reddit collection start (configs/reddit.json).
APPROVAL = {b.lower(): d for b, d in json.loads((ROOT / "configs/reddit.json").read_text())["brands"].items()}
ING_START = {i: min(APPROVAL[b] for b in bs) for i, bs in INGREDIENTS.items()}
WINDOW_END = "2026-06-30"  # last receivedate in the openFDA release; patient data are cut here too

# Raw-field identity markers (reporter product name and active-substance fields only).
COMBINATION = re.compile(r"xultophy|degludec.*liraglutide|liraglutide.*degludec")
BLINDED = re.compile(r"blind|placebo")
COMPOUNDED = re.compile(
    r"compound|cobalamin|\bb-?12\b|pyridoxine|niacinamide|nicotinamide|glycine|carnitine|carnatine|nadide"
    r"|inositol|methionine|choline|chromium|citrulline"
)
MAIN_CLASSES = ["brand", "multiple_brands", "ingredient_name", "compounded"]
SENSITIVITY_CLASSES = ["blinded", "harmonized_only"]
PRIMARY_ROLES = {"suspect", "interacting"}

AGE_GROUPS = [(13, 18, "13-18"), (19, 24, "19-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"),
              (55, 64, "55-64"), (65, 74, "65-74"), (75, 10_000, "75 or over")]
AGE_UNIT_YEARS = {"800": 10.0, "801": 1.0, "802": 1 / 12, "803": 7 / 365.25, "804": 1 / 365.25, "805": 1 / 8766}
ACTION_LABEL = {"1": "drug_withdrawn", "2": "dose_reduced", "3": "dose_increased", "4": "dose_not_changed",
                "5": "unknown", "6": "not_applicable", "": "missing"}


def read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meddra() -> tuple[pd.DataFrame, pd.DataFrame]:
    pt = pd.read_csv(MEDDRA / "pt.asc", sep="$", header=None, dtype=str, keep_default_na=False, quoting=3)
    pt = pt.iloc[:, [0, 1, 3]].set_axis(["pt_code", "pt_name", "pt_primary_soc_code"], axis=1)
    llt = pd.read_csv(MEDDRA / "llt.asc", sep="$", header=None, dtype=str, keep_default_na=False, quoting=3)
    llt = llt.iloc[:, [0, 1, 2, 9]].set_axis(["llt_code", "llt_name", "pt_code", "llt_currency"], axis=1)
    return pt, llt


def map_terms(strings: pd.Series, pt: pd.DataFrame, llt: pd.DataFrame) -> pd.DataFrame:
    """Map distinct MedDRA term strings to a 29.1 PT: PT name (case-insensitive), else LLT name -> its PT.
    '^' stored for an apostrophe is restored first. Returns one row per distinct input string."""
    by_pt = {n.lower(): c for c, n in zip(pt.pt_code, pt.pt_name)}
    by_llt = {n.lower(): (c, cur) for n, c, cur in zip(llt.llt_name, llt.pt_code, llt.llt_currency)}
    rows = []
    for raw in pd.unique(strings):
        fixed = raw.replace("^", "'")
        key = fixed.strip().lower()
        caret = "caret_fixed+" if fixed != raw else ""
        if key in by_pt:
            code, how = by_pt[key], "pt_name"
        elif key in by_llt:
            code, cur = by_llt[key]
            how = "llt_current" if cur == "Y" else "llt_noncurrent"
        else:
            code, how = "", "unmatched"
        rows.append({"raw_term": raw, "pt_code": code, "map_method": caret + how})
    m = pd.DataFrame(rows).merge(pt, on="pt_code", how="left").fillna("")
    return m


def age_group(age_years: float | None, unit: str, value: float | None) -> tuple[str, str]:
    """WebMD age group for an onset age; decade-coded ages get a group only when the decade fits in one group."""
    if value is None:
        return "", "age_missing"
    if unit not in AGE_UNIT_YEARS:
        return "", "unit_missing_or_invalid"
    if unit == "800":
        lo, hi = value * 10, value * 10 + 9
    else:
        lo = hi = age_years
    if hi > 120:
        return "", "implausible_over_120"
    if hi < 13:
        return "under 13", "ok"
    for a, b, label in AGE_GROUPS:
        if a <= int(lo) and int(hi) <= b:
            return label, "ok"
    return "", "decade_spans_groups" if unit == "800" else "under13_boundary"
