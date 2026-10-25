"""Stage manifest: hashes of inputs, scripts and outputs, the decisions applied, and the freeze time."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from common import KG, MEDDRA, PROC, ROOT, STAGE, sha256

DECISIONS = {
    "combination_products": "excluded (user 2026-10-17): reports whose only study-drug entries are insulin "
                            "degludec/liraglutide (Xultophy) are dropped; in other reports only that entry is dropped",
    "cross_sender_duplicates": "not removed (user 2026-10-17); size quantified, reported as a limitation",
    "ingredient_name_only_reports": "kept (user 2026-10-17): they form the brand-unspecified level and count at the "
                                    "ingredient level, never at the brand level",
    "time_window": "aligned (user 2026-10-17): receivedate (FAERS) and record date (patient side) from the brand's "
                   "FDA approval date (ingredient level: the ingredient's first brand approval) through 2026-06-30",
    "drug_identity": "reporter product name and active-substance fields only; openFDA harmonized fields never used",
    "role": "primary analysis: suspect or interacting role on the matched study-drug entry; sensitivity: any role",
    "meddra": "MedDRA 29.1: PT name (case-insensitive, '^' restored to an apostrophe), else LLT name to its PT",
    "compounded_products": "kept (user 2026-10-17): entries listing additive substances or 'compound' in the raw "
                           "name count at the ingredient level with brand unspecified",
    "blinded_trial_entries": "excluded from the primary analysis (user 2026-10-17); kept in the broad-identity "
                             "sensitivity analysis",
    "action_taken_denominator": "reports with a known action (codes 1-4) in the primary result; the all-reports "
                                "denominator is reported alongside (user 2026-10-17)",
    "git": "frozen/ and outputs/ are git-ignored because they hold MedDRA content and large intermediates "
           "(user 2026-10-17)",
    "nonclinical_pts": "RQ2 compares clinical-event PTs only (user 2026-10-17). Excluded on each PT's primary path, "
                       "both sides: SOC Product issues; HLGT Medication errors and other product use errors and "
                       "issues; HLGT Off label uses and intentional product misuses/use issues; HLT Therapeutic and "
                       "nontherapeutic responses; HLT Interactions; HLGT Overdoses and underdoses NEC; HLT Exposures "
                       "associated with pregnancy, delivery and lactation. Denominators become reports/records with "
                       "at least one remaining side effect",
}


def main() -> None:
    files = sorted(p for p in STAGE.rglob("*") if p.is_file() and p.name != "manifest.json"
                   and "__pycache__" not in p.parts)
    out = {
        "title": "FAERS comparator validation (openFDA release 2026-07-30) for RQ2 and RQ3",
        "status": "validated_frozen",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "created_by": "Claude (agent), session started from the WeightLoss coordination session; no paid API calls",
        "decisions": DECISIONS,
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in [
            PROC / "reports.csv.gz", PROC / "target_drug_entries.csv.gz", PROC / "reactions.csv.gz",
            MEDDRA / "pt.asc", MEDDRA / "llt.asc", MEDDRA / "meddra_release.asc", MEDDRA / "mdhier.asc",
            MEDDRA / "hlgt.asc", MEDDRA / "soc.asc", ROOT / "configs/reddit.json", KG]],
        "files": [],
    }
    for p in files:
        rec = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        if p.suffix in (".csv", ".gz") and ".csv" in p.name:
            rec["rows"] = int(len(pd.read_csv(p, dtype=str, usecols=[0])))
        out["files"].append(rec)
    (STAGE / "manifest.json").write_text(json.dumps(out, indent=1))
    print(len(out["files"]), "files recorded")


if __name__ == "__main__":
    main()
