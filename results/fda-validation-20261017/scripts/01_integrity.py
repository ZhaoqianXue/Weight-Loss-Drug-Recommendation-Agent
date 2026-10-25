"""Step 1: integrity. Re-verify every hash recorded in data/fda/manifest.json, re-decode every raw page, and
rebuild reports, drug entries and reactions from the raw JSON independently of the acquisition code, so the
processed tables used downstream can be compared row by row with the raw pages.

Reads only; writes to results/fda-validation-20261017/outputs/integrity/.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import zstandard

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/fda-validation-20261017/outputs/integrity"
MANIFEST = ROOT / "data/fda/manifest.json"
RAW = ROOT / "data/fda/raw/openfda-drug-event/release-2026-07-30"
INVENTORY = RAW / "page_inventory.csv"
INGREDIENTS = {
    "semaglutide": ["wegovy", "ozempic", "rybelsus"],
    "tirzepatide": ["mounjaro", "zepbound"],
    "liraglutide": ["victoza", "saxenda"],
    "dulaglutide": ["trulicity"],
}
TERM_TO_ING = {t: i for i, bs in INGREDIENTS.items() for t in [i, *bs]}
TOKEN = re.compile(r"[a-z0-9]+")
ZSTD = zstandard.ZstdDecompressor(max_window_size=2 ** 27)


def sha256_file(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def manifest_entries(node, trail=""):
    """Yield every {path, sha256[, bytes, rows]} dict anywhere in the manifest."""
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield trail, node
        for k, v in node.items():
            yield from manifest_entries(v, f"{trail}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from manifest_entries(v, f"{trail}[{i}]")


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def s(v) -> str:
    return "" if v is None else str(v)


def joined(values) -> str:
    return "|".join(str(v) for v in as_list(values) if v not in (None, ""))


def tokens(values) -> set[str]:
    out = set()
    for v in as_list(values):
        if isinstance(v, str):
            out.update(TOKEN.findall(v.lower()))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    # 1a. Every file hash listed in the manifest.
    manifest = json.loads(MANIFEST.read_text())
    for trail, e in manifest_entries(manifest):
        p = ROOT / e["path"]
        if not p.exists():
            checks.append({"check": "manifest_file", "path": e["path"], "ok": False, "detail": "missing"})
            continue
        n, h = sha256_file(p)
        ok = h == e["sha256"] and ("bytes" not in e or n == e["bytes"])
        checks.append({"check": "manifest_file", "path": e["path"], "ok": ok,
                       "detail": f"bytes={n} sha256={h[:16]}"})

    # 1b. Every raw page: stored hash, decoded hash, result count; independent rebuild of reports.
    inv = list(csv.DictReader(open(INVENTORY)))
    per_query = Counter()
    docs_seen = {}           # sid -> (content hash, transmissiondate, location)
    report_rows = {}         # sid -> report dict (retained version)
    entry_rows = {}          # sid -> list of entry dicts
    reaction_rows = {}       # sid -> list of reaction tuples
    membership = defaultdict(set)
    page_fail = 0
    n_docs = 0
    distinct_doc_ids = Counter()
    for k, row in enumerate(inv):
        p = ROOT / row["stored_path"]
        data = p.read_bytes()
        ok_stored = len(data) == int(row["bytes_stored"]) and hashlib.sha256(data).hexdigest() == row["sha256_stored"]
        body = ZSTD.decompress(data)
        ok_dec = len(body) == int(row["bytes_decoded"]) and hashlib.sha256(body).hexdigest() == row["sha256_decoded"]
        side = json.loads(p.with_name(p.name.replace(".json.zst", ".meta.json")).read_text())
        ok_side = side["sha256_stored"] == row["sha256_stored"] and side["sha256_decoded"] == row["sha256_decoded"]
        results = orjson.loads(body).get("results", [])
        ok_n = len(results) == int(row["n_results"])
        if not (ok_stored and ok_dec and ok_side and ok_n):
            page_fail += 1
            checks.append({"check": "raw_page", "path": row["stored_path"], "ok": False,
                           "detail": f"stored={ok_stored} decoded={ok_dec} sidecar={ok_side} n={ok_n}"})
        q = row["ingredient_query"]
        per_query[q] += len(results)
        for idx, r in enumerate(results):
            n_docs += 1
            sid = r["safetyreportid"]
            membership[sid].add(q)
            ch = hashlib.sha256(orjson.dumps(r, option=orjson.OPT_SORT_KEYS)).hexdigest()[:32]
            td = s(r.get("transmissiondate"))
            prev = docs_seen.get(sid)
            if prev is not None:
                if prev[0] == ch:
                    continue
                distinct_doc_ids[sid] += 1
                if td <= prev[1]:  # keep the later transmissiondate (acquisition rule), first seen on ties
                    continue
            docs_seen[sid] = (ch, td, (q, row["stored_path"], idx))
            patient = r.get("patient") or {}
            ps = r.get("primarysource") or {}
            drugs = as_list(patient.get("drug"))
            reactions = as_list(patient.get("reaction"))
            report_rows[sid] = {
                "safetyreportid": sid, "safetyreportversion": s(r.get("safetyreportversion")),
                "receivedate": s(r.get("receivedate")), "receiptdate": s(r.get("receiptdate")),
                "serious": s(r.get("serious")), "seriousnesshospitalization": s(r.get("seriousnesshospitalization")),
                "patientsex": s(patient.get("patientsex")), "patientonsetage": s(patient.get("patientonsetage")),
                "patientonsetageunit": s(patient.get("patientonsetageunit")),
                "occurcountry": s(r.get("occurcountry")), "companynumb": s(r.get("companynumb")),
                "primarysource_qualification": s(ps.get("qualification")),
                "n_drug_entries": str(len(drugs)), "n_reaction_entries": str(len(reactions)),
            }
            reaction_rows[sid] = [(i, s(x.get("reactionmeddrapt")), s(x.get("reactionmeddraversionpt")),
                                   s(x.get("reactionoutcome"))) for i, x in enumerate(reactions)]
            ents = []
            for di, d in enumerate(drugs):
                ofd = d.get("openfda") or {}
                subs = [a.get("activesubstancename") for a in as_list(d.get("activesubstance")) if isinstance(a, dict)]
                raw_text = (s(d.get("medicinalproduct")) + " " + " ".join(s(x) for x in subs)).lower()
                tok_any = (tokens(d.get("medicinalproduct")) | tokens(subs) | tokens(ofd.get("generic_name"))
                           | tokens(ofd.get("brand_name")) | tokens(ofd.get("substance_name")))
                tok_ings = sorted({TERM_TO_ING[t] for t in tok_any if t in TERM_TO_ING})
                sub_ings = sorted({ing for t, ing in TERM_TO_ING.items() if t in raw_text})
                if not tok_ings and not sub_ings:
                    continue
                ents.append({
                    "safetyreportid": sid, "drug_index": di, "token_ingredients": ";".join(tok_ings),
                    "substring_raw_ingredients": ";".join(sub_ings),
                    "medicinalproduct": s(d.get("medicinalproduct")), "activesubstancename": joined(subs),
                    "drugcharacterization": s(d.get("drugcharacterization")), "actiondrug": s(d.get("actiondrug")),
                    "drugindication": s(d.get("drugindication")),
                    "drugadministrationroute": s(d.get("drugadministrationroute")),
                })
            entry_rows[sid] = ents
        if k % 50 == 0:
            print(f"page {k + 1}/{len(inv)} docs={n_docs} reports={len(report_rows)}", flush=True)

    for q, n in per_query.items():
        done = json.loads((RAW / "events" / q / "complete.json").read_text())
        checks.append({"check": "query_total", "path": q, "ok": n == done["meta_total"] == done["records"],
                       "detail": f"retrieved={n} meta_total={done['meta_total']}"})
    checks.append({"check": "raw_pages_all_verified", "path": str(INVENTORY.relative_to(ROOT)),
                   "ok": page_fail == 0, "detail": f"pages={len(inv)} failed={page_fail}"})

    def write(name, cols, rows):
        with gzip.open(OUT / name, "wt", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write("raw_reports.csv.gz", list(next(iter(report_rows.values())).keys()), report_rows.values())
    ecols = ["safetyreportid", "drug_index", "token_ingredients", "substring_raw_ingredients", "medicinalproduct",
             "activesubstancename", "drugcharacterization", "actiondrug", "drugindication", "drugadministrationroute"]
    write("raw_candidate_entries.csv.gz", ecols, (e for es in entry_rows.values() for e in es))
    write("raw_reactions.csv.gz", ["safetyreportid", "reaction_index", "reactionmeddrapt", "reactionmeddraversionpt",
                                   "reactionoutcome"],
          ({"safetyreportid": sid, "reaction_index": i, "reactionmeddrapt": pt, "reactionmeddraversionpt": v,
            "reactionoutcome": o} for sid, rs in reaction_rows.items() for i, pt, v, o in rs))
    summary = {
        "documents": n_docs, "unique_reports": len(report_rows), "per_query_documents": dict(per_query),
        "reports_returned_by_multiple_queries": sum(len(v) > 1 for v in membership.values()),
        "ids_with_multiple_distinct_documents": dict(distinct_doc_ids),
        "checks_total": len(checks), "checks_failed": sum(not c["ok"] for c in checks),
    }
    (OUT / "raw_rebuild_summary.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / "hash_checks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "path", "ok", "detail"])
        w.writeheader()
        w.writerows(checks)
    print(json.dumps(summary, indent=1))
    if summary["checks_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
