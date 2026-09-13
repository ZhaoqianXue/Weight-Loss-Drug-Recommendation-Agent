"""Canonical review-page catalog. Aliases never create additional datasets."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DRUGS = json.loads((ROOT / 'config/drugs.json').read_text())
for drug in DRUGS:
    drug['url'] = 'https://reviews.webmd.com/drugs/drugreview-' + drug['slug']
BRANDS = [d['brand'] for d in DRUGS]
GENERIC_TO_BRANDS = {g: [d['brand'] for d in DRUGS if d['generic'] == g] for g in dict.fromkeys(d['generic'] for d in DRUGS)}
ALIASES = {name.lower(): d['brand'] for d in DRUGS for name in [d['brand']] + d['aliases']}
