"""Deterministic summary of annotation coverage and standardized mentions."""
from collections import Counter
import json
from weightloss.settings import get_settings
from weightloss.pipeline.refresh_data import read
from weightloss.runs import atomic_json

def summarize(rows):
    effects = Counter()
    for row in rows:
        if row.get('Extraction Status') in ('pending','failed') or row.get('Standardization Status') == 'pending':
            continue
        names = set()
        for effect in json.loads(row['standardized_info'])['side_effects']:
            value = effect['name']
            names.update(n.strip().lower() for n in (value if isinstance(value,list) else [value]) if n.strip())
        effects.update(names)
    return {'rows':len(rows),'extraction_status':dict(Counter(r['Extraction Status'] for r in rows)),'mentions':dict(sorted(effects.items())),'unit':'Unique reviews mentioning each term; pending annotations are unknown.'}

def main():
    paths = get_settings()
    report = summarize(read(paths.path('standardized')))
    atomic_json(paths.path('results')/'metrics.json',report)
    print(json.dumps(report,indent=2))
