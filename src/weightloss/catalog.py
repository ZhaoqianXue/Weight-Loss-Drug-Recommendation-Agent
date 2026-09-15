"""Canonical drug catalog. Importing this module does not load project data."""
import json
from .settings import get_settings

def load_catalog():
    drugs = json.loads(get_settings().path('catalog').read_text())
    for drug in drugs:
        drug['url'] = 'https://reviews.webmd.com/drugs/drugreview-' + drug['slug']
    return drugs

def __getattr__(name):
    # Compatibility with research consumers; loading is deferred to explicit access.
    if name == 'ROOT':
        return get_settings().root
    if name not in ('DRUGS', 'BRANDS', 'GENERIC_TO_BRANDS', 'ALIASES'):
        raise AttributeError(name)
    drugs = load_catalog()
    return {
        'DRUGS': drugs,
        'BRANDS': [d['brand'] for d in drugs],
        'GENERIC_TO_BRANDS': {g: [d['brand'] for d in drugs if d['generic'] == g] for g in dict.fromkeys(d['generic'] for d in drugs)},
        'ALIASES': {n.lower():d['brand'] for d in drugs for n in [d['brand']] + d['aliases']},
    }[name]
