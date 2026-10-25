"""Read-only access to the formal KG packages, the frozen source snapshot and the study drug catalog.

Nothing here writes. Each mapped node is bound to its package by record id, node id, item path and the sha256 of its
carrier edge (canonical JSON), so a later change of the KG is detectable. The full source record is only read when a
caller asks for it; its fields must hash to the package's text_version.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .textnorm import fold_key

RECORD_ID_RE = re.compile(rb'"record_id":\s*"([^"]+)"')
ROLE_OF_NODE = {'SideEffect': 'side_effect', 'Indication': 'indication'}
EDGE_OF_ROLE = {'side_effect': 'REPORTED_SIDE_EFFECT', 'indication': 'USED_FOR'}


class BindingError(ValueError):
    pass


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def iter_packages(path, record_ids=None):
    """Yield (1-based line, package) for every package, or only for `record_ids`, and finally verify nothing:
    callers compare `file_sha256(path)` before and after a run (the stream itself never writes)."""
    wanted = None if record_ids is None else set(record_ids)
    with open(path, 'rb') as stream:
        for lineno, line in enumerate(stream, 1):
            if wanted is not None:
                # every record_id value in one package line is the package's own id; the first one decides the skip
                m = RECORD_ID_RE.search(line)
                if m is None or m.group(1).decode() not in wanted:
                    continue
            package = json.loads(line)
            if wanted is None or package['record_id'] in wanted:
                yield lineno, package


@dataclass(frozen=True)
class NodeBinding:
    record_id: str
    kg_line: int
    role: str
    node: dict
    edge: dict
    drug: dict
    carrier_sha256: str
    source_line: int
    source_sha256: str
    text_version: str
    record_drugs: tuple = ()  # every Drug node of the same package (read-only), for purpose-relation scope

    @property
    def quotes(self):
        return [ev['quote'] for ev in self.edge['evidence']]

    @property
    def review_flags(self):
        return list(self.edge['review']['review_flags'])


def bind(package, node_id, kg_line):
    """Bind one SideEffect / Indication node to its package and its single carrier edge."""
    nodes = {n['id']: n for n in package['nodes']}
    node = nodes.get(node_id)
    if node is None or node['type'] not in ROLE_OF_NODE:
        raise BindingError(f'{node_id}: no SideEffect or Indication node in {package["record_id"]}')
    role = ROLE_OF_NODE[node['type']]
    edges = [e for e in package['edges'] if e.get('target') == node_id]
    if len(edges) != 1 or edges[0]['type'] != EDGE_OF_ROLE[role] or edges[0]['item_path'] != node['item_path']:
        raise BindingError(f'{node_id}: expected one {EDGE_OF_ROLE[role]} carrier at {node["item_path"]}')
    edge = edges[0]
    drug = nodes.get(edge['source'])
    if drug is None or drug['type'] != 'Drug':
        raise BindingError(f'{node_id}: carrier source {edge["source"]} is not a Drug node')
    snap = package['provenance']['frozen_inputs']['source_snapshot']
    record_drugs = tuple(n for n in package['nodes'] if n['type'] == 'Drug')
    return NodeBinding(record_id=package['record_id'], kg_line=kg_line, role=role, node=node, edge=edge, drug=drug,
                       carrier_sha256=canonical_sha256(edge), source_line=snap['line'], source_sha256=snap['sha256'],
                       text_version=package['source_metadata']['text_version'], record_drugs=record_drugs)


def load_bindings(kg_path, node_ids):
    """Bind the requested node ids (one streaming pass). Every id must be found exactly once."""
    by_record = {}
    for node_id in node_ids:
        record_id = node_id.split(':', 1)[1].rsplit(':', 1)[0]
        by_record.setdefault(record_id, set()).add(node_id)
    found = {}
    for lineno, package in iter_packages(kg_path, set(by_record)):
        for node_id in sorted(by_record.get(package['record_id'], ())):
            if node_id in found:
                raise BindingError(f'{node_id}: record occurs twice in the KG')
            found[node_id] = bind(package, node_id, lineno)
    missing = sorted(set(node_ids) - set(found))
    if missing:
        raise BindingError(f'{len(missing)} requested nodes not in the KG, e.g. {missing[:3]}')
    return [found[n] for n in node_ids]


def load_source_records(source_path, bindings):
    """Full source record text per record id, read from the frozen snapshot at the bound lines and checked against
    each package's text_version (sha256 of the canonical fields JSON)."""
    wanted = {b.source_line: b for b in bindings}
    out = {}
    with open(source_path, 'rb') as stream:
        for lineno, line in enumerate(stream, 1):
            b = wanted.get(lineno)
            if b is None:
                continue
            rec = json.loads(line)
            if rec['record_id'] != b.record_id:
                raise BindingError(f'source line {lineno} holds {rec["record_id"]}, expected {b.record_id}')
            digest = 'sha256:' + hashlib.sha256(json.dumps(rec['fields'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest != b.text_version:
                raise BindingError(f'{b.record_id}: source fields do not match text_version')
            out[b.record_id] = rec['fields']
    missing = sorted({b.record_id for b in bindings} - set(out))
    if missing:
        raise BindingError(f'{len(missing)} records not found in the source snapshot, e.g. {missing[:3]}')
    return out


def check_evidence(binding, fields):
    """Every carrier evidence span must be the exact slice of its source field (no re-search, no tolerance)."""
    for i, ev in enumerate(binding.edge['evidence']):
        text = fields.get(ev['field'])
        if not isinstance(text, str) or not (0 <= ev['start'] < ev['end'] <= len(text)) or text[ev['start']:ev['end']] != ev['quote']:
            raise BindingError(f'{binding.node["id"]}: evidence[{i}] is not the source slice at its offsets')


def compact_key(text):
    """Spacing/hyphen-insensitive key for catalog names only ("Victoza 2-Pak" = "victoza2pak")."""
    return re.sub(r'[\s\-_]+', '', fold_key(text))


@dataclass(frozen=True)
class Catalog:
    sha256: str
    brands: frozenset
    generics: frozenset
    aliases: frozenset
    compact: dict  # compact key -> 'catalog_brand' | 'catalog_generic' (keys claimed by both are dropped)
    names: tuple = ()      # every catalog name as written (brand, generic, alias), for relation scope only
    identity: dict = None  # compact key -> ('brand', brand key) | ('generic', generic key)
    generic_of: dict = None  # brand key -> generic key


def load_catalog(path):
    drugs = json.loads(Path(path).read_text())
    brands = frozenset(fold_key(d['brand']) for d in drugs)
    generics = frozenset(fold_key(d['generic']) for d in drugs)
    aliases = frozenset(fold_key(a) for d in drugs for a in d['aliases'])
    claims = {}
    for names, category in ((brands | aliases, 'catalog_brand'), (generics, 'catalog_generic')):
        for name in names:
            claims.setdefault(compact_key(name), set()).add(category)
    compact = {key: next(iter(cats)) for key, cats in claims.items() if len(cats) == 1}
    identity, generic_of, names = {}, {}, []
    for d in drugs:
        brand, generic = fold_key(d['brand']), fold_key(d['generic'])
        generic_of[brand] = generic
        for name in [d['brand'], *d['aliases']]:
            identity[compact_key(name)] = ('brand', brand)
            names.append(name)
        identity[compact_key(d['generic'])] = ('generic', generic)
        names.append(d['generic'])
    return Catalog(sha256=file_sha256(path), brands=brands, generics=generics, aliases=aliases, compact=compact,
                   names=tuple(dict.fromkeys(names)), identity=identity, generic_of=generic_of)


def _drug_identity(name, catalog):
    return catalog.identity.get(compact_key(name), ('name', compact_key(name)))


def scope_class(name, carrier, catalog):
    """How a drug name written in the source relates to the carrier's drug, for purpose-relation scope only (never an
    identity decision): carrier | related (same ingredient at another brand/generic level, or one name contained in the
    other across the catalog boundary: not uniquely the carrier) | other."""
    a, b = compact_key(name), compact_key(carrier)
    if not a or not b:
        return 'other'
    if a == b:
        return 'carrier'
    ia, ib = _drug_identity(name, catalog), _drug_identity(carrier, catalog)
    if ia[0] != 'name' and ia == ib:
        return 'carrier'
    if {ia[0], ib[0]} == {'brand', 'generic'}:
        brand, generic = (ia[1], ib[1]) if ia[0] == 'brand' else (ib[1], ia[1])
        return 'related' if catalog.generic_of.get(brand) == generic else 'other'
    fa, fb = f' {fold_key(name)} ', f' {fold_key(carrier)} '
    if fa in fb or fb in fa:  # "metformin" / "Metformin ER", "tirzepatide" / "compound tirzepatide"
        return 'carrier' if ia[0] == ib[0] == 'name' else 'related'
    return 'other'


def scope_names(binding, catalog):
    """(name, class) pairs a purpose relation can be governed by: the carrier's own surface, every other named or
    class Drug node of the same package and every catalog name. Pronoun / dose / unspecified references are never
    names. When one written name gets several classes the most carrier-like one wins."""
    carrier = binding.drug['surface']
    rank = {'carrier': 0, 'related': 1, 'other': 2}
    out = {fold_key(carrier): (carrier, 'carrier')}
    candidates = [n['surface'] for n in binding.record_drugs
                  if n['id'] != binding.drug['id'] and n.get('reference_type') in ('name', 'class')]
    for name in candidates + list(catalog.names if catalog is not None and catalog.names else ()):
        key = fold_key(name)
        if not key:
            continue
        cls = 'carrier' if key == fold_key(carrier) else (scope_class(name, carrier, catalog) if catalog is not None else 'other')
        if key not in out or rank[cls] < rank[out[key][1]]:
            out[key] = (name, cls)
    return tuple(sorted(out.values(), key=lambda v: (-len(v[0]), fold_key(v[0]))))


def drug_category(drug, catalog):
    """Study-catalog category of the carrier's drug, for stratification only (never an identity decision)."""
    if drug['identity']['status'] != 'as_extracted':
        return 'held'
    if drug['reference_type'] != 'name':
        return drug['reference_type']  # class | unspecified
    key = fold_key(drug['surface'])
    if key in catalog.brands or key in catalog.aliases:
        return 'catalog_brand'
    if key in catalog.generics:
        return 'catalog_generic'
    return catalog.compact.get(compact_key(drug['surface']), 'name_not_in_catalog')
