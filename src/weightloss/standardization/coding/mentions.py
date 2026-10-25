"""One row per KG side-effect or purpose edge: the verbatim term, its evidence quotes and role."""
from __future__ import annotations

import json
from pathlib import Path

ROLES = {'REPORTED_SIDE_EFFECT': ('side_effect', 'effect'), 'USED_FOR': ('purpose', 'indication')}


def iter_mentions(kg_packages):
    with open(kg_packages) as f:
        for line in f:
            package = json.loads(line)
            for edge in package.get('edges', []):
                role = ROLES.get(edge.get('type'))
                if role is None:
                    continue
                verbatim = (edge.get('original_item') or {}).get(role[1])
                if not verbatim:
                    continue
                yield {'mention_id': edge['id'], 'record_id': edge['record_id'], 'platform': edge.get('platform'),
                       'role': role[0], 'verbatim': verbatim, 'node_id': edge['target'],
                       'quotes': [q['quote'] for q in edge.get('evidence', [])]}


def build(kg_packages, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.tmp')
    n = 0
    with open(tmp, 'w') as f:
        for row in iter_mentions(kg_packages):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
    tmp.rename(out_path)
    return n


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
