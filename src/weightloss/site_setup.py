"""Set up the AskGLP1 website on a fresh checkout.

The code is in git; the data are not, because they hold MedDRA terms and Reddit-derived statements (the repository is
public). They travel as one zip, the site data bundle, shared in the project's Google Drive folder:

- kg.sqlite and its manifest.json (the knowledge graph), unpacked to .cache/reextraction-20261016/kg/full-v7/;
- frozen/, the validated FAERS tables with the MedDRA PT hierarchy, unpacked to results/fda-validation-20261017/frozen/;
- bundle.json, the SHA-256 of every file.

`python -m weightloss setup-site --bundle <zip>` checks the bundle, unpacks it, imports the graph into the project-local
Neo4j (installed separately: brew install neo4j), builds the analysis database the agent's code tool reads, and checks
the OpenAI key and the frontend build. Each step is skipped when its result is already in place, so the command can be
run again after an interruption. `--make-bundle <zip>` writes the bundle from this machine's data.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _targets():
    from weightloss.graph import neo4j_db
    kg_dir = neo4j_db.KG.parent
    return {'kg.sqlite': neo4j_db.KG, 'manifest.json': kg_dir / 'manifest.json'}, neo4j_db.FROZEN


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_bundle(out):
    """Zip this machine's knowledge graph and frozen FAERS tables into the site data bundle."""
    files, frozen = _targets()
    members = {name: path for name, path in files.items()}
    members.update({f'frozen/{p.name}': p for p in sorted(frozen.iterdir()) if p.is_file()})
    missing = [str(p) for p in members.values() if not p.is_file()]
    if missing:
        raise SystemExit(f'cannot make the bundle; missing: {missing}')
    meta = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'files': {name: {'sha256': _sha256(path), 'bytes': path.stat().st_size} for name, path in members.items()}}
    out = Path(out)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in members.items():
            z.write(path, name)
        z.writestr('bundle.json', json.dumps(meta, indent=1))
    print(f'wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {len(members)} files)')
    return out


def _inputs_unchanged(recorded):
    """Every input a build recorded (repository-relative path -> SHA-256) still has that hash: the graph, each frozen
    FAERS table, the MedDRA hierarchy and the approval dates. A new bundle that changes any of them forces a rebuild."""
    if not recorded:
        return False
    return all((ROOT / rel).is_file() and _sha256(ROOT / rel) == sha for rel, sha in recorded.items())


def _step(text):
    print(f'\n== {text}', flush=True)


def setup(bundle=None):
    from weightloss.graph import analysis_db, neo4j_db
    files, frozen = _targets()
    ok = True

    _step('1/5 Site data (knowledge graph and frozen FAERS tables)')
    if bundle:
        with zipfile.ZipFile(bundle) as z:
            meta = json.loads(z.read('bundle.json'))
            for name, info in meta['files'].items():
                target = files.get(name) or frozen / name.split('/', 1)[1]
                if target.is_file() and target.stat().st_size == info['bytes'] and _sha256(target) == info['sha256']:
                    print(f'   {name}: already in place')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.part')
                with z.open(name) as src, open(tmp, 'wb') as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                if _sha256(tmp) != info['sha256']:
                    tmp.unlink()
                    raise SystemExit(f'{name}: checksum mismatch; the bundle is damaged, download it again')
                tmp.replace(target)
                print(f'   {name}: unpacked to {target.relative_to(ROOT)}')
    needed = [neo4j_db.KG] + [frozen / n for n in ('meddra_pt_hierarchy.csv', 'faers_reports.csv.gz',
                                                   'faers_report_strata.csv.gz', 'faers_report_pts.csv.gz')]
    absent = [str(p.relative_to(ROOT)) for p in needed if not p.is_file()]
    if absent:
        raise SystemExit(f'missing site data: {absent}. Download the site data bundle (askglp1-site-data-*.zip) from the '
                         'project Google Drive folder and run: make site-setup BUNDLE=<path to the zip>')
    print('   all site data present')

    _step('2/5 Neo4j knowledge graph')
    if neo4j_db.installation() is None:
        raise SystemExit('Neo4j is not installed. On macOS: brew install neo4j (it brings Java 21), then run this again.')
    current = neo4j_db.manifest() if neo4j_db.MANIFEST.is_file() else None
    if current and _inputs_unchanged(current.get('inputs_sha256')):
        neo4j_db.start()
        print('   graph already imported; database started')
    else:
        print('   importing the graph (a few minutes)...', flush=True)
        m = neo4j_db.load()
        print(f"   imported {m['db_counts']['nodes']:,} nodes and {m['db_counts']['relationships']:,} relationships")

    _step('3/5 Analysis database (read by the agent\'s code tool)')
    stale = True
    if analysis_db.PATH.is_file():
        try:
            import sqlite3
            con = sqlite3.connect(f'file:{analysis_db.PATH}?mode=ro', uri=True)
            built = json.loads(con.execute("SELECT value FROM build WHERE key = 'inputs_sha256'").fetchone()[0])
            con.close()
            stale = not _inputs_unchanged(built)
        except Exception:
            stale = True
    if stale:
        print('   building (about a minute)...', flush=True)
        analysis_db.build()
    print(f'   {analysis_db.PATH.relative_to(ROOT)} ready')

    _step('4/5 OpenAI key for the agent')
    from weightloss.agent import agent
    try:
        agent.load_key()
        from openai import AuthenticationError, OpenAI
        try:  # a free call: reading the model's description checks that the key is accepted
            OpenAI(max_retries=1, timeout=30).models.retrieve(agent.MODEL)
            print(f'   OPENAI_API_KEY accepted by OpenAI (model {agent.MODEL} available)')
        except AuthenticationError:
            ok = False
            print('   REJECTED: OpenAI does not accept the OPENAI_API_KEY in .env (or, without one there, in your shell '
                  'environment). Put a valid key in .env.')
        except Exception as error:
            ok = False
            print(f'   could not reach OpenAI to check the key ({type(error).__name__}); check the network and run this again.')
    except SystemExit:
        ok = False
        print('   MISSING: add a line OPENAI_API_KEY=... to the file .env in the repository folder (never commit it). '
              'Without it the graph pages work but the agent cannot answer.')

    _step('5/5 Python packages and frontend')
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        print('   pandas and numpy available for the agent\'s code tool')
    except ImportError:
        ok = False
        print('   MISSING pandas/numpy in this Python: run make install')
    if (ROOT / 'apps/web/dist/index.html').is_file():
        print('   frontend built')
    else:
        ok = False
        print('   frontend not built: run make web-install web-build')

    print('\nSetup complete. Start the site with: make demo   (then open http://127.0.0.1:5001)' if ok else
          '\nSetup finished with the items marked MISSING above; fix them and run this again.')
    return 0 if ok else 1
