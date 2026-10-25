"""Read immutable verified exports; write resumable, auditable derived data only.

Inspired by the AMIA semaglutide project's preparation design, independently
implemented for flat Arctic Shift records. No network or raw-database writes.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata

from weightloss.ingestion.reddit import sha256_file, utc
from weightloss.ingestion.reddit_production import source_done, write_json

URL = re.compile(r'(?:https?://|www\.)[^\s<>]+', re.I)
IMAGE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
WORDS = re.compile(r'[^\W\d_]+', re.UNICODE)
FULLNAME = re.compile(r't[13]_[a-z0-9]+\Z')
PLACEHOLDERS = {'[deleted]', '[removed]', '[removed by reddit]'}
BOT_TEMPLATE = re.compile(r"\bi\s+am\s+a\s+bot\b[\s\S]{0,100}\bthis\s+action\s+was\s+performed\s+automatically\b", re.I)


def normalize(text):
    if text is None:
        return ''
    if not isinstance(text, str):
        raise ValueError('Expected a string or null text field')
    text = unicodedata.normalize('NFC', html.unescape(text)).replace('\r\n', '\n').replace('\r', '\n')
    text = ''.join(c for c in text if c in '\n\t' or unicodedata.category(c) != 'Cc').replace('\u200b', '')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


class FastTextLanguage:
    def __init__(self, model_path):
        import fasttext
        path = Path(model_path).expanduser().resolve()
        self.fingerprint = {'engine':'fasttext', 'package':'fasttext-wheel',
                            'package_version':version('fasttext-wheel'),
                            'model_sha256':sha256_file(path), 'model_file':path.name}
        self.model = fasttext.load_model(str(path))

    def predict(self, text):
        labels, scores = self.model.predict(' '.join(text.split()), k=1)
        return labels[0].removeprefix('__label__'), min(1.0, max(0.0, float(scores[0])))


def clean_record(row, kind, spec, detector, policy, source_absent=False):
    """A disposition is reversible; raw values remain in the referenced export."""
    if kind not in ('posts', 'comments') or not isinstance(row, dict):
        raise ValueError('Invalid record type')
    ident, stamp = row.get('id'), row.get('created_utc')
    if not isinstance(ident, str) or not re.fullmatch('[a-z0-9]+', ident):
        raise ValueError('Invalid Reddit ID')
    if type(stamp) is not int or not spec['start'] <= stamp < spec['end']:
        raise ValueError('Record outside frozen FDA/source range')
    if row.get('subreddit', '').casefold() != spec['subreddit']:
        raise ValueError('Record community differs from input manifest')
    flags = []
    title = normalize(row.get('title')) if kind == 'posts' else ''
    body = normalize(row.get('selftext') if kind == 'posts' else row.get('body'))
    unavailable = body.casefold() in PLACEHOLDERS
    if unavailable:
        body = ''
        flags.append('body_unavailable')
    if title.casefold() in PLACEHOLDERS:
        title = ''
        flags.append('title_unavailable')
    text = '\n\n'.join(p for p in [title, body] if p)
    fullname = ('t3_' if kind == 'posts' else 't1_') + ident
    parent, link = row.get('parent_id'), row.get('link_id')
    for value in [parent, link, row.get('crosspost_parent')]:
        if value is not None and (not isinstance(value, str) or not FULLNAME.fullmatch(value)):
            flags.append('invalid_relationship_id')
    if kind == 'comments' and (not parent or not link):
        flags.append('missing_relationship_id')
    if kind == 'comments' and link and not link.startswith('t3_'):
        flags.append('invalid_relationship_id')
    if source_absent:
        flags.append('documented_source_absence')
    if BOT_TEMPLATE.search(text):
        flags.append('possible_bot_template')
    if row.get('crosspost_parent'):
        flags.append('crosspost')
    classification_text = URL.sub(' ', IMAGE.sub(' ', text))
    words = WORDS.findall(classification_text)
    decision, reason, lang, confidence = 'keep', 'text_eligible', None, None
    if not text:
        decision, reason = 'exclude', 'unavailable_text' if unavailable else 'empty_text'
    elif not any(c.isalnum() for c in classification_text):
        decision, reason = 'exclude', ('url_or_media_only' if URL.search(text) or IMAGE.search(text) else 'no_alphanumeric_text')
    elif len(words) < policy['language_min_words']:
        decision, reason = 'review', 'short_text'
    else:
        lang, confidence = detector.predict(classification_text[:policy['language_max_chars']])
        if not isinstance(lang, str) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError('Invalid language classifier response')
        if confidence < policy['language_confidence']:
            decision, reason = 'review', 'uncertain_language'
        elif lang != 'en':
            decision, reason = 'exclude', 'non_english_high_confidence'
    if decision == 'keep' and any(f in flags for f in ['possible_bot_template', 'invalid_relationship_id', 'missing_relationship_id']):
        decision, reason = 'review', 'structural_or_bot_review'
    return dict(fullname=fullname, id=ident, kind=kind,
        document_type=('post' if kind == 'posts' else 'reply' if isinstance(parent,str) and parent.startswith('t1_') else 'comment'),
        subreddit=spec['subreddit'], created_utc=stamp,
        date_utc=datetime.fromtimestamp(stamp, timezone.utc).date().isoformat(),
        text=text, title=title if kind=='posts' else None, parent_id=parent, link_id=link,
        crosspost_parent=row.get('crosspost_parent'), score=row.get('score'),
        num_comments=row.get('num_comments'), word_count=len(words), language=lang,
        language_score=confidence, decision=decision, reason=reason, flags=sorted(set(flags)),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        author_bot_assessment='unavailable_author_not_collected')


def discover(production, config, selected=None):
    scope = json.loads((production/'production-spec.json').read_text())
    if set(scope['targets']) != set(config['targets']):
        raise ValueError('Production inventory differs from active configuration')
    wanted = set(selected or config['targets'])
    if not wanted <= set(config['targets']) or wanted.intersection(config.get('excluded_subreddits', [])):
        raise ValueError('Cleaning requested an unconfigured/excluded source')
    inputs, pending = [], []
    for name in sorted(wanted):
        path = production/name/'manifest.json'
        if not path.exists():
            pending.append(name); continue
        raw = path.read_bytes(); manifest = json.loads(raw)
        if not source_done(manifest):
            pending.append(name); continue
        expected = scope['targets'][name]
        spec = manifest['spec']
        if any(spec.get(k) != v for k,v in expected.items()) or spec['end'] != scope['snapshot_end_exclusive'] or spec.get('direction') != 'asc':
            raise ValueError('Manifest differs from frozen primary scope')
        for kind in ['posts','comments']:
            f = (production/name/manifest['export_files'][kind]).resolve()
            if not f.is_relative_to((production/name/'exports').resolve()):
                raise ValueError('Expected an immutable export inside its community directory')
            inputs.append(dict(key=name+'/'+kind, path=str(f), kind=kind, subreddit=name,
                sha256=manifest['files_sha256'][kind+'.jsonl'], expected_rows=manifest['counts'][kind]['count'],
                spec=spec, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                verification=manifest['verification'], context_audit=manifest['context_audit']))
    return scope, inputs, pending


def summarize(db, directory, scope, pending, selected=None):
    decisions = dict(db.execute('SELECT decision,count(*) FROM records GROUP BY decision'))
    sources = [dict(zip(['key','rows_read','expected_rows','done'],r)) for r in db.execute('SELECT key,lines,expected_rows,done FROM inputs ORDER BY key')]
    unfinished = [r['key'] for r in sources if not r['done']]
    report = dict(schema_version=1, generated_at=utc(datetime.now(timezone.utc).timestamp()),
        status='partial' if unfinished else 'ready_sources_cleaned_waiting_for_collection' if pending else 'selected_sources_cleaned' if selected else 'all_sources_cleaned',
        production_scope_count=len(scope['targets']), pending_collection=pending, unfinished_cleaning=unfinished,
        selected_scope_count=len(selected or scope['targets']),
        unique_records=sum(decisions.values()), dispositions=decisions,
        reasons=dict(db.execute('SELECT reason,count(*) FROM records GROUP BY reason')),
        by_community=[dict(zip(['subreddit','decision','count'],r)) for r in db.execute('SELECT i.subreddit,r.decision,count(*) FROM records r JOIN inputs i ON i.key=r.input_key GROUP BY 1,2 ORDER BY 1,2')],
        duplicate_id_rows=db.execute('SELECT count(*) FROM duplicates').fetchone()[0],
        same_text_multiple_ids=db.execute("SELECT count(*) FROM (SELECT text_hash FROM records WHERE text_length>0 GROUP BY text_hash HAVING count(*)>1)").fetchone()[0],
        inputs=sources, database=str(directory/'cleaned.sqlite'),
        limitations=['Stage1 text eligibility only: not drug attribution, personal use or adverse-effect classification.',
                      'Language scores are not calibrated probabilities; short/uncertain text remains in review.',
                      'Author metadata was not collected; comprehensive bot-account removal is unavailable.',
                      'Same text under different Reddit IDs is retained and grouped by text_sha256.',
                      'Context-only parent records are not counted as primary observations; relationships remain available in raw checkpoints.'])
    return report


def clean_production(production_dir, output_dir, report_file, policy, config, detector,
                     selected=None, max_records=None):
    production, directory, report_path = (Path(p).expanduser().resolve() for p in (production_dir, output_dir, report_file))
    if directory == production or directory.is_relative_to(production) or production.is_relative_to(directory):
        raise ValueError('Cleaning output must be separate from production')
    if report_path == directory or report_path.is_relative_to(production) or report_path == production:
        raise ValueError('Report must not replace raw data or output directory')
    if report_path.suffix != '.json':
        raise ValueError('Report requires a JSON path')
    if max_records is not None and max_records < 1:
        raise ValueError('max_records must be positive')
    if policy.get('schema_version') != 1 or not 0.5 <= policy['language_confidence'] <= 1 or policy['language_min_words'] < 1 or policy['language_max_chars'] < 1 or policy['commit_every'] < 1 or policy['reserve_gib'] < 0:
        raise ValueError('Invalid cleaning policy')
    scope, inputs, pending = discover(production, config, selected)
    fingerprint = dict(schema_version=1, production_directory=str(production), production_scope=scope,
        policy=policy, configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        cleaner_sha256=sha256_file(Path(__file__)), language=detector.fingerprint,
        selected=sorted(selected) if selected else None)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Refusing a nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.cleaning.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another cleaner owns this output directory')
        spec_path = directory/'run-spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != fingerprint:
            raise ValueError('Cleaning rules/model/scope changed; use a new derived run')
        if not spec_path.exists(): write_json(spec_path, fingerprint)
        db = sqlite3.connect(directory/'cleaned.sqlite')
        try:
            db.executescript('''CREATE TABLE IF NOT EXISTS inputs (
                key TEXT PRIMARY KEY,subreddit TEXT,kind TEXT,path TEXT,sha256 TEXT,expected_rows INTEGER,
                offset INTEGER DEFAULT 0,lines INTEGER DEFAULT 0,done INTEGER DEFAULT 0,metadata TEXT);
                CREATE TABLE IF NOT EXISTS records (fullname TEXT PRIMARY KEY,input_key TEXT,line INTEGER,
                raw_hash TEXT,text_hash TEXT,text_length INTEGER,decision TEXT,reason TEXT,record_json TEXT);
                CREATE INDEX IF NOT EXISTS records_input ON records(input_key);
                CREATE INDEX IF NOT EXISTS records_text ON records(text_hash);
                CREATE TABLE IF NOT EXISTS duplicates (input_key TEXT,line INTEGER,fullname TEXT,raw_hash TEXT,
                PRIMARY KEY(input_key,line));
                CREATE VIEW IF NOT EXISTS kept_records AS SELECT * FROM records WHERE decision='keep';
                CREATE VIEW IF NOT EXISTS review_records AS SELECT * FROM records WHERE decision='review';
                CREATE VIEW IF NOT EXISTS excluded_records AS SELECT * FROM records WHERE decision='exclude';''')
            known = {r[0] for r in db.execute('SELECT key FROM inputs')}
            if not known <= {i['key'] for i in inputs}:
                raise ValueError('A previously selected source is no longer verified/available')
            for entry in inputs:
                old = db.execute('SELECT path,sha256,expected_rows FROM inputs WHERE key=?',(entry['key'],)).fetchone()
                if old and old != (entry['path'], entry['sha256'], entry['expected_rows']):
                    raise ValueError('Frozen source export changed; use a new derived run')
                with db:
                    db.execute('INSERT OR IGNORE INTO inputs(key,subreddit,kind,path,sha256,expected_rows,metadata) VALUES (?,?,?,?,?,?,?)',
                               (entry['key'],entry['subreddit'],entry['kind'],entry['path'],entry['sha256'],entry['expected_rows'],json.dumps(entry)))
            processed = 0
            for entry in inputs:
                key = entry['key'];offset,lines,done = db.execute('SELECT offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
                path = Path(entry['path'])
                if sha256_file(path) != entry['sha256']:
                    raise ValueError('Source export hash mismatch')
                if done: continue
                absent = {(r['kind'],r['id']) for r in entry['verification'].get('documented_source_absences',[])}
                with path.open('rb') as stream:
                    stream.seek(offset)
                    eof = False
                    while not eof and (max_records is None or processed < max_records):
                        if shutil.disk_usage(directory).free < policy['reserve_gib']*2**30:
                            raise ValueError('Disk reserve reached; cleaning checkpoint retained')
                        with db:
                            for _ in range(policy['commit_every']):
                                if max_records is not None and processed >= max_records: break
                                line = stream.readline()
                                if not line: eof=True; break
                                lines += 1
                                row = json.loads(line)
                                record = clean_record(row,entry['kind'],entry['spec'],detector,policy,(entry['kind'],row.get('id')) in absent)
                                raw_hash = hashlib.sha256(line).hexdigest()
                                record['provenance'] = dict(input_key=key,line=lines,raw_line_sha256=raw_hash)
                                old = db.execute('SELECT raw_hash FROM records WHERE fullname=?',(record['fullname'],)).fetchone()
                                if old:
                                    if old[0] != raw_hash: raise ValueError('Conflicting versions of the same Reddit fullname')
                                    db.execute('INSERT INTO duplicates VALUES (?,?,?,?)',(key,lines,record['fullname'],raw_hash))
                                else:
                                    db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)',(record['fullname'],key,lines,raw_hash,record['text_sha256'],len(record['text']),record['decision'],record['reason'],json.dumps(record,ensure_ascii=False)))
                                processed += 1
                            db.execute('UPDATE inputs SET offset=?,lines=? WHERE key=?',(stream.tell(),lines,key))
                        if eof:
                            if lines != entry['expected_rows'] or sha256_file(path) != entry['sha256']:
                                raise ValueError('Source count/hash changed; input not accepted as complete')
                            with db: db.execute('UPDATE inputs SET done=1 WHERE key=?',(key,))
                    if max_records is not None and processed >= max_records: break
            report = summarize(db,directory,scope,pending,selected)
            report['records_processed_this_invocation'] = processed
            report_path.parent.mkdir(parents=True,exist_ok=True)
            write_json(report_path,report)
            return report
        finally: db.close()
