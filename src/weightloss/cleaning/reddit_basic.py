"""Conservative pre-Jev preparation over ALL frozen v2 evidence, never v3 keeps.

Candidates are readable text, not relevant medication discussions or patients.
Old runs stay frozen. Different Reddit IDs are never collapsed by text alone.
"""
import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3

URL = re.compile(r'(?:https?://|www\.)[^\s<>]+', re.I)
WORDS = re.compile(r'[^\W\d_]+', re.UNICODE)
PLACEHOLDERS = {'[deleted]', '[removed]', '[removed by reddit]'}
# Full-segment matching only. Mentioning, discussing, or quoting a bot is not
# evidence that the whole observation was automatically generated.
SIGNATURE = re.compile(
    r'I am a bot,?\s+and this action was performed automatically\.?', re.I)
CONTACT = re.compile(
    r'Please contact the moderators of this subreddit if you have any questions or concerns\.?', re.I)
FULLNAME = re.compile(r't[13]_[a-z0-9]+\Z')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_policy(policy):
    if policy.get('schema_version') != 1 or type(policy.get('english_only')) is not bool:
        raise ValueError('Invalid basic policy')
    if not .9 <= policy['non_english_confidence'] <= 1 or policy['reserve_gib'] < 0:
        raise ValueError('Invalid language confidence/disk policy')
    for key in ('non_english_min_words', 'commit_every'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError('Invalid positive policy field')


def readable(text):
    return any(c.isalnum() for c in URL.sub(' ', text))


def template_only(text):
    compact = ' '.join(text.split())
    match = SIGNATURE.match(compact)
    return bool(match and (not compact[match.end():].strip() or
                          CONTACT.fullmatch(compact[match.end():].strip())))


def known_automation(feature, policy):
    """Only reviewed, whole-message templates in their source community.

    A new paragraph, quoted block, or human suffix invalidates the match.
    No fuzzy matching and no inferred account identity (authors unavailable).
    """
    segments = feature['segments']
    if not segments or any(s['role'] != 'prose' or s.get('field') != 'body' for s in segments):
        return None
    texts = [' '.join(s['text'].split()) for s in segments]
    for template in policy.get('automation_templates', []):
        if (feature.get('kind') == 'comments' and
                feature.get('subreddit', '').casefold() == template['subreddit'].casefold() and
                texts == [' '.join(t.split()) for t in template['segments']]):
            return template['id']
    return None


def courtesy_only(segments, policy):
    # Do not erase emoji or internal punctuation: they may add symptom content.
    phrases = set(policy.get('courtesy_templates', []))
    return bool(segments) and all(s['role'] == 'prose' and
        s['text'].casefold().strip(' \t\r\n.!?') in phrases for s in segments)


def language_scope(feature, retained, removed, policy):
    """Uncertainty passes onward. Cached scores are NOT calibrated accuracy.

No short-text cutoff for candidates. The length threshold only limits when the
cheap language detector may EXCLUDE. Mixed, incomplete or transformed evidence
cannot cause exclusion. Quotes/code/alt text keep their own attribution roles.
"""
    evidence = feature.get('language_evidence', {})
    status = evidence.get('status', 'undetermined')
    if not policy['english_only']:
        return dict(exclude=False, basis='all_languages_in_scope')
    if removed or any(s['role'] != 'prose' and readable(s['text']) for s in retained):
        return dict(exclude=False, basis='language_requires_semantic_screen')
    words = sum(len(WORDS.findall(URL.sub(' ', s['text']))) for s in retained)
    if words < policy['non_english_min_words']:
        return dict(exclude=False, basis='short_language_evidence_retained')
    pieces = [p for p in evidence.get('segments', []) if p.get('chars', 0) > 0]
    if status != 'non_english_supported' or not pieces:
        return dict(exclude=False, basis='english_mixed_or_uncertain_retained')
    predictions = pieces + ([evidence['whole']] if evidence.get('whole') else [])
    languages = {p.get('language') for p in predictions}
    complete = all(isinstance(p.get('score'), (int, float)) and
                   p['score'] >= policy['non_english_confidence'] for p in predictions)
    excluded = complete and len(languages) == 1 and not languages.intersection({None, 'en'})
    return dict(exclude=excluded, basis='consistent_non_english' if excluded else
                'incomplete_language_evidence_retained', scores_are_calibrated=False)


def prepare(feature, policy):
    """Build a role-preserving candidate; no drug/symptom/personal-use rules."""
    if not isinstance(feature.get('segments'), list):
        raise ValueError('Expected complete v2 segments, not filtered v3 features')
    original = feature['segments']
    if any(s.get('role') not in {'prose', 'quote', 'code', 'quoted_code', 'image_alt'}
           or not isinstance(s.get('text'), str) for s in original):
        raise ValueError('Malformed source segments')
    flags = set(feature.get('flags', []))
    automation_id = known_automation(feature, policy)
    signature_present = any(s['role'] == 'prose' and s.get('field') != 'title'
                            and template_only(s['text']) for s in original)
    retained, removed = [], []
    for index, segment in enumerate(original):
        text = segment['text'].strip()
        if automation_id:
            removed.append(dict(segment=index, reason='reviewed_automation_template'))
        elif not text or text.casefold() in PLACEHOLDERS:
            removed.append(dict(segment=index, reason='empty_or_placeholder'))
        elif segment['role'] == 'image_alt' and text.casefold() in {'img', 'image', 'gif'}:
            removed.append(dict(segment=index, reason='generic_media_label'))
        elif segment['role'] == 'prose' and segment.get('field') != 'title' and (
                template_only(text) or signature_present and CONTACT.fullmatch(' '.join(text.split()))):
            removed.append(dict(segment=index, reason='exact_automation_boilerplate'))
        else:
            retained.append(copy.deepcopy(segment))
    usable = any(readable(s['text']) for s in retained)
    language = language_scope(feature, retained, removed, policy)
    if not usable:
        reason = ('automation_template_only' if any(r['reason'] in {'exact_automation_boilerplate', 'reviewed_automation_template'}
                  for r in removed) else 'no_readable_text')
    elif courtesy_only(retained, policy):
        reason = 'courtesy_template_only'
    elif language['exclude']:
        reason = 'outside_english_scope'
    else:
        reason = 'ready_for_semantic_screen'
    if any(s['role'] in {'quote', 'quoted_code'} for s in retained):
        flags.add('quoted_material_retained_not_current_speaker')
    if usable and not any(s['role'] == 'prose' and readable(s['text']) for s in retained):
        flags.add('no_unquoted_prose_semantic_judgment_deferred')
    context = copy.deepcopy(feature.get('context', {}))
    issues = set(context.get('issues', [])) - {'depth_limit'}
    nodes = list(context.get('ancestors', [])) + ([context['thread']] if context.get('thread') else [])
    references, seen = [], {feature['fullname']}
    if issues:
        flags.add('context_unusable_body_retained')
    else:
        for node in nodes:
            ident = node.get('fullname')
            if not isinstance(ident, str) or not FULLNAME.fullmatch(ident):
                flags.add('context_unusable_body_retained')
                references = []
                break
            if ident not in seen:
                seen.add(ident)
                references.append(dict(fullname=ident, provenance=node.get('provenance'),
                                       role='context_only_not_current_speaker'))
    if feature.get('crosspost_parent'):
        flags.add('crosspost_retained')
    return dict(fullname=feature['fullname'], decision='candidate' if reason == 'ready_for_semantic_screen' else 'exclude',
                reason=reason, segments=retained, removed_segments=removed,
                context_references=references, context_issues=context.get('issues', []),
                flags=sorted(flags), language_scope=language,
                automation_template_id=automation_id,
                clinical_eligibility='not_assessed', semantic_screen='pending',
                attribution='segment_roles_not_verified_authorship')


SCHEMA = '''
CREATE TABLE IF NOT EXISTS progress (id INTEGER PRIMARY KEY CHECK(id=1), offset INTEGER NOT NULL);
INSERT OR IGNORE INTO progress VALUES (1,0);
CREATE TABLE IF NOT EXISTS decisions (
 source_rowid INTEGER PRIMARY KEY, fullname TEXT UNIQUE NOT NULL,
 decision TEXT NOT NULL CHECK(decision IN ('candidate','exclude')), reason TEXT NOT NULL,
 source_feature_sha256 TEXT NOT NULL, evidence_json TEXT NOT NULL);
CREATE VIEW IF NOT EXISTS candidate_records AS SELECT * FROM decisions WHERE decision='candidate';
CREATE VIEW IF NOT EXISTS excluded_records AS SELECT * FROM decisions WHERE decision='exclude';
'''


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run(source_dir, output_dir, policy, sample_size=None, seed=20260925, max_records=None,
        preparer=None, supporting_code_paths=()):
    """Resumable lean overlay; sample output is never labeled a full release."""
    validate_policy(policy)
    preparer = preparer or prepare
    for number in (sample_size, max_records):
        if number is not None and (type(number) is not int or number < 1):
            raise ValueError('Sample/record limit must be positive')
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    source_spec_path = source_dir / 'run-spec.json'
    source_spec = json.loads(source_spec_path.read_text())
    baseline = Path(source_spec['baseline_directory']).resolve()
    base_spec = json.loads((baseline / 'run-spec.json').read_text())
    production = Path(base_spec['production_directory']).resolve()
    for protected in (source_dir, baseline, production):
        if output_dir == protected or output_dir.is_relative_to(protected) or protected.is_relative_to(output_dir):
            raise ValueError('Output must be separate from source and frozen runs')
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir / 'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path, mode in ((source_dir / '.cleaning.lock', fcntl.LOCK_SH),
                           (output_dir / '.cleaning.lock', fcntl.LOCK_EX)):
            handle = stack.enter_context(path.open('r' if mode == fcntl.LOCK_SH else 'a'))
            try:
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Cleaner is active; checkpoint preserved') from error
        wal = source_dir / 'cleaned.sqlite-wal'
        if wal.exists() and wal.stat().st_size:
            raise ValueError('Source has an uncheckpointed WAL; requires a frozen snapshot')
        source = sqlite3.connect(f'file:{source_dir / "cleaned.sqlite"}?mode=ro', uri=True)
        stack.callback(source.close)
        source.execute('PRAGMA query_only=ON')
        if source.execute('SELECT count(*) FROM inputs WHERE done!=1').fetchone()[0]:
            raise ValueError('Requires a completed frozen v2 source')
        # Bind the whole cached evidence snapshot, not just the old policy file.
        # Streaming hash reads the source without copying tens of GB of text.
        fingerprint = dict(schema_version=1, source=str(source_dir), policy=policy,
                           code_sha256=file_digest(__file__), source_spec_sha256=file_digest(source_spec_path),
                           source_database_sha256=file_digest(source_dir / 'cleaned.sqlite'),
                           mode='sample' if sample_size else 'full', sample_size=sample_size, seed=seed)
        fingerprint['supporting_code_sha256'] = {str(Path(p).resolve()): file_digest(p) for p in supporting_code_paths}
        fingerprint['preparer'] = preparer.__module__ + '.' + preparer.__name__
        fp = output_dir / 'run-spec.json'
        if fp.exists() and json.loads(fp.read_text()) != fingerprint:
            raise ValueError('Rules/source/sample changed; use a new run')
        if not fp.exists():
            write_json(fp, fingerprint)
        # Preserve exact runnable source alongside each new run for later audits.
        archive = output_dir / 'source'
        archive.mkdir(exist_ok=True)
        for path in (Path(__file__), *(Path(p) for p in supporting_code_paths)):
            shutil.copyfile(path, archive / path.name)
        db = sqlite3.connect(output_dir / 'candidates.sqlite')
        stack.callback(db.close)
        db.executescript(SCHEMA)
        offset = db.execute('SELECT offset FROM progress WHERE id=1').fetchone()[0]
        expected = source.execute('SELECT sum(expected_rows) FROM inputs').fetchone()[0] or 0
        maximum = source.execute('SELECT max(rowid) FROM records').fetchone()[0] or 0
        minimum = source.execute('SELECT min(rowid) FROM records').fetchone()[0]
        selected = None
        if sample_size:
            if minimum != 1 or maximum != expected:
                raise ValueError('Sample requires dense frozen source rowids')
            selected = sorted(random.Random(seed).sample(range(1, maximum + 1), min(sample_size, maximum)))
        processed = 0
        while max_records is None or processed < max_records:
            if shutil.disk_usage(output_dir).free < policy['reserve_gib'] * 2**30:
                raise ValueError('Disk reserve reached; checkpoint retained')
            limit = policy['commit_every']
            if max_records is not None:
                limit = min(limit, max_records - processed)
            if selected is not None:
                batch_ids = [i for i in selected if i > offset][:limit]
                rows = [source.execute('SELECT rowid,fullname,features_json FROM records WHERE rowid=?', (i,)).fetchone()
                        for i in batch_ids]
                if any(row is None for row in rows):
                    raise ValueError('Missing sampled source row')
            else:
                rows = source.execute('SELECT rowid,fullname,features_json FROM records WHERE rowid>? ORDER BY rowid LIMIT ?',
                                      (offset, limit)).fetchall()
            if not rows:
                break
            with db:
                for rid, ident, payload in rows:
                    feature = json.loads(payload)
                    if feature['fullname'] != ident:
                        raise ValueError('Source identity mismatch')
                    result = preparer(feature, policy)
                    # Text remains only in the source; reconstruct with prepare()
                    # after checking the stored feature digest for downstream use.
                    evidence = {k: v for k, v in result.items() if k not in {'segments', 'fullname', 'decision', 'reason'}}
                    db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)',
                               (rid, ident, result['decision'], result['reason'], digest(payload), canonical(evidence)))
                    offset = rid
                db.execute('UPDATE progress SET offset=? WHERE id=1', (offset,))
            processed += len(rows)
        count = db.execute('SELECT count(*) FROM decisions').fetchone()[0]
        target = len(selected) if selected is not None else expected
        if count > target or (not rows and count != target):
            raise ValueError('Source count mismatch')
        report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                      status=('sample_complete' if sample_size else 'full_basic_complete') if count == target else 'partial',
                      mode=fingerprint['mode'], source_records=expected, target_records=target, processed_records=count,
                      processed_this_invocation=processed,
                      dispositions=dict(db.execute('SELECT decision,count(*) FROM decisions GROUP BY decision')),
                      reasons=dict(db.execute('SELECT reason,count(*) FROM decisions GROUP BY reason')),
                      independent_accuracy=None, clinical_eligibility='not_assessed', semantic_screen='pending',
                      duplicate_policy='unique_source_identity; different IDs retained even with identical text',
                      fingerprint=fingerprint)
        write_json(output_dir / 'manifest.json', report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('.cache/reddit-cleaning/v2-strict'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-basic.json'))
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--max-records', type=int)
    args = parser.parse_args()
    report = run(args.source_dir, args.output_dir, json.loads(args.policy.read_text()),
                 args.sample_size, args.seed, args.max_records)
    print(json.dumps({k: report[k] for k in ('status', 'processed_records', 'dispositions', 'reasons')}, indent=2))


if __name__ == '__main__':
    main()
