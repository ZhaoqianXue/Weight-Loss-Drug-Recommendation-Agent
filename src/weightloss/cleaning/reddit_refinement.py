"""Independent, context-aware cleaning candidate and reproducible v1 comparison.

All output is additive. No raw or v1 records are modified. A sample comparison
measures routing changes, not accuracy, and never certifies personal drug use.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata

from weightloss.cleaning.reddit import (
    BOT_TEMPLATE, FULLNAME, PLACEHOLDERS, URL, WORDS, FastTextLanguage,
)
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json

PROMOTION = re.compile(r'\b(?:use\s+my\s+(?:code|link)|my\s+(?:referral|affiliate)\s+(?:code|link)|'
                       r'(?:dm|message)\s+me\s+(?:for|to\s+(?:buy|order))|'
                       r'(?:buy|order)\s+now|(?:guaranteed|miracle)\s+(?:weight\s*loss|cure))\b', re.I)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def source_text(value):
    """Preserve indentation and entities until the Markdown parser sees them."""
    if value is None:
        return ''
    if not isinstance(value, str):
        raise ValueError('Expected text or null')
    if value.strip().casefold() in PLACEHOLDERS:
        return ''
    return unicodedata.normalize('NFC', value).replace('\r\n', '\n').replace('\r', '\n')


class TextSegments:
    """Syntactic attribution only: unquoted prose is NOT proof of authorship."""
    def __init__(self):
        from markdown_it import MarkdownIt
        self.parser = MarkdownIt('commonmark', {'html': False})
        self.fingerprint = dict(parser='markdown-it-py', version=version('markdown-it-py'),
                                preset='commonmark', html=False)

    def parse(self, raw, kind):
        fields = [('title', raw.get('title')), ('selftext', raw.get('selftext'))] if kind == 'posts' else [('body', raw.get('body'))]
        segments, flags = [], set()
        for field, value in fields:
            text = source_text(value)
            # Titles are plain text on Reddit; leading '>' or '#' is not markup.
            if field == 'title':
                if text:
                    segments.append(dict(role='prose', field=field, lines=None, text=text))
                continue
            depth = 0
            for token in self.parser.parse(text):
                if token.type == 'blockquote_open':
                    depth += 1
                elif token.type == 'blockquote_close':
                    depth -= 1
                elif token.type in ('fence', 'code_block'):
                    segments.append(dict(role='quoted_code' if depth else 'code', field=field,
                                         lines=token.map, text=token.content))
                elif token.type == 'inline':
                    chunks = []
                    for child in token.children or []:
                        if child.type in ('text', 'code_inline'):
                            chunks.append(child.content)
                            if child.type == 'code_inline':
                                flags.add('inline_code_retained')
                        elif child.type in ('softbreak', 'hardbreak'):
                            chunks.append('\n')
                        elif child.type == 'image':
                            flags.add('has_media')
                            segments.append(dict(role='image_alt', field=field, lines=token.map,
                                                 text=child.content))
                            chunks.append(' ')
                    prose = ''.join(chunks).strip()
                    if prose:
                        segments.append(dict(role='quote' if depth else 'prose', field=field,
                                             lines=token.map, text=prose))
        return segments, flags


def language_evidence(segments, detector, policy):
    """Every prose character is covered; no silent first-20k truncation.

    Very short segments remain undecided. Whole-message prediction helps with
    fragmented lists; a confident segment disagreement forces review.
    """
    prose = [s for s in segments if s['role'] == 'prose']
    pieces = []
    for index, segment in enumerate(prose):
        text = URL.sub(' ', segment['text'])
        for offset in range(0, len(text), policy['language_segment_chars']):
            chunk = text[offset:offset+policy['language_segment_chars']]
            count = len(WORDS.findall(chunk))
            # CJK has few whitespace words: allow sufficiently long script text.
            cjk = sum('\u3400' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in chunk)
            lang, score = detector.predict(chunk) if count >= policy['language_min_words'] or cjk >= 12 else (None, None)
            pieces.append(dict(segment=index, offset=offset, chars=len(chunk), words=count,
                               sha256=digest(chunk), language=lang, score=score))
    text = '\n\n'.join(URL.sub(' ', s['text']) for s in prose)
    words = len(WORDS.findall(text))
    # Whole-message score is optional for long messages; all chunks still run.
    whole = None
    if text and len(text) <= policy['language_segment_chars'] and (
            words >= policy['language_min_words'] or any(p['language'] for p in pieces)):
        lang, score = detector.predict(text)
        whole = dict(language=lang, score=score)
    predictions = [p for p in pieces if p['language']] + ([whole] if whole else [])
    for p in predictions:
        if not isinstance(p['language'], str) or not isinstance(p['score'], (int, float)) or not 0 <= p['score'] <= 1:
            raise ValueError('Invalid language classifier response')
    strong = {p['language'] for p in predictions if p['score'] >= policy['language_confidence']}
    if len(strong) > 1:
        status = 'mixed_or_conflicting'
    elif whole and whole['score'] >= policy['language_confidence']:
        status = 'english_supported' if whole['language'] == 'en' else 'non_english_supported'
    elif not whole and pieces and all(p['language'] and p['score'] >= policy['language_confidence'] for p in pieces) and len(strong) == 1:
        status = 'english_supported' if strong == {'en'} else 'non_english_supported'
    else:
        status = 'undetermined'
    return dict(status=status, whole=whole, segments=pieces, words=words,
                threshold=policy['language_confidence'], scores_are_calibrated=False)


def context_bundle(record, lookup, policy):
    if record['kind'] == 'posts':
        return dict(status='root_post', ancestors=[], thread=None, issues=[])
    ancestors, issues, visited = [], [], {record['fullname']}
    pointer = record.get('parent_id')
    link = record.get('link_id')

    def fetch(ident, child_time):
        if not isinstance(ident, str) or not FULLNAME.fullmatch(ident):
            issues.append('invalid_relationship'); return None
        parent = lookup(ident)
        if not parent:
            issues.append('missing_in_completed_primary_view'); return None
        if parent['subreddit'] != record['subreddit']:
            issues.append('cross_community_relationship'); return None
        if ident.startswith('t3_') and ident != link or ident.startswith('t1_') and parent.get('link_id') != link:
            issues.append('inconsistent_thread_relationship'); return None
        if parent['created_utc'] > child_time:
            issues.append('parent_timestamp_after_child'); return None
        return parent

    def excerpt(parent):
        text = parent['text']
        return dict(fullname=parent['fullname'], text=text[:policy['context_excerpt_chars']],
                    truncated=len(text) > policy['context_excerpt_chars'],
                    source_text_sha256=parent['text_sha256'],
                    language=parent['language'], language_score=parent['language_score'],
                    baseline_decision=parent['decision'],
                    evidence_role='context_only_not_current_speaker',
                    provenance=parent['provenance'])

    child_time = record['created_utc']
    for _ in range(policy['context_depth']):
        if isinstance(pointer, str) and pointer in visited:
            issues.append('cycle'); break
        parent = fetch(pointer, child_time)
        if parent is None:
            break
        visited.add(pointer); ancestors.append(excerpt(parent))
        if pointer.startswith('t3_'):
            break
        pointer, child_time = parent.get('parent_id'), parent['created_utc']
    else:
        if pointer and pointer not in visited:
            issues.append('depth_limit')
    thread = next((a for a in ancestors if a['fullname'] == link), None)
    if thread is None:
        root = fetch(link, record['created_utc'])
        if root:
            thread = excerpt(root)
    return dict(status='available' if ancestors else 'unresolved', ancestors=ancestors,
                thread=thread, issues=sorted(set(issues)))


def refine(raw, baseline, detector, parser, lookup, policy):
    segments, flags = parser.parse(raw, baseline['kind'])
    prose = '\n\n'.join(s['text'] for s in segments if s['role'] == 'prose')
    quoted = '\n\n'.join(s['text'] for s in segments if s['role'] in ('quote', 'quoted_code'))
    if quoted:
        flags.add('has_quoted_material')
    if any(s['role'] in ('code', 'quoted_code') for s in segments):
        flags.add('has_code_block')
    bot = bool(BOT_TEMPLATE.search(prose))
    if bot:
        flags.add('unquoted_automation_disclosure')
    if BOT_TEMPLATE.search(quoted):
        flags.add('quoted_automation_disclosure')
    # Clear only v1's text-wide heuristic; other source/structure flags survive.
    flags.update(f for f in baseline['flags'] if f != 'possible_bot_template')
    evidence = language_evidence(segments, detector, policy)
    context = context_bundle(baseline, lookup, policy)
    if context['issues']:
        flags.add('context_has_gaps')
    if baseline['crosspost_parent']:
        flags.add('crosspost_lineage')
    cleaned_prose = URL.sub(' ', prose).strip()
    target_mentions = sorted(term for term in policy.get('target_terms', [])
                             if re.search(r'\b'+re.escape(term)+r'\b', cleaned_prose, re.I))
    short = (evidence['words'] < policy['minimum_prose_words'] or
             len(cleaned_prose) < policy['minimum_prose_chars'])
    if not baseline['text']:
        reason = 'unavailable'
    elif bot:
        reason = 'automation_disclosure'
    elif {'invalid_relationship_id', 'missing_relationship_id'} & flags:
        reason = 'invalid_structure'
    elif not any(c.isalnum() for c in cleaned_prose):
        reason = 'no_original_prose'
    elif short:
        reason = 'short_or_sparse_prose'
    elif policy.get('require_explicit_target_mention', False) and not target_mentions:
        reason = 'no_explicit_target_mention'
    elif evidence['status'] == 'mixed_or_conflicting':
        reason = 'mixed_or_conflicting_language'
    elif evidence['status'] == 'non_english_supported':
        reason = 'non_english'
    elif evidence['status'] == 'undetermined':
        reason = 'uncertain_language'
    elif any(p['language'] and (p['language'] != 'en' or p['score'] < policy['language_confidence'])
             for p in evidence['segments']):
        reason = 'uncertain_language_segment'
    elif PROMOTION.search(prose):
        reason = 'promotional_pattern'
    elif baseline['crosspost_parent']:
        reason = 'crosspost'
    elif policy['require_comment_context'] and baseline['kind'] == 'comments' and (
        context['status'] != 'available' or not context['ancestors'][0]['text'] or
        not context['thread'] or not context['thread']['text'] or
        set(context['issues']) - {'depth_limit'}):
        reason = 'unresolved_context'
    else:
        reason = 'quality_eligible'
    decision = 'keep' if reason == 'quality_eligible' else 'exclude'
    return dict(fullname=baseline['fullname'], subreddit=baseline['subreddit'], kind=baseline['kind'],
                provenance=baseline['provenance'], baseline_decision=baseline['decision'],
                baseline_reason=baseline['reason'], decision=decision, reason=reason, flags=sorted(flags),
                segments=segments, prose_text=prose, language_evidence=evidence, context=context,
                crosspost_parent=baseline['crosspost_parent'],
                target_mentions=target_mentions,
                exact_text_group=baseline['text_sha256'],
                clinical_eligibility='not_assessed', attribution='syntax_only_not_personal_experience')


def boundaries(record):
    text = record['text']
    return [name for name, found in [
        ('quote_marker', bool(re.search(r'(?m)^\s*>', text))),
        ('code_marker', '```' in text or '~~~' in text),
        ('automation_template', 'possible_bot_template' in record['flags']),
        ('crosspost', bool(record.get('crosspost_parent'))),
        ('long_text', len(text) > 20000),
        ('non_english', record['reason'] == 'non_english_high_confidence'),
        ('short_reply', record['kind'] == 'comments' and record['reason'] == 'short_text'),
    ] if found]


def bottom_sample(heap, priority, fullname, limit):
    item = (-priority, fullname)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def sample_population(db, policy):
    counts, heaps = Counter(), defaultdict(list)
    for (payload,) in db.execute('SELECT record_json FROM records WHERE input_key IN (SELECT key FROM inputs WHERE done=1)'):
        r = json.loads(payload)
        strata = ['stratum/'+r['subreddit']+'/'+r['kind']+'/'+r['reason']]
        strata += ['boundary/'+b for b in boundaries(r)]
        priority = int(digest(policy['sampling_seed']+'\0'+r['fullname']), 16)
        for key in strata:
            counts[key] += 1
            limit = policy['sample_per_boundary'] if key.startswith('boundary/') else policy['sample_per_stratum']
            bottom_sample(heaps[key], priority, r['fullname'], limit)
    memberships = defaultdict(list)
    for key, heap in heaps.items():
        for _, fullname in heap:
            memberships[fullname].append(key)
    return memberships, dict(sorted(counts.items())), {k: len(h) for k, h in sorted(heaps.items())}


def audit(baseline_dir, output_dir, report_path, model_path, policy):
    baseline_dir, output_dir, report_path = (Path(p).resolve() for p in (baseline_dir, output_dir, report_path))
    base_spec = json.loads((baseline_dir/'run-spec.json').read_text())
    production = Path(base_spec['production_directory']).resolve()
    for protected in (baseline_dir, production):
        if output_dir == protected or output_dir.is_relative_to(protected) or protected.is_relative_to(output_dir) or report_path.is_relative_to(protected):
            raise ValueError('Output must be separate from raw and baseline data')
    if report_path.is_relative_to(output_dir) or report_path.suffix != '.json':
        raise ValueError('Public aggregate report must be a separate JSON file')
    if policy['schema_version'] != 2 or any(policy[k] < 1 for k in ['language_min_words', 'language_segment_chars', 'context_depth', 'context_excerpt_chars', 'minimum_prose_words', 'minimum_prose_chars', 'commit_every', 'sample_per_stratum', 'sample_per_boundary']) or not .5 <= policy['language_confidence'] <= 1 or policy['reserve_gib'] < 0:
        raise ValueError('Invalid refinement policy')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Audit output already exists; preserve the earlier comparison')
    detector, parser = FastTextLanguage(model_path), TextSegments()
    with (baseline_dir/'.cleaning.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Baseline cleaner is active; retry after its transaction run')
        db = sqlite3.connect(f'file:{baseline_dir / "cleaned.sqlite"}?mode=ro', uri=True)
        try:
            db.execute('PRAGMA query_only=ON')
            if sha256_file(Path(__file__).with_name('reddit.py')) != base_spec['cleaner_sha256']:
                raise ValueError('Baseline implementation changed')
            if detector.fingerprint != base_spec['language']:
                raise ValueError('Comparison requires the pinned baseline language model')
            memberships, populations, sample_counts = sample_population(db, policy)
            chosen = {}
            for fullname in sorted(memberships):
                chosen[fullname] = json.loads(db.execute('SELECT record_json FROM records WHERE fullname=?', (fullname,)).fetchone()[0])
            selected_lines = defaultdict(dict)
            for fullname, r in chosen.items():
                selected_lines[r['provenance']['input_key']][r['provenance']['line']] = fullname
            sources, raw_rows = [], {}
            for key, lines in selected_lines.items():
                path, expected_hash, expected_rows = db.execute('SELECT path,sha256,expected_rows FROM inputs WHERE key=? AND done=1', (key,)).fetchone()
                path = Path(path).resolve()
                if not path.is_relative_to(production):
                    raise ValueError('Baseline export path is outside production')
                hasher = hashlib.sha256(); total = 0
                with path.open('rb') as stream:
                    for total, raw_line in enumerate(stream, 1):
                        hasher.update(raw_line)
                        if total in lines:
                            fullname = lines[total]
                            if hashlib.sha256(raw_line).hexdigest() != chosen[fullname]['provenance']['raw_line_sha256']:
                                raise ValueError('Sample provenance differs from raw export')
                            raw_rows[fullname] = json.loads(raw_line)
                if hasher.hexdigest() != expected_hash or total != expected_rows:
                    raise ValueError('Source hash/count mismatch')
                sources.append(dict(key=key, sha256=expected_hash, rows=expected_rows))
            if set(raw_rows) != set(chosen):
                raise ValueError('Missing sampled source rows')
            if shutil.disk_usage(output_dir.parent).free < policy['reserve_gib']*2**30:
                raise ValueError('Disk reserve reached')
            output_dir.mkdir(parents=True)
            fingerprint = dict(schema_version=2, candidate_sha256=sha256_file(Path(__file__)),
                baseline_spec_sha256=sha256_file(baseline_dir/'run-spec.json'),
                baseline_directory=str(baseline_dir), sources=sources, policy=policy,
                parser=parser.fingerprint, language=detector.fingerprint)
            write_json(output_dir/'run-spec.json', fingerprint)
            cache = {}

            def lookup(ident):
                if ident not in cache:
                    row = db.execute('SELECT record_json FROM records WHERE fullname=? AND input_key IN (SELECT key FROM inputs WHERE done=1)', (ident,)).fetchone()
                    cache[ident] = json.loads(row[0]) if row else None
                return cache[ident]

            routes, transitions, flag_counts, contexts, language_statuses, reasons = (Counter() for _ in range(6))
            by_stratum = defaultdict(Counter)
            with (output_dir/'refined.jsonl').open('w') as out, (output_dir/'annotation-template.jsonl').open('w') as ann:
                for fullname, baseline in chosen.items():
                    refined = refine(raw_rows[fullname], baseline, detector, parser, lookup, policy)
                    refined['sampling_strata'] = sorted(memberships[fullname])
                    refined['baseline_text'] = baseline['text']
                    refined['same_text_record_count'] = db.execute('SELECT count(*) FROM records WHERE text_hash=?', (baseline['text_sha256'],)).fetchone()[0]
                    out.write(canonical(refined)+'\n')
                    # Keep labels independent of candidate routes; split fixed before review.
                    split = 'held_out' if int(digest('split\0'+fullname), 16) % 5 == 0 else 'development'
                    ann.write(canonical(dict(fullname=fullname, split=split, label_status='unlabeled',
                        annotator=None, language=None, quotation_separation_correct=None,
                        automation=None, context_sufficient=None, personal_experience=None,
                        correct_route=None, notes=None))+'\n')
                    routes[refined['decision']] += 1
                    reasons[refined['reason']] += 1
                    transitions[baseline['decision']+' -> '+refined['decision']] += 1
                    flag_counts.update(refined['flags']); contexts[refined['context']['status']] += 1
                    language_statuses[refined['language_evidence']['status']] += 1
                    for key in memberships[fullname]:
                        by_stratum[key][refined['decision']] += 1
            report = dict(schema_version=2, status='strict_binary_sample_audit',
                generated_at=datetime.now(timezone.utc).isoformat(),
                complete_input_records=sum(n for k,n in populations.items() if k.startswith('stratum/')),
                complete_input_files=len(sources), communities=sorted({r['subreddit'] for r in chosen.values()}),
                sample_records=len(chosen), population_by_stratum=populations,
                sample_by_stratum=sample_counts, dispositions=dict(routes), reasons=dict(reasons), transitions=dict(transitions),
                flags=dict(flag_counts), contexts=dict(contexts), language_statuses=dict(language_statuses),
                dispositions_by_stratum={k: dict(v) for k,v in sorted(by_stratum.items())},
                private_output=str(output_dir), fingerprint_sha256=digest(canonical(fingerprint)),
                validation=dict(independent_labels=0, precision=None, recall=None,
                    interpretation='Stratified boundary-enriched comparison; unweighted percentages are not population estimates. No independent accuracy estimate.'),
                limitations=['Only completed primary sources provide context; context-only raw parents are not yet joined.',
                    'CommonMark syntax is an approximation to Reddit formatting; inline quotation and copied articles are not fully identified.',
                    'Language scores are uncalibrated; short, uncertain or mixed text is excluded from the analysis view.',
                    'Same-text IDs are retained; no author-level deduplication or patient counts are inferred.',
                    'No brand attribution, FDA-per-brand filtering, clinical relation extraction or personal-use eligibility is decided.'])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(report_path, report)
            return {k: report[k] for k in ['status','complete_input_records','sample_records','dispositions','reasons','transitions','contexts','language_statuses']}
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-file', type=Path, required=True)
    parser.add_argument('--language-model', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=Path('configs/reddit-refinement.json'))
    args = parser.parse_args()
    print(json.dumps(audit(args.baseline_dir,args.output_dir,args.report_file,args.language_model,
                           json.loads(args.policy.read_text())), indent=2))


if __name__ == '__main__':
    main()
