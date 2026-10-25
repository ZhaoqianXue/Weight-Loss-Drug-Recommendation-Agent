"""Evidence-aware binary v3 final (open vocabulary for explicit long narratives). Reuse frozen v2 evidence without editing old runs.

Keep means relevant, interpretable medication discussion, not a verified patient
or adverse-event case. All decisions remain binary and uncertainty is excluded.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

from weightloss.cleaning.reddit import WORDS, URL
from weightloss.cleaning.reddit_refinement import BOT_TEMPLATE, PROMOTION, canonical
from weightloss.cleaning.reddit_strict import SCHEMA, insert_record, finalize_input
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json

PHATIC = re.compile(r'\b(?:thank\s+you|thanks|appreciate)\b[\s\S]*\b(?:community|kind\s+words|support|wonderful\s+people)\b', re.I)

PERSON = re.compile(r"\b(?:i|me|my|mine|we|our)\b", re.I)
SOLICITATION = re.compile(r'\b(?:join\b.{0,60}\b(?:facebook\s+group|discord\s+server|telegram\s+(?:group|channel))|'
                          r'(?:created|launched|started)\s+(?:a\s+)?(?:new\s+)?(?:facebook\s+|support\s+)?group|'
                          r'(?:take|complete|fill\s+out)\s+(?:my|our|this)\s+(?:survey|questionnaire))\b', re.I|re.S)
TREATMENT_REFERENCE = re.compile(r'\b(?:inject\w*|shots?|pens?|medication|medicine|prescrib\w*|prescription\w*|'
                                 r'(?:taking|take|on|off)\s+(?:it|this|these|the\s+medication))\b', re.I)
SIGNALS = {
    'symptom': r'\b(?:nause\w*|vomit\w*|diarrh\w*|constipat\w*|reflux|heartburn|burp\w*|dizz\w*|woozy|fatigue|exhaust\w*|headache\w*|pain\w*|cramp\w*|bloat\w*|rash\w*|itch\w*|bruisi\w*|sick|sickness|hypoglyc\w*|dehydrat\w*|insomnia|tremor\w*|shak\w*|anxiety|depress\w*|hair\s+loss|side\s+effects?)\b',
    'dose': r'\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?)\b|dos(?:e|es|age|ages|ing)\b|titrati\w*\b)',
    'duration': r'\b(?:\d+|one|two|three|four|five|six|seven|eight|first|second|third|last|next|few|several)\s*(?:days?|weeks?|months?|years?)\b',
    'outcome': r'\b(?:appetite|hungry|hunger|food\s+noise|cravings?|blood\s+sugar|glucose|a1c|hba1c|weight\s+(?:loss|gain|change)|(?:lost|gained|losing|gaining)\s+(?:(?:about|around|almost|over|another|only)\s+)?\d+|\d+(?:\.\d+)?\s*(?:lbs?|pounds?|kgs?|kilograms?)\b|plateau\w*)\b',
    'treatment': r'\b(?:inject\w*|prescrib\w*|prescription\w*|switched|switching|started|starting|stopped|stopping|restart\w*|taking|on\s+(?:this|it|the\s+medication)|off\s+(?:this|it|the\s+medication))\b',
    'access': r'\b(?:insurance|coverage|copay|co-pay|cost|costs|price|pricing|savings\s+card|coupon\w*|afford\w*|refill\w*|shortage\w*|pharmac\w*|out\s+of\s+pocket|prior\s+auth\w*|medicare)\b',
}
SIGNALS = {k: re.compile(v, re.I) for k,v in SIGNALS.items()}


def information(text):
    return sorted(name for name, pattern in SIGNALS.items() if pattern.search(text))


def mentions(text, policy):
    result = []
    for generic, brands in policy['medications'].items():
        for term in [generic]+brands:
            if re.search(r'\b'+re.escape(term)+r'\b', text, re.I):
                result.append(dict(term=term, ingredient=generic, brand=None if term==generic else term))
    return result


def english_evidence(feature, policy):
    e = feature['language_evidence']; threshold=policy['language_confidence']
    # A confident foreign-language span still vetoes, even if the whole is English.
    foreign = [p for p in e['segments'] if p['language'] not in (None,'en') and p['score'] >= threshold]
    if foreign:
        return dict(passed=False, basis='confident_non_english_span', conflict_count=len(foreign))
    whole=e.get('whole')
    if whole and whole['language']!='en' and whole['score']>=threshold:
        return dict(passed=False,basis='confident_non_english_document')
    if whole and whole['language']=='en' and whole['score'] >= threshold:
        return dict(passed=True, basis='document_english', score=whole['score'])
    # Long documents / weak short titles: substantive body blocks can establish
    # language. Short headings/signoffs do not have to pass an independent test.
    prose=[s for s in feature['segments'] if s['role']=='prose']
    body=[p for p in e['segments'] if p['words']>=policy['substantive_segment_words']
          and prose[p['segment']]['field']!='title']
    if body and all(p['language']=='en' and p['score']>=threshold for p in body):
        return dict(passed=True, basis='substantive_body_english', blocks=len(body))
    return dict(passed=False, basis='language_not_established')


def relevant_context(feature, lookup, policy):
    """Resolve discussion topic only. Never assign another speaker's exposure."""
    ctx=feature['context']
    if feature['kind']=='posts' or ctx['status']!='available' or not ctx['ancestors'] or not ctx['thread']:
        return dict(passed=False, reason='no_reliable_target_context')
    if set(ctx['issues'])-{'depth_limit'}:
        return dict(passed=False, reason='invalid_target_context')
    nodes = list(ctx['ancestors'])+[ctx['thread']]
    found, evidence, seen = [], [], set()
    for node in nodes:
        ident=node['fullname']
        if ident in seen: continue
        seen.add(ident); parent=lookup(ident)
        if not parent:
            return dict(passed=False, reason='missing_target_context')
        prose=URL.sub(' ',parent['prose_text'])
        if not prose.strip() or BOT_TEMPLATE.search(prose) or PROMOTION.search(prose) or SOLICITATION.search(prose):
            return dict(passed=False, reason='unreliable_target_context')
        # Do not resolve a reply about a competing medicine to the sole target
        # medicine mentioned elsewhere. This list is conservative, not exhaustive.
        if any(re.search(r'\b'+re.escape(t)+r'\b',prose,re.I) for t in policy['context_competing_terms']):
            return dict(passed=False, reason='competing_medication_context')
        names=mentions(prose,policy); found.extend(names)
        if names:
            evidence.append(dict(fullname=ident,mentions=names,provenance=parent['provenance']))
    ingredients={m['ingredient'] for m in found};brands={m['brand'] for m in found if m['brand']}
    if len(ingredients)!=1 or len(brands)>1:
        return dict(passed=False, reason='ambiguous_or_missing_target_context')
    return dict(passed=True,reason='unambiguous_thread_topic',ingredient=next(iter(ingredients)),
                topic_brands=sorted(brands),evidence=evidence,
                attribution='topic_relevance_only_not_current_speaker_exposure')


def decide(feature, lookup, policy):
    text=URL.sub(' ', feature['prose_text']).strip()
    words=len(WORDS.findall(text)); signals=information(text)
    own=mentions(text,policy); person=bool(PERSON.search(text))
    language=english_evidence(feature,policy)
    relevance=dict(passed=bool(own),reason='explicit_target' if own else 'no_explicit_target',mentions=own)
    # Full records need concrete information, not length alone. Short records
    # need two dimensions or a first-person symptom/outcome statement.
    # Do not impose a closed symptom vocabulary on already explicit, developed
    # medication narratives: rare or unlisted effects must remain discoverable.
    explicit_long = bool(own) and words>=policy['short_text_words'] and not (not signals and PHATIC.search(text))
    informative = explicit_long or (bool(signals) and words>=policy['minimum_interpretable_words'] and (
        words>=policy['short_text_words'] or len(signals)>=2 or
        person and bool({'symptom','outcome'} & set(signals))))
    flags=set(feature['flags'])
    personal_medical = bool(set(signals)&{'symptom','outcome','dose'}) or bool(TREATMENT_REFERENCE.search(text))
    if not own and informative and language['passed'] and person and personal_medical:
        if any(re.search(r'\b'+re.escape(t)+r'\b',text,re.I) for t in policy['context_competing_terms']):
            relevance=dict(passed=False,reason='competing_medication_in_reply')
        else:
            relevance=relevant_context(feature,lookup,policy)
    if not text or not any(c.isalnum() for c in text): reason='no_usable_prose'
    elif 'unquoted_automation_disclosure' in flags or BOT_TEMPLATE.search(text): reason='automation_disclosure'
    elif {'invalid_relationship_id','missing_relationship_id'} & flags: reason='invalid_structure'
    elif PROMOTION.search(text): reason='promotional_pattern'
    elif SOLICITATION.search(text): reason='solicitation'
    elif feature['crosspost_parent']: reason='crosspost'
    elif not informative: reason='insufficient_medication_information'
    elif not language['passed']: reason=language['basis']
    elif not relevance['passed']: reason=relevance['reason']
    else: reason='quality_eligible'
    # Keep already established evidence separate from final clinical extraction.
    return dict(fullname=feature['fullname'],subreddit=feature['subreddit'],kind=feature['kind'],
                provenance=feature['provenance'],decision='keep' if reason=='quality_eligible' else 'exclude',
                reason=reason,prose_text=feature['prose_text'] if reason=='quality_eligible' else None,
                information=dict(words=words,signals=signals,first_person_cue=person,passed=informative,
                                 basis='explicit_target_developed_prose' if explicit_long else 'information_dimensions'),
                language=language,relevance=relevance,context=feature['context'],flags=feature['flags'],
                v2_evidence_fullname=feature['fullname'],clinical_eligibility='not_assessed',
                independent_patient='not_established')


def source_digest(db, key):
    h=hashlib.sha256();n=0
    for row in db.execute('SELECT fullname,raw_hash,text_hash,created_utc,quality_decision,quality_reason,features_json FROM records WHERE input_key=? ORDER BY rowid',(key,)):
        h.update(canonical(row).encode());h.update(b'\n');n+=1
    return h.hexdigest(),n


def validate_policy(policy):
    if policy['schema_version']!=3 or not .5<=policy['language_confidence']<=1 or policy['reserve_gib']<0:
        raise ValueError('Invalid evidence policy')
    for key in ['substantive_segment_words','minimum_interpretable_words','short_text_words','commit_every']:
        if type(policy[key]) is not int or policy[key]<1: raise ValueError('Invalid positive policy field')
    if not policy['medications']: raise ValueError('Missing medication scope')


def run(source_dir, directory, report_path, policy, max_records=None):
    source_dir,directory,report_path=(Path(p).resolve() for p in (source_dir,directory,report_path))
    validate_policy(policy)
    if max_records is not None and max_records<1: raise ValueError('max_records must be positive')
    spec=json.loads((source_dir/'run-spec.json').read_text())
    baseline=Path(spec['baseline_directory'])
    base_spec=json.loads((baseline/'run-spec.json').read_text())
    production=Path(base_spec['production_directory']).resolve()
    for protected in [source_dir,baseline,production]:
        if directory==protected or directory.is_relative_to(protected) or protected.is_relative_to(directory) or report_path.is_relative_to(protected):
            raise ValueError('Output must be outside all source directories')
    if report_path.suffix!='.json' or report_path.is_relative_to(directory): raise ValueError('Invalid aggregate report path')
    module_dir=Path(__file__).parent
    if sha256_file(baseline/'run-spec.json')!=spec['baseline_spec_sha256']:
        raise ValueError('Baseline provenance changed')
    for name,key in [('reddit_strict.py','strict_sha256'),('reddit_refinement.py','refinement_sha256')]:
        if sha256_file(module_dir/name)!=spec[key]: raise ValueError('Pinned v2 implementation changed')
    if json.loads((production/'production-spec.json').read_text())!=base_spec['production_scope']:
        raise ValueError('Raw source scope changed')
    fingerprint=dict(schema_version=3,source_directory=str(source_dir),source_spec_sha256=sha256_file(source_dir/'run-spec.json'),
                     code_sha256=sha256_file(Path(__file__)),policy=policy)
    if directory.exists() and any(directory.iterdir()) and not (directory/'run-spec.json').exists():
        raise ValueError('Nonempty unowned output directory')
    directory.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for folder,mode in [(directory,fcntl.LOCK_EX),(source_dir,fcntl.LOCK_SH)]:
            lock=stack.enter_context((folder/'.cleaning.lock').open('a'))
            try: fcntl.flock(lock,mode|fcntl.LOCK_NB)
            except BlockingIOError: raise ValueError('Another cleaner is active')
        fp=directory/'run-spec.json'
        if fp.exists() and json.loads(fp.read_text())!=fingerprint: raise ValueError('Rules/source/code changed; create a new run')
        if not fp.exists(): write_json(fp,fingerprint)
        source=sqlite3.connect(f'file:{source_dir / "cleaned.sqlite"}?mode=ro',uri=True)
        source.execute('PRAGMA query_only=ON');stack.callback(source.close)
        db=sqlite3.connect(directory/'cleaned.sqlite');stack.callback(db.close)
        db.executescript(SCHEMA+'''
            CREATE TABLE IF NOT EXISTS comparisons(fullname TEXT PRIMARY KEY,v2_decision TEXT,v2_reason TEXT,
                old_short INTEGER,old_no_name INTEGER,old_language_veto INTEGER);
        ''')
        inputs=source.execute('SELECT key,path,sha256,expected_rows FROM inputs WHERE done=1 ORDER BY key').fetchall()
        if not {r[0] for r in db.execute('SELECT key FROM inputs')} <= {r[0] for r in inputs}:
            raise ValueError('A completed source input disappeared')
        @lru_cache(maxsize=8192)
        def lookup(fullname):
            row=source.execute('SELECT features_json FROM records WHERE fullname=? AND input_key IN (SELECT key FROM inputs WHERE done=1)',(fullname,)).fetchone()
            return json.loads(row[0]) if row else None
        processed=0
        for key,path,raw_sha,raw_count in inputs:
            evidence_sha,expected=source_digest(source,key)
            previous=db.execute('SELECT sha256,expected_rows,offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
            if previous and previous[:2]!=(evidence_sha,expected): raise ValueError('Frozen source evidence changed')
            with db:
                db.execute('INSERT OR IGNORE INTO inputs(key,path,sha256,expected_rows) VALUES (?,?,?,?)',(key,str(source_dir/'cleaned.sqlite'),evidence_sha,expected))
            offset,lines,done=db.execute('SELECT offset,lines,done FROM inputs WHERE key=?',(key,)).fetchone()
            if done: continue
            while max_records is None or processed<max_records:
                if shutil.disk_usage(directory).free<policy['reserve_gib']*2**30: raise ValueError('Disk reserve reached; checkpoint retained')
                limit=policy['commit_every'] if max_records is None else min(policy['commit_every'],max_records-processed)
                rows=source.execute('''SELECT r.rowid,r.fullname,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,
                    r.quality_decision,r.quality_reason,r.features_json,c.fullname
                    FROM records r LEFT JOIN canonical_text c ON r.text_hash=c.text_hash
                    WHERE r.input_key=? AND r.rowid>? ORDER BY r.rowid LIMIT ?''',(key,offset,limit)).fetchall()
                if not rows:
                    if lines!=expected: raise ValueError('Evidence count mismatch')
                    with db: finalize_input(db,key)
                    break
                with db:
                    for rid,fullname,raw_hash,text_hash,stamp,sub,kind,old_dec,old_reason,payload,canonical_id in rows:
                        feature=json.loads(payload); result=decide(feature,lookup,policy)
                        base=dict(fullname=fullname,provenance=feature['provenance'],text_sha256=text_hash,
                                  created_utc=stamp,subreddit=sub,kind=kind)
                        insert_record(db,base,result,raw_hash)
                        if old_dec=='keep' and canonical_id!=fullname: old_dec,old_reason='exclude','duplicate_text'
                        db.execute('INSERT INTO comparisons VALUES (?,?,?,?,?,?)',(fullname,old_dec,old_reason,
                            feature['language_evidence']['words']<20 or len(URL.sub(' ',feature['prose_text']).strip())<80,
                            not feature['target_mentions'],old_reason in ['uncertain_language_segment','uncertain_language']))
                        offset=rid;lines+=1;processed+=1
                    db.execute('UPDATE inputs SET offset=?,lines=? WHERE key=?',(offset,lines,key))
                if lines%10000<limit: print(json.dumps(dict(input=key,rows=lines,expected=expected)),flush=True)
            if max_records is not None and processed>=max_records: break
        counts=dict(db.execute('SELECT decision,count(*) FROM cleaned_records GROUP BY decision'))
        reasons=dict(db.execute('SELECT reason,count(*) FROM cleaned_records GROUP BY reason'))
        source_inputs=[dict(zip(['key','processed','expected','done'],r)) for r in db.execute('SELECT key,lines,expected_rows,done FROM inputs ORDER BY key')]
        complete=len(source_inputs)==len(inputs) and all(r['done'] for r in source_inputs)
        transitions=[dict(zip(['v2','v3','records'],r)) for r in db.execute('SELECT c.v2_decision,r.decision,count(*) FROM comparisons c JOIN cleaned_records r USING(fullname) GROUP BY c.v2_decision,r.decision')]
        restored_gates=dict(zip(['old_short','old_no_name','old_language_veto'],db.execute("SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM comparisons c JOIN cleaned_records r USING(fullname) WHERE c.v2_decision='exclude' AND r.decision='keep'").fetchone()))
        communities=[dict(zip(['subreddit','decision','records'],r)) for r in db.execute('SELECT subreddit,decision,count(*) FROM cleaned_records GROUP BY subreddit,decision')]
        report=dict(schema_version=3,status='all_current_inputs_cleaned' if complete else 'partial',
                    generated_at=datetime.now(timezone.utc).isoformat(),source_input_files=len(inputs),inputs=source_inputs,
                    full16_sources_cleaned=complete and len(inputs)==32,completed_records=sum(counts.values()),
                    processed_records=db.execute('SELECT count(*) FROM records').fetchone()[0],
                    processed_this_invocation=processed,dispositions=counts,reasons=reasons,transitions=transitions,
                    newly_retained_old_gate_hits_overlapping=restored_gates,by_community=communities,
                    independent_precision=None,clinical_eligibility='not_assessed',policy=fingerprint)
        report_path.parent.mkdir(parents=True,exist_ok=True);write_json(report_path,report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report-file',type=Path,required=True);p.add_argument('--max-records',type=int)
    p.add_argument('--policy',type=Path,default=Path('configs/reddit-evidence-final.json'))
    a=p.parse_args();r=run(a.source_dir,a.output_dir,a.report_file,json.loads(a.policy.read_text()),a.max_records)
    print(json.dumps({k:r[k] for k in ['status','completed_records','processed_this_invocation','dispositions','transitions']},indent=2))


if __name__=='__main__': main()
