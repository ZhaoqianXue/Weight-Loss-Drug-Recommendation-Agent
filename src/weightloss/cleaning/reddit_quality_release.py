"""Final binary publication checks over v3; preserve underlying decisions.

Run the resumable evidence stage, then apply pinned low-quality-content vetoes.
An override is an exclusion, never a review class. Underlying records stay intact.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sqlite3

from weightloss.cleaning.reddit_evidence_final import run, SIGNALS, TREATMENT_REFERENCE
from weightloss.ingestion.reddit import sha256_file
from weightloss.ingestion.reddit_production import write_json

BIO_PROMOTION=re.compile(r'\b(?:link(?:ed|s)?\s+(?:is\s+)?in\s+(?:my|our|the)\s+(?:bio|profile)|'
                         r'(?:check|see|visit)\s+(?:out\s+)?(?:my|our)\s+(?:bio|profile)\b.{0,60}\b(?:link|code|discount|coupon)|'
                         r'(?:discount|referral|affiliate|coupon)\s+(?:code|link)\b.{0,60}\b(?:my|our)\s+(?:bio|profile))',re.I|re.S)


def veto(feature):
    text=feature.get('prose_text') or ''
    if BIO_PROMOTION.search(text):return 'profile_link_promotion'
    if feature.get('relevance',{}).get('reason')=='unambiguous_thread_topic':
        signals=set(feature['information']['signals'])
        symptom_terms={m.group().casefold().strip() for m in SIGNALS['symptom'].finditer(text)}
        if signals <= {'symptom','duration'} and symptom_terms and symptom_terms <= {'side effect','side effects'} and not TREATMENT_REFERENCE.search(text):
            return 'vague_context_reply'
    return None


def release(directory, report_path):
    directory,report_path=Path(directory).resolve(),Path(report_path).resolve()
    if not (directory/'run-spec.json').exists() or report_path.is_relative_to(directory) or report_path.suffix!='.json':
        raise ValueError('Expected an existing derived run and separate aggregate report')
    spec=json.loads((directory/'run-spec.json').read_text())
    for protected in [Path(spec['source_directory']), Path(json.loads((Path(spec['source_directory'])/'run-spec.json').read_text())['baseline_directory'])]:
        if report_path.is_relative_to(protected):raise ValueError('Protected report path')
    fingerprint=dict(schema_version=1,code_sha256=sha256_file(Path(__file__)),base_spec_sha256=sha256_file(directory/'run-spec.json'))
    with (directory/'.cleaning.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Evidence cleaner is still active')
        fp=directory/'quality-screen-spec.json'
        if fp.exists() and json.loads(fp.read_text())!=fingerprint:raise ValueError('Publication rules changed')
        if not fp.exists():write_json(fp,fingerprint)
        db=sqlite3.connect(directory/'cleaned.sqlite')
        try:
            db.executescript('CREATE TABLE IF NOT EXISTS quality_exclusions(fullname TEXT PRIMARY KEY,reason TEXT);')
            # Idempotent: only newly arrived keep candidates need screening.
            old=db.execute('SELECT count(*) FROM quality_exclusions').fetchone()[0]
            new_rows=[]
            for ident,payload in db.execute("SELECT r.fullname,r.features_json FROM records r JOIN inputs i ON r.input_key=i.key AND i.done=1 WHERE r.quality_decision='keep' AND r.fullname NOT IN (SELECT fullname FROM quality_exclusions)"):
                reason=veto(json.loads(payload))
                if reason:new_rows.append((ident,reason))
            with db:
                db.executemany('INSERT INTO quality_exclusions VALUES (?,?)',new_rows)
                # One transaction publishes the override-aware partition.
                db.execute('DROP VIEW IF EXISTS kept_records');db.execute('DROP VIEW IF EXISTS excluded_records');db.execute('DROP VIEW IF EXISTS cleaned_records')
                db.execute('''CREATE VIEW cleaned_records AS
                    SELECT r.fullname,r.input_key,r.raw_hash,r.text_hash,r.created_utc,r.subreddit,r.kind,
                    CASE WHEN q.fullname IS NOT NULL THEN 'exclude' WHEN r.quality_decision='keep' AND c.fullname=r.fullname THEN 'keep' ELSE 'exclude' END AS decision,
                    CASE WHEN q.fullname IS NOT NULL THEN q.reason WHEN r.quality_decision='keep' AND c.fullname!=r.fullname THEN 'duplicate_text' ELSE r.quality_reason END AS reason,
                    c.fullname AS canonical_fullname,r.features_json
                    FROM records r JOIN inputs i ON r.input_key=i.key AND i.done=1
                    LEFT JOIN canonical_text c ON r.text_hash=c.text_hash LEFT JOIN quality_exclusions q ON r.fullname=q.fullname''')
                db.execute("CREATE VIEW kept_records AS SELECT * FROM cleaned_records WHERE decision='keep'")
                db.execute("CREATE VIEW excluded_records AS SELECT * FROM cleaned_records WHERE decision='exclude'")
            report=json.loads(report_path.read_text())
            report['dispositions']=dict(db.execute('SELECT decision,count(*) FROM cleaned_records GROUP BY decision'))
            report['reasons']=dict(db.execute('SELECT reason,count(*) FROM cleaned_records GROUP BY reason'))
            report['completed_records']=sum(report['dispositions'].values())
            report['transitions']=[dict(zip(['v2','v3','records'],r)) for r in db.execute('SELECT c.v2_decision,r.decision,count(*) FROM comparisons c JOIN cleaned_records r USING(fullname) GROUP BY c.v2_decision,r.decision')]
            report['by_community']=[dict(zip(['subreddit','decision','records'],r)) for r in db.execute('SELECT subreddit,decision,count(*) FROM cleaned_records GROUP BY subreddit,decision')]
            report['newly_retained_old_gate_hits_overlapping']=dict(zip(['old_short','old_no_name','old_language_veto'],db.execute("SELECT coalesce(sum(old_short),0),coalesce(sum(old_no_name),0),coalesce(sum(old_language_veto),0) FROM comparisons c JOIN cleaned_records r USING(fullname) WHERE c.v2_decision='exclude' AND r.decision='keep'").fetchone()))
            report['quality_screen']=dict(status='complete_for_current_completed_inputs',previous_exclusions=old,
                added_exclusions=len(new_rows),total_exclusions=old+len(new_rows),fingerprint=fingerprint)
            report['generated_at']=datetime.now(timezone.utc).isoformat();report['quality_ready']=True
            write_json(report_path,report)
            return report
        finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=Path('.cache/reddit-cleaning/v2-strict'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--report-file',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=Path('configs/reddit-evidence-final.json'))
    p.add_argument('--release-only',action='store_true')
    a=p.parse_args()
    if not a.release_only:run(a.source_dir,a.output_dir,a.report_file,json.loads(a.policy.read_text()))
    r=release(a.output_dir,a.report_file)
    print(json.dumps({k:r[k] for k in ['status','completed_records','dispositions','transitions','quality_screen']},indent=2))


if __name__=='__main__':main()
