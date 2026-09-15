"""Publish current reviews with exact-match historical annotations and explicit coverage.

Does not fabricate annotations or make model calls. Run incremental extraction and
standardization separately when their credentials and dependencies are available.
"""
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from weightloss.catalog import load_catalog
from weightloss.settings import get_settings

def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def date(value):
    for fmt in ('%m/%d/%Y','%Y-%m-%d'):
        try: return datetime.strptime(value,fmt).strftime('%Y-%m-%d')
        except ValueError: pass
    raise ValueError('Unrecognized date: ' + value)

def key(row):
    norm = lambda v: re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(v))).strip()
    # Author/date alone are insufficient. Require matching full review characters too; the old scraper joined
    # expanded text spans without spaces. Ignore whitespace only, never words.
    return row['Brand Name'],date(row['Date']),norm(row['User']) or 'Anonymous',re.sub(r'\s+','',norm(row['Textual Review']))

def index(rows):
    result = defaultdict(list)
    for row in rows: result[key(row)].append(row)
    return result

def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)

def metadata(row):
    return {'drug':{'name':row['Brand Name'],'dosage':None,'dosage_form':None,'duration':row['Medication Duration'] or None,'continued_use':None,'alternative_drug_considered':None},'condition':{'name':row['Condition'] or None,'severity':None},'side_effects':[]}

def annotation(row, candidates, info_field, relation_field):
    matches = candidates.get(key(row),[])
    # Ambiguous matches are not reused, even if they have identical authors/text.
    if len(matches)==1:
        old=matches[0]
        try:
            info=json.loads(old[info_field]); relations=json.loads(old[relation_field])
            if not isinstance(info,dict) or not isinstance(info.get('side_effects'),list) or not isinstance(relations,list): raise ValueError('Malformed annotation')
            if any(e.get('associated_drug') and e['associated_drug'] != row['Brand Name'] for e in info['side_effects']):
                raise ValueError('Historical side effect attributed to another medication; re-extract')
            if any(r.get('start',{}).get('label') == 'Medication' and r['start']['properties'].get('name') != row['Brand Name'] for r in relations):
                raise ValueError('Historical relation belongs to another medication; re-extract')
            status=old.get('Extraction Status') or 'historical_reused'
            if status not in ('pending','failed'):
                return info,relations,status
        except (ValueError,TypeError,KeyError): pass
    # Unknown extraction is represented by status, never interpreted as absence.
    relations=[]
    if row['Condition']:
        relations=[{'start':{'label':'Medication','properties':{'name':row['Brand Name']}},'end':{'label':'Disease','properties':{'name':row['Condition']}},'relation':'reviewed_for','properties':{'source':'WebMD condition field','approval':None,'off_label':None}}]
    return metadata(row),relations, 'pending'

def main():
    paths = get_settings()
    paths.require_writable()
    raw=read(paths.raw)
    manifest=json.loads((paths.collection_manifest).read_text())
    if hashlib.sha256((paths.raw).read_bytes()).hexdigest()!=manifest['csv_sha256']:
        raise ValueError('Raw CSV does not match validated collection manifest')
    backup=paths.path('annotation_seed_dir')
    extracted=index(read(backup/'extracted_reviews_all.csv'))
    standardized=index(read(backup/'standardized_reviews_all.csv'))
    # On subsequent runs prefer successful current annotations; keep historical fallback.
    for name, destination in [(paths.path('extracted'),extracted),(paths.path('standardized'),standardized)]:
        for k, group in index(read(name)).items():
            if len(group)==1 and group[0].get('Extraction Status') not in ('pending','failed'): destination[k]=group
    exrows, strows=[],[]
    for row in raw:
        for candidates, info_field, rel_field, target in [(extracted,'structured_info','relations',exrows),(standardized,'standardized_info','standardized_relations',strows)]:
            info, relations, status=annotation(row,candidates,info_field,rel_field)
            out=dict(row)
            out['Extraction Status']=status
            out['Annotation Method']='historical automated extraction; exact non-whitespace content + author/date match' if status=='historical_reused' else ('not processed' if status=='pending' else 'incremental LLM extraction')
            if info_field=='standardized_info':
                out['Date']=date(out['Date'])
                out['Standardization Status']='historical_reused' if status=='historical_reused' else ('pending' if status=='pending' else 'completed')
            out[info_field]=json.dumps(info,ensure_ascii=False)
            out[rel_field]=json.dumps(relations,ensure_ascii=False)
            target.append(out)
    write(paths.path('extracted'),exrows)
    write(paths.path('standardized'),strows)
    write_dataset_manifest(strows, manifest)

def write_dataset_manifest(rows=None, collection=None):
    paths = get_settings()
    source=paths.path('standardized')
    rows=rows if rows is not None else read(source)
    collection=collection or json.loads((paths.collection_manifest).read_text())
    coverage=[]
    for drug in load_catalog():
        group=[r for r in rows if r['Brand Name']==drug['brand']]
        coverage.append({'generic':drug['generic'],'brand':drug['brand'],'reviews':len(group),'text_reviews':sum(bool(r['Textual Review'].strip()) for r in group),'extraction_status':dict(Counter(r['Extraction Status'] for r in group))})
    report={'collection_completed_at':collection['completed_at'],'total_reviews':len(rows),'generics':len(set(d['generic'] for d in load_catalog())),'brands':coverage,'raw_csv_sha256':collection['csv_sha256'],'standardized_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limitations':['Pending extraction is unknown, not an absence of adverse effects.','Historical annotations are retained without clinical revalidation.','Wegovy pages may combine formulations; brand alone does not establish dosage form.','Historical FDA classifications are not current regulatory evidence.']}
    paths.dataset_manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


