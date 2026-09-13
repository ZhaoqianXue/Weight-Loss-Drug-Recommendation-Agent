"""Collector failure cases and published snapshot invariants; no live requests."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from drug_catalog import DRUGS, ALIASES, ROOT
from code_scraping.scraper import parse_page, collect
from code_pipeline.refresh_data import key, annotation, index, read
from dataset_provenance import verify_index, index_metadata

DRUG=next(d for d in DRUGS if d['brand']=='Trulicity')

def page(records, cards, total=None, title='Trulicity (Dulaglutide)'):
    state={'all_reviews':{'drug_name':title,'total_review':len(records) if total is None else total,'drug_review_nimvs':[{'review_nimvs':records}]}}
    return '<html><script>window.__INITIAL_STATE__='+json.dumps(state)+'</script>'+cards+'</html>'

def record(id='1',author='Tester'):
    return {'userReviewId':id,'DisplayName':author,'DatePosted':'1/2/2026 1:00:00 PM','OverAll_UserReviewRating':5,'RatingCriteria1':'5','RatingCriteria2':'5','RatingCriteria3':'5','UserExperience':'Full review — café.','Gender':'Transgender','Usertype':'Caregiver'}

def card(author='Tester', rating=5):
    return '<div class="review-details-holder"><div class="details">'+author+' | 55-64 | Male | On supplement for 1 to 6 months | Patient</div><div class="date">1/2/2026</div><div class="overall-rating"><div role="slider" aria-valuenow="'+str(rating)+'"></div></div></div>'

class CollectorTests(unittest.TestCase):
    def test_visible_demographics_and_unicode(self):
        _, rows=parse_page(page([record()],card()),DRUG,1,'now')
        self.assertEqual(rows[0]['Gender'],'Male');self.assertEqual(rows[0]['Patient Type'],'Patient')
        self.assertEqual(rows[0]['Medication Duration'],'1 to 6 months')
        self.assertEqual(rows[0]['Textual Review'],'Full review — café.')
    def test_embedded_only_record_does_not_shift_next_card(self):
        _, rows=parse_page(page([record('1','NotRendered'),record('2')],card()),DRUG,1,'now')
        self.assertEqual(rows[0]['Source Visibility'],'embedded_only');self.assertEqual(rows[0]['Gender'],'')
        self.assertEqual(rows[1]['Source Visibility'],'rendered');self.assertEqual(rows[1]['Gender'],'Male')
    def test_oldest_age_group(self):
        _,rows=parse_page(page([record()],card().replace('55-64','75 or over')),DRUG,1,'now')
        self.assertEqual(rows[0]['Age'],'75 or over')
    def test_email_mask_is_preserved(self):
        _,rows=parse_page(page([record(author='example@example.test')],card('[email\xa0protected]')),DRUG,1,'now')
        self.assertEqual(rows[0]['User'],'[email protected]')
    def test_wrong_redirect_rejected(self):
        with self.assertRaises(ValueError): parse_page(page([record()],card(),title='Wrong Drug'),DRUG,1,'now')
    def test_rating_mismatch_rejected(self):
        with self.assertRaises(ValueError): parse_page(page([record()],card(rating=1)),DRUG,1,'now')
    def test_missing_records_rejected(self):
        with self.assertRaises(ValueError): parse_page(page([],''),DRUG,1,'now')
    def test_repeated_pagination_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _,rows=parse_page(page([record()],card()),DRUG,1,'now')
            for p in (1,2): (Path(temp)/('Trulicity-%03d.json'%p)).write_text(json.dumps({'total':2,'rows':rows}))
            with self.assertRaisesRegex(ValueError,'Duplicate'): collect(DRUG,temp,0)
    def test_failure_does_not_publish(self):
        from code_scraping import batch_scraper
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'webmd_all_reviews.csv';output.write_text('previous snapshot')
            with patch('sys.argv',['batch_scraper','--output-dir',temp]), patch.object(batch_scraper,'collect',side_effect=ValueError('failed page')):
                with self.assertRaises(ValueError): batch_scraper.main()
            self.assertEqual(output.read_text(),'previous snapshot')

class SnapshotTests(unittest.TestCase):
    def test_catalog_and_counts(self):
        rows=read(ROOT/'data_webmd/webmd_all_reviews.csv');manifest=json.loads((ROOT/'data_webmd/collection_manifest.json').read_text())
        self.assertEqual(len(rows),2727)
        self.assertEqual(len(rows),manifest['total_reviews'])
        self.assertEqual(len({r['Review ID'] for r in rows}),len(rows))
        self.assertEqual(len(DRUGS),8);self.assertEqual(len({d['generic'] for d in DRUGS}),4)
        for drug, summary in zip(DRUGS,manifest['brands']):
            group=[r for r in rows if r['Brand Name']==drug['brand']]
            self.assertEqual(len(group),summary['headline_count'])
            self.assertEqual(read(ROOT/('data_webmd/webmd_'+drug['brand'].lower()+'_reviews.csv')),group)
            self.assertTrue(all(r['Drug Name']==drug['generic'] and r['Source URL']==drug['url'] for r in group))
        self.assertEqual(ALIASES['wegovy hd'],'Wegovy');self.assertEqual(ALIASES['victoza 2-pak'],'Victoza')
    def test_all_current_outputs_share_ids_and_provenance(self):
        raw=read(ROOT/'data_webmd/webmd_all_reviews.csv');ids=[r['Review ID'] for r in raw]
        for name in ('data_extracted/extracted_reviews_all.csv','data_standardized/standardized_reviews_all.csv','code_website/static/standardized_reviews_all.csv','code_website/data/standardized_reviews_all.csv'):
            rows=read(ROOT/name);self.assertEqual([r['Review ID'] for r in rows],ids)
            self.assertTrue(all(r['Extraction Status'] in ('pending','failed','completed','historical_reused') for r in rows))
        canonical=(ROOT/'data_standardized/standardized_reviews_all.csv').read_bytes()
        for name in ('code_website/static/standardized_reviews_all.csv','code_website/data/standardized_reviews_all.csv'):
            self.assertEqual(canonical,(ROOT/name).read_bytes())
        manifest=json.loads((ROOT/'data_standardized/dataset_manifest.json').read_text())
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),manifest['standardized_csv_sha256'])
    def test_pending_has_no_invented_effects(self):
        for row in read(ROOT/'data_standardized/standardized_reviews_all.csv'):
            if row['Extraction Status']=='pending':
                self.assertEqual(json.loads(row['standardized_info'])['side_effects'],[])
                self.assertFalse(any(r['relation']=='causes' for r in json.loads(row['standardized_relations'])))
    def test_join_requires_same_content_author_date_and_unique_match(self):
        row={'Brand Name':'Wegovy','Date':'1/2/2026','User':'Tester','Textual Review':'long text','Condition':'Other','Medication Duration':''}
        info={'side_effects':[]};old={**row,'structured_info':json.dumps(info),'relations':'[]'}
        self.assertEqual(key(row),key({**row,'Textual Review':'longtext'}))
        self.assertNotEqual(key(row),key({**row,'Textual Review':'different text'}))
        self.assertEqual(annotation(row,index([old]),'structured_info','relations')[2],'historical_reused')
        self.assertEqual(annotation(row,index([old,old]),'structured_info','relations')[2],'pending')
    def test_foreign_drug_annotations_are_pending(self):
        row={'Brand Name':'Wegovy','Date':'1/2/2026','User':'Tester','Textual Review':'review','Condition':'Other','Medication Duration':''}
        old={**row,'structured_info':json.dumps({'side_effects':[{'name':'nausea','associated_drug':'Metformin'}]}),'relations':'[]'}
        self.assertEqual(annotation(row,index([old]),'structured_info','relations')[2],'pending')
    def test_stale_index_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):verify_index(temp,'test')
            path=Path(temp)/'dataset_manifest.json';path.write_text(json.dumps(index_metadata('test')))
            verify_index(temp,'test')
            with self.assertRaises(ValueError):verify_index(temp,'different-model')

if __name__=='__main__':unittest.main()
