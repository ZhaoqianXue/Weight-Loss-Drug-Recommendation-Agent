"""Synthetic collection parsing and failure checks."""
import csv

import hashlib

import json

import tempfile

import unittest

from pathlib import Path

from unittest.mock import patch

from weightloss.catalog import DRUGS, ALIASES, ROOT

from weightloss.ingestion.scraper import parse_page, collect

from weightloss.pipeline.refresh_data import key, annotation, index, read

from weightloss.provenance import verify_index, index_metadata

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
        from weightloss.ingestion import batch_scraper
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'webmd_all_reviews.csv';output.write_text('previous snapshot')
            with patch('sys.argv',['batch_scraper','--output-dir',temp]), patch.object(batch_scraper,'collect',side_effect=ValueError('failed page')):
                with self.assertRaises(ValueError): batch_scraper.main()
            self.assertEqual(output.read_text(),'previous snapshot')

