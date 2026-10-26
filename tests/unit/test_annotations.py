"""Annotation reuse and index provenance contracts."""
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


class AnnotationTests(unittest.TestCase):
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

