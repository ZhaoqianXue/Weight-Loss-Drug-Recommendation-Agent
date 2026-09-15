"""Consistency checks against the frozen study snapshot."""
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


class SnapshotTests(unittest.TestCase):
    def test_catalog_and_counts(self):
        rows=read(ROOT/'data/raw/webmd/2026-09-12/webmd_all_reviews.csv');manifest=json.loads((ROOT/'data/raw/webmd/2026-09-12/collection_manifest.json').read_text())
        self.assertEqual(len(rows),2727)
        self.assertEqual(len(rows),manifest['total_reviews'])
        self.assertEqual(len({r['Review ID'] for r in rows}),len(rows))
        self.assertEqual(len(DRUGS),8);self.assertEqual(len({d['generic'] for d in DRUGS}),4)
        for drug, summary in zip(DRUGS,manifest['brands']):
            group=[r for r in rows if r['Brand Name']==drug['brand']]
            self.assertEqual(len(group),summary['headline_count'])
            self.assertEqual(read(ROOT/('data/raw/webmd/2026-09-12/webmd_'+drug['brand'].lower()+'_reviews.csv')),group)
            self.assertTrue(all(r['Drug Name']==drug['generic'] and r['Source URL']==drug['url'] for r in group))
        self.assertEqual(ALIASES['wegovy hd'],'Wegovy');self.assertEqual(ALIASES['victoza 2-pak'],'Victoza')

    def test_all_current_outputs_share_ids_and_provenance(self):
        raw=read(ROOT/'data/raw/webmd/2026-09-12/webmd_all_reviews.csv');ids=[r['Review ID'] for r in raw]
        for name in ('data/interim/migrated-2026-09-14/extracted_reviews_all.csv','data/processed/migrated-2026-09-14/standardized_reviews_all.csv','artifacts/web/static/standardized_reviews_all.csv','artifacts/web/static/standardized_reviews_all.csv'):
            rows=read(ROOT/name);self.assertEqual([r['Review ID'] for r in rows],ids)
            self.assertTrue(all(r['Extraction Status'] in ('pending','failed','completed','historical_reused') for r in rows))
        canonical=(ROOT/'data/processed/migrated-2026-09-14/standardized_reviews_all.csv').read_bytes()
        for name in ('artifacts/web/static/standardized_reviews_all.csv','artifacts/web/static/standardized_reviews_all.csv'):
            self.assertEqual(canonical,(ROOT/name).read_bytes())
        manifest=json.loads((ROOT/'data/processed/migrated-2026-09-14/dataset_manifest.json').read_text())
        self.assertEqual(hashlib.sha256(canonical).hexdigest(),manifest['standardized_csv_sha256'])

    def test_pending_has_no_invented_effects(self):
        for row in read(ROOT/'data/processed/migrated-2026-09-14/standardized_reviews_all.csv'):
            if row['Extraction Status']=='pending':
                self.assertEqual(json.loads(row['standardized_info'])['side_effects'],[])
                self.assertFalse(any(r['relation']=='causes' for r in json.loads(row['standardized_relations'])))

