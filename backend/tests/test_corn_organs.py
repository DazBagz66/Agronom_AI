"""Organ routing: holdout integrity and abstention at inference."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.diagnostics.corn_organ import detect_corn_organ, load_organ_model


class OrganModelTests(unittest.TestCase):
    def test_installed_model_matches_approved_checksums(self):
        _,head,manifest = load_organ_model()
        self.assertEqual(manifest['release_status'],'approved')
        self.assertEqual(manifest['architecture'], 'mobilenet_v3_large')
        self.assertEqual(head['coef'].shape,(4,960))
        self.assertTrue(np.isfinite(head['coef']).all())
        self.assertTrue((head['scale']>0).all())

    def test_low_confidence_and_other_abstain(self):
        head = {'mean':np.zeros(1),'scale':np.ones(1),'coef':np.zeros((4,1)),
                'intercept':np.zeros(4),'temperature':1.}
        manifest = {'classes':['ear','leaf','other','whole'],'threshold':.8,'sha256':'a'*64}
        with patch('app.diagnostics.corn_organ.load_organ_model',return_value=(None,head,manifest)), patch('app.diagnostics.corn_organ.extract_features',return_value=np.zeros(1)):
            self.assertEqual(detect_corn_organ(None)['part'],'unknown')
            for index,expected in [(0,'ear'),(1,'leaf'),(2,'unknown'),(3,'whole')]:
                head['intercept'] = np.eye(4)[index]*10
                self.assertEqual(detect_corn_organ(None)['part'],expected)

    def test_organ_holdout_has_no_shared_images_or_groups(self):
        path = ROOT/'backend/data/datasets/corn_organs/dataset.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for field in ['sha256','organ_group','partition_group','duplicate_group']:
            split_sets = {split:{r[field] for r in rows if r['split']==split} for split in ['train','validation','test']}
            self.assertFalse(split_sets['train'] & split_sets['validation'],field)
            self.assertFalse(split_sets['train'] & split_sets['test'],field)
            self.assertFalse(split_sets['validation'] & split_sets['test'],field)
        self.assertTrue(all(r['split']=='test' for r in rows if r['source']=='idadp'))
        self.assertTrue(all('feedback' not in r['source'] for r in rows))
