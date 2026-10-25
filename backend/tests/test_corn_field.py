"""Regression tests for field splits and human review trust boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_corn_field import read_reviews, split_rows, ANHUI_LABELS
from scripts.activate_model import activate
from app.crops import CROPS
from app.diagnostics import crop_model


def image_row(n,label='healthy',**extra):
    sha=hashlib.sha256(str(n).encode()).hexdigest()
    return {'path':f'{n}.jpg','sha256':sha,'pixel_sha256':sha,
        'phash':f'{(n*0x123456789abcdef)&((1<<63)-1):016x}','source':'new','labels':[label],
        'width':200,'height':200,'gray_std':30,'organ':'leaf','review_required':False,
        **extra}


class FieldSplitTests(unittest.TestCase):
    def test_heldout_duplicate_never_enters_training(self):
        a=image_row(1,protected=True,source='anhui')
        b=image_row(1,source='ecuador',fixed_split='train')
        rows=split_rows([a,b],{})
        self.assertTrue(all(r['split'] not in {'train','validation'} for r in rows))

    def test_site_group_does_not_create_false_label_conflict(self):
        a=image_row(1,split_group_hint='site:1',fixed_split='validation')
        b=image_row(300,'southern_leaf_blight',split_group_hint='site:1',fixed_split='train')
        rows=split_rows([a,b],{})
        self.assertEqual({r['split'] for r in rows},{'validation'})
        self.assertFalse(any(r['label_conflict'] for r in rows))
        self.assertEqual(rows[0]['partition_group'],rows[1]['partition_group'])

    def test_unreviewed_observation_is_quarantined(self):
        row=image_row(1,'common_smut',organ='unverified',review_required=True)
        self.assertEqual(split_rows([row],{})[0]['split'],'quarantine')

    def test_idadp_uncertain_ear_taxonomy_stays_out_of_training(self):
        from scripts.prepare_idadp_corn import MAPPING
        label, organ = MAPPING['穗腐病']
        row = image_row(1, label, organ=organ, source='idadp', review_required=True,
                        protected=True, fixed_split='test', split_group_hint='idadp:entire_source')
        actual = split_rows([row], {})[0]
        self.assertEqual(actual['split'], 'external_review')
        self.assertIn('taxonomy_or_organ_review_required', actual['exclusion_reasons'])
        self.assertEqual(MAPPING['健康'], ('healthy', 'leaf'))

    def test_accepting_review_cannot_release_test(self):
        row=image_row(1,'common_smut',protected=True,organ='unverified',review_required=True)
        reviews={row['sha256']:{'status':'accepted','labels':['common_smut'],'organ':'ear',
                               'reviewer':'A','evidence':'record'}}
        actual=split_rows([row],reviews)[0]
        self.assertEqual(actual['split'],'test')
        self.assertEqual(actual['organ'],'ear')

    def test_conflicting_duplicate_labels_excluded(self):
        rows=split_rows([image_row(1),image_row(1,'common_rust')],{})
        self.assertTrue(all(r['split']=='quarantine' for r in rows))

    def test_review_requires_identity_and_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'reviews.csv'
            path.write_text('sha256,labels,organ,reviewer,evidence,status\n'+'1'*64+',healthy,leaf,,,accepted\n')
            with self.assertRaisesRegex(ValueError,'reviewer'):read_reviews(path)

    def test_stale_review_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'absent'):
            split_rows([image_row(1)],{'f'*64:{'status':'rejected'}})

    def test_species_mapping_is_explicit(self):
        self.assertEqual(ANHUI_LABELS['Bipolaris_Maydis'],'southern_leaf_blight')
        self.assertNotIn('Own-Spot',ANHUI_LABELS)

    def test_duplicates_between_fixed_sites_are_quarantined(self):
        rows=split_rows([image_row(1,fixed_split='train',split_group_hint='site:1'),
                         image_row(1,fixed_split='validation',split_group_hint='site:2'),
                         image_row(300,'southern_leaf_blight',fixed_split='train',split_group_hint='site:1')],{})
        self.assertEqual(rows[0]['split'],'quarantine')
        self.assertEqual(rows[1]['split'],'quarantine')
        self.assertEqual(rows[2]['split'],'train')


class CandidateRuntimeTests(unittest.TestCase):
    def test_capabilities_hide_unreleased_classes_and_missing_ear(self):
        manifest = {'classes': ['common_rust', 'common_smut'], 'release_classes': ['common_rust']}
        with patch.object(crop_model, 'load_model', side_effect=[(None, manifest), ValueError('not installed')]):
            capabilities = crop_model.crop_capabilities('Кукуруза')
        self.assertEqual(capabilities['parts'], ['leaf'])
        self.assertEqual(set(capabilities['labels']), {'common_rust'})

    def test_sunflower_leaf_model_serves_leaf_and_flower(self):
        manifest = {'classes': ['gray_mold', 'healthy'], 'release_classes': ['gray_mold', 'healthy']}
        with patch.object(crop_model, 'load_model', return_value=(None, manifest)):
            capabilities = crop_model.crop_capabilities('Подсолнечник')
        self.assertEqual(capabilities['parts'], ['leaf', 'flower'])
        self.assertEqual(set(capabilities['labels']), {'gray_mold', 'healthy'})

    def test_unvalidated_probability_is_not_redistributed(self):
        class Net:
            def setInput(self, blob):
                self.shape = blob.shape

            def forward(self):
                return np.log([[.2, .8]])
        net = Net()
        manifest = {'format': 'onnx_logits_v1', 'classes': ['common_rust', 'common_smut'],
                    'release_classes': ['common_rust'], 'temperature': 1.}
        with patch.object(crop_model, 'load_model', return_value=(net, manifest)):
            result = crop_model.classify_crop('Кукуруза', np.zeros((160, 240, 3), dtype=np.uint8))
        self.assertAlmostEqual(result['common_rust'], .2)
        self.assertAlmostEqual(result['unvalidated'], .8)
        self.assertEqual(net.shape, (1, 3, 224, 224))

    def test_missing_registry_entry_means_model_not_installed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(crop_model, 'REGISTRY', Path(tmp) / 'registry.json'):
            with self.assertRaises(crop_model.ModelNotInstalledError):
                crop_model.model_directory('corn', 'leaf')
            (Path(tmp) / 'registry.json').write_text(json.dumps({'corn': {'leaf': 'backend/data/models/x'}}))
            with self.assertRaises(crop_model.ModelNotInstalledError):
                crop_model.model_directory('corn', 'ear')

    def test_registry_cannot_escape_model_storage(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(crop_model, 'REGISTRY', Path(tmp) / 'registry.json'):
            (Path(tmp) / 'registry.json').write_text(json.dumps({'corn': {'leaf': 'backend/outside'}}))
            with self.assertRaisesRegex(ValueError, 'outside'):
                crop_model.model_directory('corn', 'leaf')

    def test_unapproved_or_foreign_manifest_is_rejected(self):
        manifest = {'format': 'onnx_logits_v1', 'release_status': 'candidate', 'supported_organ': 'leaf',
                    'classes': ['healthy'], 'release_classes': ['healthy'], 'temperature': 1.}
        with self.assertRaises(crop_model.ModelNotInstalledError):
            crop_model.validate_manifest(manifest, CROPS['Кукуруза'], 'leaf')
        approved = {**manifest, 'release_status': 'approved',
                    'classes': ['potato_blight'], 'release_classes': ['potato_blight']}
        with self.assertRaisesRegex(ValueError, 'taxonomy'):
            crop_model.validate_manifest(approved, CROPS['Кукуруза'], 'leaf')

    def candidate(self, root, release_classes):
        directory = root / 'backend/data/models/corn_candidates/leaf'
        directory.mkdir(parents=True)
        (directory / 'model.onnx').write_bytes(b'test')
        manifest = {'format': 'onnx_logits_v1', 'supported_organ': 'leaf', 'classes': ['common_rust', 'healthy'],
                    'release_classes': release_classes, 'temperature': 1., 'release_status': 'candidate',
                    'files': {'model.onnx': hashlib.sha256(b'test').hexdigest()}, 'onnx_max_abs_error': 0.}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory

    def test_activation_requires_released_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / 'backend/data/models'
            directory = self.candidate(root, [])
            with patch.multiple(crop_model, ROOT=root, MODELS=models, REGISTRY=models / 'registry.json'):
                with self.assertRaisesRegex(ValueError, 'release rule'):
                    activate(directory, 'corn')
                self.assertFalse((models / 'registry.json').exists())
                result = activate(directory, 'corn', classes=['common_rust'])
            self.assertEqual(result['release_classes'], ['common_rust'])
            registry = json.loads((models / 'registry.json').read_text())
            self.assertEqual(registry, {'corn': {'leaf': 'backend/data/models/corn_candidates/leaf'}})
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(manifest['release_status'], 'approved')
            self.assertEqual(manifest['release_decision'], 'manual')


if __name__=='__main__':unittest.main()
