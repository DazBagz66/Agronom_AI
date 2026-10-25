"""Comparison integrity, paired metrics and health-error regressions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    from scripts.train_corn_comparison import crop_box, health_metrics, load_dataset
    from scripts.evaluate_corn_comparison import pair_metrics, prediction_summary
except ModuleNotFoundError as error:
    if error.name in {'torch','torchvision'}:
        raise unittest.SkipTest('Install requirements-training.txt for comparison tests') from error
    raise


class ComparisonTests(unittest.TestCase):
    def test_abstention_is_not_counted_as_a_disease_diagnosis(self):
        classes = ['healthy','rust']
        actual = np.array([0,0,0,1])
        p = np.array([[.2,.8],[.49,.51],[.9,.1],[.8,.2]])
        result = health_metrics(actual,p,classes)
        self.assertEqual(result['healthy_called_diseased_argmax'],2)
        self.assertEqual(result['healthy_called_diseased_accepted'],1)
        self.assertEqual(result['healthy_abstained'],1)
        self.assertEqual(result['diseased_called_healthy_accepted'],1)

    def test_out_of_vocabulary_prediction_counts_as_shared_class_error(self):
        result = prediction_summary([{'actual':'healthy','predicted':'southern_leaf_blight','confidence':.8}])
        self.assertEqual(result['shared_class_count'],1)
        self.assertEqual(result['shared_class_correct'],0)
        self.assertEqual(result['healthy_called_diseased_accepted'],1)

    def test_pairing_rejects_reordered_or_relabelled_photos(self):
        a = {'sha256':'a','actual':'healthy','predicted':'healthy','confidence':.9}
        b = {**a,'sha256':'b'}
        with self.assertRaisesRegex(ValueError,'identical'):
            pair_metrics([a,b],[b,a])
        with self.assertRaisesRegex(ValueError,'identical'):
            pair_metrics([a],[{**a,'actual':'common_rust'}])

    def test_crop_tradeoff_counts_regressions_as_well_as_corrections(self):
        full = [
            {'sha256':'a','actual':'healthy','predicted':'common_rust','confidence':.9},
            {'sha256':'b','actual':'common_rust','predicted':'common_rust','confidence':.8}]
        cropped = [{**full[0],'predicted':'healthy'},{**full[1],'predicted':'healthy'}]
        result = pair_metrics(full,cropped)
        self.assertEqual(result['shared_corrected'],1)
        self.assertEqual(result['shared_broken'],1)
        self.assertEqual(result['crop']['diseased_called_healthy_accepted'],1)

    def test_crop_coordinates_are_bounded_and_keep_context(self):
        image = Image.new('RGB',(200,100))
        self.assertEqual(crop_box(image,[.25,.25,.75,.75],margin=0).size,(100,50))
        self.assertEqual(crop_box(image,[0,0,1,1]).size,image.size)
        for box in [[-.1,0,1,1],[.8,0,.2,1],[0,0,1,2]]:
            with self.assertRaises(ValueError):
                crop_box(image,box)

    def test_frozen_manifest_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory/'protocol.json').write_text(json.dumps({'manifest_hashes':{'train':'0'*64}}))
            (directory/'train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'manifest changed'):
                load_dataset(directory)

    def test_valid_manifest_hash_does_not_bypass_protected_photo_check(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            photo = directory/'photo.png'
            Image.new('RGB',(100,100),(20,40,60)).save(photo)
            row = {'path':'photo.png','sha256':hashlib.sha256(photo.read_bytes()).hexdigest(),
                   'protected':True,'duplicate_group':'one','partition_group':'one'}
            hashes = {}
            for split in ['train','validation','test']:
                path = directory/f'{split}.jsonl'
                path.write_text(json.dumps(row)+'\n' if split=='train' else '')
                hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            (directory/'protocol.json').write_text(json.dumps({'manifest_hashes':hashes}))
            with patch('scripts.train_corn_field.ROOT',directory):
                with self.assertRaisesRegex(ValueError,'protected'):
                    load_dataset(directory)


if __name__=='__main__':
    unittest.main()
