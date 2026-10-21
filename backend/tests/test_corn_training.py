"""Training data isolation and source weighting; no real training or network."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_corn_expanded import LABELS, augmented, class_source_weights, validate_splits


class CornTrainingTests(unittest.TestCase):
    def rows(self, prefix):
        return [{'source':'tari','labels':['healthy_leaf' if label=='healthy' else label],
                 'sha256':prefix+str(i),'duplicate_group':prefix+str(i)} for i,label in enumerate(LABELS)]

    def test_external_sources_cannot_enter_training(self):
        train,val=self.rows('t'),self.rows('v')
        validate_splits(train,val,[])
        train[0]['source']='user_review'
        with self.assertRaises(ValueError):validate_splits(train,val,[])

    def test_group_leakage_and_ambiguous_labels_are_rejected(self):
        train,val=self.rows('t'),self.rows('v')
        external=[copy.deepcopy(train[0])]
        with self.assertRaises(ValueError):validate_splits(train,val,external)
        train[0]['labels']=['blight_unspecified']
        with self.assertRaises(ValueError):validate_splits(train,val,[])

    def test_weights_balance_classes_and_sources_within_class(self):
        rows=[{'source':'legacy_kaggle','labels':['healthy_leaf']} for _ in range(10)]
        rows += [{'source':'tari','labels':['healthy_leaf']} for _ in range(2)]
        rows += [{'source':'tari','labels':['northern_leaf_blight']} for _ in range(3)]
        weights=class_source_weights(rows)
        self.assertAlmostEqual(weights.mean(),1.)
        self.assertAlmostEqual(weights[:10].sum(),weights[10:12].sum())
        self.assertAlmostEqual(weights[:12].sum(),weights[12:].sum())

    def test_augmentation_is_reproducible_without_changing_input(self):
        image=np.random.default_rng(42).integers(0,256,(120,180,3),dtype=np.uint8)
        original=image.copy()
        self.assertTrue(np.array_equal(augmented(image,42),augmented(image,42)))
        self.assertTrue(np.array_equal(image,original))


if __name__=='__main__':unittest.main()
