import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'backend/data/datasets/corn_ear_v2'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


@unittest.skipUnless((DATA / 'dataset_card.json').exists(), 'Ear corpus not acquired')
class EarDatasetTests(unittest.TestCase):
    def test_protected_partitions_are_unchanged_and_pending_labels_do_not_train(self):
        inventory = rows(DATA / 'inventory.jsonl')
        for split in ['train', 'validation', 'test']:
            self.assertEqual(rows(DATA / f'{split}.jsonl'), rows(ROOT / f'backend/data/datasets/corn_comparison_v1/ears/{split}.jsonl'))
        for record in inventory:
            if record['source'] in {'idadp', 'hf_ricemaize_ear_rot', 'openear_development', 'bugwood_original'}:
                self.assertFalse(record['training_eligible'])
        self.assertFalse(json.loads((DATA / 'dataset_card.json').read_text())['multiclass_disease_training_ready'])

    def test_downloads_have_matching_hashes_attribution_and_original_license(self):
        for receipt in ['acquisition.json', 'bugwood_acquisition.json']:
            for row in json.loads((DATA / receipt).read_text(encoding='utf-8'))['records']:
                self.assertEqual(hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest(), row['sha256'])
                self.assertTrue(row['attribution'])
                self.assertTrue(row['license'])
                self.assertTrue(row['source_url'].startswith('https://'))
        inventory = rows(DATA / 'inventory.jsonl')
        self.assertTrue(all(not r['training_eligible'] for r in inventory if 'NC' in r['license']))


if __name__ == '__main__':
    unittest.main()
