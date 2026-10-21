"""Offline checks of source integrity, multi-label mapping and split leakage."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_corn_sources import download, extract
from scripts.sample_pretoria import RemoteZip, select_rows
from scripts.audit_corn_sources import HashTree, PRETORIA_LABELS, assign_groups, assign_splits, parse_pretoria


class SourceAuditTests(unittest.TestCase):
    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escaped.txt', 'bad')
            with self.assertRaises(ValueError):
                extract(archive, root / 'extracted')
            self.assertFalse((root / 'escaped.txt').exists())

    def test_archive_case_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('leaf.JPG', 'one')
                z.writestr('LEAF.jpg', 'two')
            with self.assertRaises(ValueError):
                extract(archive, root / 'extracted')

    def test_bad_existing_download_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data.zip').write_bytes(b'bad')
            entry = {'name': 'data.zip', 'bytes': 3, 'md5': hashlib.md5(b'yes').hexdigest(), 'url': 'https://unused.invalid'}
            with patch('urllib.request.urlopen', side_effect=AssertionError('No network')), self.assertRaises(ValueError):
                download(entry, root)

    def test_download_resumes_with_exact_range_and_checks_digest(self):
        payload = b'abcdef'
        class Response(io.BytesIO):
            status = 206
            headers = {'Content-Range': 'bytes 3-5/6'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data.zip.part').write_bytes(payload[:3])
            entry = {'name': 'data.zip', 'bytes': 6, 'md5': hashlib.md5(payload).hexdigest(), 'url': 'https://example.org/data.zip'}
            with patch('urllib.request.urlopen', return_value=Response(payload[3:])) as fetch:
                path = download(entry, root)
            self.assertEqual(fetch.call_args.args[0].get_header('Range'), 'bytes=3-')
            self.assertEqual(path.read_bytes(), payload)

    def test_remote_zip_rejects_a_full_response_to_a_range_request(self):
        class Response(io.BytesIO):
            status = 200
            headers = {}
        with patch('urllib.request.urlopen', return_value=Response(b'abcdef')):
            with self.assertRaises(ValueError):
                RemoteZip('https://example.org/data.zip', 6).read(3)

    def test_pretoria_preserves_multiple_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Database.csv'
            row = dict.fromkeys(PRETORIA_LABELS, '0')
            row.update(imgID_id='1', filePath='leaf.jpg', GLS='1', NCLB='1', Other='1')
            with path.open('w', newline='', encoding='utf-8') as out:
                writer = csv.DictWriter(out, fieldnames=['imgID_id', 'filePath', *PRETORIA_LABELS])
                writer.writeheader(); writer.writerow(row)
            labels, original = parse_pretoria(path)['leaf.jpg']
            self.assertEqual(set(labels), {'gray_leaf_spot', 'northern_leaf_blight', 'other_damage'})
            self.assertEqual(original['NCLB'], '1')

    def test_sampling_is_deterministic_and_includes_each_label(self):
        rows = [{'imgID_id': str(i), 'filePath': f'{i}.jpg', 'A': str(i % 2), 'B': str(1-i%2)} for i in range(40)]
        sample = select_rows(rows, per_label=3, mixed_count=0)
        self.assertEqual(sample, select_rows(list(reversed(rows)), per_label=3, mixed_count=0))
        self.assertEqual(len(sample), 6)
        self.assertEqual(sum(r['A']=='1' for r in sample), 3)

    def test_hash_tree_matches_brute_force(self):
        rng = random.Random(42)
        values = [rng.getrandbits(16) for _ in range(200)]
        tree = HashTree()
        for i, value in enumerate(values):
            tree.add(value, i)
        for value in values[:10]:
            expected = {i for i, h in enumerate(values) if (value ^ h).bit_count() <= 4}
            self.assertEqual(set(tree.neighbors(value, 4)), expected)

    def test_cross_source_duplicates_and_conflicts_are_quarantined(self):
        rows = [
            {'source':'tari', 'labels':['healthy_leaf'], 'phash':'0', 'sha256':'a'*64, 'pixel_sha256':'1'},
            {'source':'user_review', 'labels':['common_smut'], 'phash':'1', 'sha256':'b'*64, 'pixel_sha256':'2'},
            {'source':'tom2024', 'labels':['helminthosporiosis_unspecified'], 'phash':'ffffffffffffffff', 'sha256':'c'*64, 'pixel_sha256':'3'},
        ]
        for row in rows:
            row.update(width=224, height=224, gray_std=30)
        assign_groups(rows)
        assign_splits(rows)
        self.assertEqual(rows[0]['duplicate_group'], rows[1]['duplicate_group'])
        self.assertTrue(rows[0]['label_conflict'])
        self.assertEqual(rows[0]['split'], 'quarantine')
        self.assertEqual(rows[1]['split'], 'external_review')
        self.assertIn('overlaps_external_review_group', rows[0]['exclusion_reasons'])
        self.assertEqual(rows[2]['split'], 'quarantine')
        self.assertIn('taxonomy_requires_review', rows[2]['exclusion_reasons'])

    def test_different_crops_of_same_capture_stay_together(self):
        rows = [
            {'source':'tom2024', 'labels':['healthy_leaf'], 'phash':'0', 'sha256':'a'*64, 'pixel_sha256':'1', 'capture_group_hint':'tom:123'},
            {'source':'tom2024', 'labels':['healthy_leaf'], 'phash':'ffffffffffffffff', 'sha256':'b'*64, 'pixel_sha256':'2', 'capture_group_hint':'tom:123'},
        ]
        assign_groups(rows)
        self.assertEqual(rows[0]['duplicate_group'], rows[1]['duplicate_group'])


if __name__ == '__main__':
    unittest.main()
