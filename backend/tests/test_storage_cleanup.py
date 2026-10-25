"""Deletion preconditions: intact copies, traversal protection and audit-only default."""
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.optimize_storage import verified_archive, safe_path, optimize


class StorageCleanupTests(unittest.TestCase):
    def make_archive(self, root, content=b'original'):
        archive = root / '.cache/datasets/corn/dataset.zip'
        archive.parent.mkdir(parents=True)
        image = archive.parent / 'images/data/leaf.png'
        image.parent.mkdir(parents=True)
        image.write_bytes(content)
        with zipfile.ZipFile(archive, 'w') as out:
            out.writestr('data/leaf.png', b'original')
        return archive, image

    def test_audit_keeps_files_and_apply_removes_only_verified_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, image = self.make_archive(root)
            model = root / 'model.onnx'
            model.write_bytes(b'weights')
            optimize(root)
            self.assertTrue(archive.exists())
            report = optimize(root, apply=True)
            self.assertFalse(archive.exists())
            self.assertEqual(image.read_bytes(), b'original')
            self.assertEqual(model.read_bytes(), b'weights')
            self.assertGreater(report['removed_bytes'], 0)

    def test_same_size_but_different_extracted_content_keeps_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _ = self.make_archive(root, b'changed!')
            report = optimize(root, apply=True)
            self.assertTrue(archive.exists())
            self.assertEqual(report['removed_bytes'], 0)
            self.assertIn('differs', report['skipped'][0]['reason'])

    def test_missing_copy_keeps_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, image = self.make_archive(root)
            image.unlink()
            with self.assertRaises(ValueError):
                verified_archive(root, archive.relative_to(root), '.cache/datasets/corn/images')
            self.assertTrue(archive.exists())

    def test_paths_cannot_escape_workspace_or_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                safe_path(root, '../outside')
            archive, _ = self.make_archive(root)
            with zipfile.ZipFile(archive, 'w') as out:
                out.writestr('../escape', b'data')
            with self.assertRaises(ValueError):
                verified_archive(root, archive.relative_to(root), '.cache/datasets/corn/images')
            self.assertTrue(archive.exists())


if __name__ == '__main__':
    unittest.main()
