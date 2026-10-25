"""Remove verified archive duplicates and disposable caches; preserve all images/models.

Default is an audit. --apply removes only the fixed, project-local allowlist.
Run with dataset downloads and package installations stopped.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def safe_path(root, relative):
    root = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError(f'Path outside workspace: {relative}')
    # Reject symlinks/junctions even if they point back into the workspace.
    current = path
    while current != root:
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'Reparse point not supported: {relative}')
        current = current.parent
    return path


def inventory(root):
    total, groups = 0, {}
    def scan(directory, group=None):
        nonlocal total
        with os.scandir(directory) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    continue
                category = group or entry.name
                if entry.is_dir(follow_symlinks=False):
                    scan(entry.path, category)
                else:
                    total += info.st_size
                    groups[category] = groups.get(category, 0) + info.st_size
    scan(root)
    return {'logical_bytes': total, 'top_level_bytes': groups}


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified_archive(root, archive_relative, extracted_relative):
    """Every ZIP member must have a byte-identical, independently stored copy."""
    archive = safe_path(root, archive_relative)
    extracted = safe_path(root, extracted_relative)
    if not archive.is_file() or not extracted.is_dir():
        raise ValueError('Archive or extracted directory missing')
    before = archive.stat()
    checked, seen = 0, set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            destination = safe_path(root, extracted.relative_to(root) / member.filename)
            if not destination.is_relative_to(extracted) or destination == extracted:
                raise ValueError(f'Unsafe member: {member.filename}')
            key = str(destination).casefold()
            if key in seen or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Duplicate member or symlink: {member.filename}')
            seen.add(key)
            if not destination.is_file() or destination.stat().st_size != member.file_size:
                raise ValueError(f'Missing/different extracted member: {member.filename}')
            with source.open(member) as original, destination.open('rb') as copy:
                while chunk := original.read(1024 * 1024):
                    if chunk != copy.read(len(chunk)):
                        raise ValueError(f'Extracted content differs: {member.filename}')
                if copy.read(1):
                    raise ValueError(f'Extracted file grew: {member.filename}')
            checked += 1
    if not checked:
        raise ValueError('Empty archive')
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Archive changed during verification')
    return {'path': str(archive_relative), 'extracted': str(extracted_relative),
            'bytes': after.st_size, 'sha256': file_sha(archive), 'verified_members': checked,
            'mtime_ns': after.st_mtime_ns}


def archive_pairs(root):
    for crop in ['corn', 'sunflower']:
        yield f'.cache/datasets/{crop}/dataset.zip', f'.cache/datasets/{crop}/images'
    for source in ['tari', 'tom2024']:
        folder = safe_path(root, f'.cache/datasets/corn_expansion/{source}/archives')
        for path in sorted(folder.glob('*.zip')):
            yield path.relative_to(root).as_posix(), f'.cache/datasets/corn_expansion/{source}/extracted/{path.stem}'
    yield '.cache/datasets/corn_field/anhui/Corn-leaf-disease-data-main.zip', '.cache/datasets/corn_field/anhui/extracted'


DISPOSABLE = [
    '.cache/npm', '.cache/pip',
    # Derived 320px training copies; Images.__getitem__ rebuilds these from originals.
    '.cache/datasets/corn_comparison_v1/rgb320_v1',
    '.cache/datasets/corn_comparison_v1/rgb320_mobile_v1',
    '.cache/features/corn_expanded',
    '.cache/features/corn_organs/bfa8f7bdd68ff12b519fe81f420ba370904b33b6580b01fb907bd05a3b6959cc-trim-square224-v1',
    '.cache/datasets/corn_expansion/pretoria/archives/leaf_images.zip.part',
    '.cache/datasets/corn_expansion/pretoria/archives/leaf_images.zip.part.chunks',
]


def disposable_size(root, relative):
    path = safe_path(root, relative)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    # Validate descendants before recursive removal, including Windows junctions.
    total = 0
    for folder, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            child = safe_path(root, (Path(folder) / name).relative_to(root))
            if child.is_file():
                total += child.stat().st_size
    return total


def optimize(root=ROOT, apply=False):
    root = root.resolve()
    output = safe_path(root, 'backend/data/maintenance')
    output.mkdir(parents=True, exist_ok=True)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'applied': apply,
              'before': inventory(root), 'actions': [], 'skipped': []}
    report_path = output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '_storage.json')
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    for archive, extracted in archive_pairs(root):
        if not safe_path(root, archive).exists():
            continue
        print(f'Verifying {archive}', flush=True)
        try:
            item = {'kind': 'duplicate_archive', **verified_archive(root, archive, extracted), 'removed': False}
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            report['skipped'].append({'path': archive, 'reason': str(error)})
            save()
            continue
        report['actions'].append(item)
        save()  # Durable receipt before deleting an archive.
        if apply:
            path = safe_path(root, archive)
            info = path.stat()
            if (info.st_size, info.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
                raise ValueError('Archive changed after verification')
            path.unlink()
            item['removed'] = True
            save()
    for relative in DISPOSABLE:
        size = disposable_size(root, relative)
        if not size:
            continue
        item = {'kind': 'disposable_cache', 'path': relative, 'bytes': size, 'removed': False}
        report['actions'].append(item)
        save()
        if apply:
            path = safe_path(root, relative)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            item['removed'] = True
            save()
    report['after'] = inventory(root)
    report['removed_bytes'] = sum(x['bytes'] for x in report['actions'] if x['removed'])
    save()
    print(json.dumps({'report': str(report_path), 'before_bytes': report['before']['logical_bytes'],
                      'after_bytes': report['after']['logical_bytes'], 'removed_bytes': report['removed_bytes'],
                      'skipped': report['skipped']}, ensure_ascii=True), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    optimize(apply=parser.parse_args().apply)
