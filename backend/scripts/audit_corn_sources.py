"""Audit corn source images, taxonomy and duplicates before any new training.

Writes a manifest and candidate splits under .cache, plus a compact report in
backend/data/datasets/corn_expansion. Does not train or touch application data.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'backend/data/datasets/corn_expansion'
CACHE = ROOT / '.cache/datasets/corn_expansion'
OUTPUT = CACHE / 'review'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
PRETORIA_LABELS = {'GLS': 'gray_leaf_spot', 'NCLB': 'northern_leaf_blight',
    'PLS': 'phaeosphaeria_leaf_spot', 'CR': 'common_rust', 'SR': 'southern_rust',
    'NoFoliarSymptoms': 'healthy_leaf', 'Other': 'other_damage', 'UnidentifiedDisease': 'unknown_disease'}
TARI_LABELS = {'ZEAMX_healthy leaves': 'healthy_leaf',
    'ZEAMX_SETOTU_Exserohilum turcicum': 'northern_leaf_blight',
    'ZEAMX_MSV000_Maize streak virus (MSV)': 'maize_streak_virus',
    'ZEAMX Maize lethal necrosis': 'maize_lethal_necrosis'}
TOM_LABELS = {'Curvulariose-D': 'curvularia_leaf_spot', 'Feuille saine': 'healthy_leaf',
    'Helminthosporiose-D': 'helminthosporiosis_unspecified', 'Rouille-D': 'rust_unspecified',
    'Striure-D': 'streak_unspecified', 'Virose-D': 'virus_unspecified', 'Maladie abiotique-D': 'abiotic_damage_unspecified'}
LEGACY_LABELS = {'Blight': 'blight_unspecified', 'Common_Rust': 'common_rust',
                 'Gray_Leaf_Spot': 'gray_leaf_spot', 'Healthy': 'healthy_leaf'}
AMBIGUOUS = {'blight_unspecified', 'helminthosporiosis_unspecified', 'rust_unspecified',
             'streak_unspecified', 'virus_unspecified', 'abiotic_damage_unspecified'}
HOLDOUT_SOURCES = {'pretoria', 'user_review'}


def parse_pretoria(path):
    result = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {'imgID_id', 'filePath', *PRETORIA_LABELS}:
            raise ValueError('Unexpected Pretoria CSV schema; review mapping')
        for row in reader:
            if row['filePath'] in result:
                raise ValueError('Duplicate Pretoria filename')
            if any(row[k] not in {'0', '1'} for k in PRETORIA_LABELS):
                raise ValueError('Non-binary Pretoria annotation')
            labels = sorted(PRETORIA_LABELS[k] for k in PRETORIA_LABELS if row[k] == '1')
            if not labels:
                raise ValueError('Unlabelled Pretoria image')
            result[row['filePath']] = (labels, row)
    return result


def image_paths(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)


def enumerate_images():
    def record(path, source, labels, raw, **extra):
        return {'path': path.relative_to(ROOT).as_posix(), 'source': source, 'labels': labels,
                'raw_label': raw, 'organ': 'leaf', 'organ_basis': 'publisher_dataset_scope_not_image_verified', **extra}
    old = ROOT / '.cache/datasets/corn/images/data'
    for path in image_paths(old):
        if path.parent.name not in LEGACY_LABELS:
            raise ValueError(f'Unknown original dataset label: {path.parent.name}')
        yield record(path, 'legacy_kaggle', [LEGACY_LABELS[path.parent.name]], path.parent.name)
    for source, mapping in [('tari', TARI_LABELS), ('tom2024', TOM_LABELS)]:
        folder = CACHE / source / 'extracted'
        # Require the entire source's installer receipt, never silently use a partial archive.
        if not (REPORTS / f'{source}_installed.json').exists():
            continue
        for path in image_paths(folder):
            ancestors = [part for part in path.relative_to(folder).parts[:-1] if part in mapping]
            if not ancestors or len(set(ancestors)) != 1:
                raise ValueError(f'Unknown/ambiguous {source} label: {path}')
            raw = ancestors[0]
            row = record(path, source, [mapping[raw]], raw)
            # TOM filenames contain a millisecond timestamp. Group equal timestamps
            # conservatively even if two crops of the original photo differ in pHash.
            if source == 'tom2024':
                stamp = re.search(r'(?:^|_)(\d{13})(?:_|\.)', path.name)
                if stamp:
                    row['capture_group_hint'] = source + ':' + stamp[1]
            yield row
    csv_path = CACHE / 'pretoria/archives/Database.csv'
    if csv_path.exists():
        annotations = parse_pretoria(csv_path)
        full = CACHE / 'pretoria/extracted/leaf_images/leaf_images'
        sample = CACHE / 'pretoria/sample'
        folder = full if (REPORTS / 'pretoria_installed.json').exists() else sample
        if folder == sample:
            receipt_path = REPORTS / 'pretoria_sample.json'
            if not receipt_path.exists():
                raise ValueError('Pretoria sample is incomplete; finish sampling first')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            paths = [ROOT / r['local_path'] for r in receipt['images']]
            if len(paths) != receipt['selected_images']:
                raise ValueError('Incomplete sample receipt')
        else:
            paths = image_paths(folder)
            if len(paths) != len(annotations):
                raise ValueError('Incomplete Pretoria extraction')
        for path in paths:
            labels, raw = annotations[path.name]
            yield record(path, 'pretoria', labels, raw, source_image_id=raw['imgID_id'])
    user_path = ROOT / 'backend/data/evaluation/corn_external_review.json'
    if user_path.exists():
        for row in json.loads(user_path.read_text(encoding='utf-8'))['cases']:
            label = 'healthy_leaf' if row['user_label'] == 'healthy' else row['user_label']
            path = ROOT / row['image_path']
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError('User review image missing or changed')
            yield {**record(path, 'user_review', [label], row['user_label_name']),
                   'organ': 'unverified', 'organ_basis': 'not_recorded', 'history_id': row['history_id'],
                   'label_basis': 'user_report_not_independently_verified'}


def inspect_image(row):
    data = (ROOT / row['path']).read_bytes()
    row = {**row, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {**row, 'error': 'unreadable_image'}
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (32, 32)).astype(np.float32)
    values = cv2.dct(reduced)[:8, :8].ravel()[1:]
    bits = values > np.median(values)
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    row.update(width=w, height=h, phash=f'{phash:016x}',
               pixel_sha256=hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
               gray_std=round(float(gray.std()), 3),
               laplacian_variance=round(float(cv2.Laplacian(cv2.resize(gray, (224,224)), cv2.CV_64F).var()), 3))
    return row


class HashTree:
    """BK-tree on Hamming distance, so candidate search does not scan all pairs."""
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def neighbors(self, value, radius=6):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                yield from node[1]
            pending.extend(child for edge, child in node[2].items() if distance-radius <= edge <= distance+radius)


def assign_groups(rows):
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[root(a)] = root(b)
    tree = HashTree()
    exact = {}
    captures = {}
    for i, row in enumerate(rows):
        if 'error' in row:
            continue
        value = int(row['phash'], 16)
        for j in tree.neighbors(value):
            union(i, j)
        tree.add(value, i)
        for field in ['sha256', 'pixel_sha256']:
            key = (field, row[field])
            if key in exact:
                union(i, exact[key])
            exact[key] = i
        capture = row.get('capture_group_hint')
        if capture:
            if capture in captures:
                union(i, captures[capture])
            captures[capture] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    for members in groups.values():
        group_id = min(r['sha256'] for r in members)[:24]
        sources = {r['source'] for r in members}
        labels = {tuple(r['labels']) for r in members}
        for row in members:
            row['duplicate_group'] = group_id
            row['group_size'] = len(members)
            row['group_sources'] = sorted(sources)
            row['label_conflict'] = len(labels) > 1
            row['overlaps_external'] = bool(sources & HOLDOUT_SOURCES)
    return list(groups.values())


def assign_splits(rows):
    """Quarantine ambiguous/low-quality/conflicting data; keep holdout groups out of development."""
    representatives = set()
    holdout = REPORTS / 'legacy_holdout.json'
    if holdout.exists():
        legacy_reserved = set(json.loads(holdout.read_text(encoding='utf-8'))['reserved_sha256'])
    else:
        legacy_report = json.loads((ROOT / 'backend/data/models/corn/training_report.json').read_text(encoding='utf-8'))
        if legacy_report.get('schema_version', 1) != 1:
            raise ValueError('Original holdout pin missing; do not use the expanded model to redefine splits')
        legacy_reserved = {r['sha256'] for s in ['test', 'validation'] for r in legacy_report['splits'][s]}
    reserved_groups = {r['duplicate_group'] for r in rows if r['sha256'] in legacy_reserved}
    for row in rows:
        row['split'] = 'quarantine'
        row['exclusion_reasons'] = []
        if row['source'] in HOLDOUT_SOURCES:
            row['split'] = 'external_review'
            continue
        if 'error' in row:
            row['exclusion_reasons'].append(row['error'])
        elif min(row['width'], row['height']) < 96 or row['gray_std'] < 5:
            row['exclusion_reasons'].append('low_resolution_or_flat_image')
        if row['label_conflict']:
            row['exclusion_reasons'].append('similar_images_with_different_labels_review_required')
        if row['overlaps_external']:
            row['exclusion_reasons'].append('overlaps_external_review_group')
        if set(row['labels']) & AMBIGUOUS:
            row['exclusion_reasons'].append('taxonomy_requires_review')
        if row['duplicate_group'] in reserved_groups:
            row['exclusion_reasons'].append('reserved_by_existing_model_validation_or_test')
        if row['exclusion_reasons']:
            continue
        if row['duplicate_group'] in representatives:
            row['exclusion_reasons'].append('duplicate_representative_already_selected')
            continue
        representatives.add(row['duplicate_group'])
        # Deterministic group split, provisional until class counts are reviewed.
        row['split'] = 'validation_candidate' if int(row['duplicate_group'][:8], 16) % 5 == 0 else 'train_candidate'


def main():
    cv2.setNumThreads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / 'image_audit_cache.json'
    cached = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    rows = []
    next_cache = {}
    for raw in enumerate_images():
        stat = (ROOT / raw['path']).stat()
        fingerprint = [stat.st_size, stat.st_mtime_ns]
        entry = cached.get(raw['path'])
        if entry and entry['fingerprint'] == fingerprint:
            row = {**entry['row'], **raw}
        else:
            row = inspect_image(raw)
        next_cache[raw['path']] = {'fingerprint': fingerprint, 'row': row}
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f'Inspected {len(rows)} images', flush=True)
    if not rows:
        raise ValueError('No source images installed')
    cache_path.write_text(json.dumps(next_cache, ensure_ascii=False), encoding='utf-8')
    groups = assign_groups(rows)
    assign_splits(rows)
    with (OUTPUT / 'manifest.jsonl').open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    for split in ['train_candidate', 'validation_candidate', 'external_review', 'quarantine']:
        with (OUTPUT / f'{split}.jsonl').open('w', encoding='utf-8') as out:
            for row in rows:
                if row['split'] == split:
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
    by_source = {}
    for source in sorted({r['source'] for r in rows}):
        subset = [r for r in rows if r['source'] == source]
        by_source[source] = {'images': len(subset),
            'labels': dict(Counter(label for r in subset for label in r['labels'])),
            'splits': dict(Counter(r['split'] for r in subset)),
            'low_resolution': sum(min(r.get('width',0),r.get('height',0)) < 96 for r in subset),
            'unreadable': sum('error' in r for r in subset)}
    report = {'sources': by_source, 'total_images': len(rows),
        'duplicate_method': 'Exact file SHA256, decoded pixels SHA256, 63-bit pHash distance <=6, inferred equal TOM filename timestamps; candidate matches require review, crops/rotations may be missed.',
        'duplicate_groups': len(groups), 'groups_with_repeats': sum(len(g)>1 for g in groups),
        'cross_source_groups': sum(len({r['source'] for r in g})>1 for g in groups),
        'conflicting_groups': sum(g[0]['label_conflict'] for g in groups),
        'splits': dict(Counter(r['split'] for r in rows)),
        'exclusion_reasons': dict(Counter(reason for r in rows for reason in r['exclusion_reasons'])),
        'train_classes': dict(Counter(label for r in rows if r['split']=='train_candidate' for label in r['labels'])),
        'validation_classes': dict(Counter(label for r in rows if r['split']=='validation_candidate' for label in r['labels'])),
        'target_gaps': ['southern_leaf_blight: no acquired training data', 'common_smut: no acquired training data',
                        'ear/stem/root diseases: no acquired training data', 'healthy ears/stems/roots: no acquired training data'],
        'limitations': ['Candidate splits are not a trained model or validated diagnoses.',
                       'Pretoria sample is deliberately stratified, not representative of field prevalence.',
                       'Existing model was already trained on legacy train; legacy is not an independent field test.',
                       'No guaranteed separation by plant/site where source metadata is absent.',
                       'Unknown/multiple diseases remain in the external review; not forced to one class.',
                       'Downloaded licence attribution is in sources.json and installed receipts.'],
        'manifest': (OUTPUT/'manifest.jsonl').relative_to(ROOT).as_posix()}
    (REPORTS / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
