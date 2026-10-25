"""Import the 600-image IDADP subset; reserve the whole source for evaluation.

The ear-rot publisher label is unresolved: sampled ears show extensive dark
replacement of the ear, needing differential review against head/common smut.
Do not turn this folder into confirmed Fusarium or healthy-ear training data.
Generated image descriptions are not agronomic RAG sources.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/datasets/corn_field/idadp'
REPORTS = ROOT / 'backend/data/datasets/corn_field'
SOURCE = 'https://doi.org/10.57760/sciencedb.21290'
MAPPING = {
    '健康': ('healthy', 'leaf'),
    '南方锈病': ('southern_rust_unverified', 'leaf'),
    '大斑病': ('northern_leaf_blight', 'leaf'),
    '小斑病': ('southern_leaf_blight', 'leaf'),
    '穗腐病': ('ear_rot_unverified', 'ear'),
    '锈病': ('common_rust', 'leaf'),
}


def prepare():
    receipt = json.loads((CACHE / 'download_receipt.json').read_text(encoding='utf-8'))
    if receipt['md5'] != '984b063bb8c248f595a94a18cc39f815':
        raise ValueError('Unexpected IDADP archive version')
    annotations = json.loads((CACHE / 'IDADP_images_with_descriptions_600.json').read_text(encoding='utf-8'))
    records = []
    for annotation in annotations:
        raw_label = annotation['disease_name']
        label, organ = MAPPING[raw_label]
        path = CACHE / 'image' / raw_label / Path(annotation['path']).name
        if not path.resolve().is_relative_to(CACHE.resolve()):
            raise ValueError('Invalid annotation path')
        payload = path.read_bytes()
        records.append({'path': path.relative_to(ROOT).as_posix(), 'source': 'idadp',
            'source_url': SOURCE, 'source_id': str(annotation['idx']),
            'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
            'raw_label': raw_label, 'labels': [label], 'organ': organ,
            'license': 'CC BY 4.0', 'attribution': 'CHEN Lei; SHI Boru; YUAN Yuan (2025), IDADP, V1',
            'label_basis': 'publisher_category_not_independent_pathogen_confirmation',
            'review_required': label.endswith('_unverified'),
            'split_group_hint': 'idadp:entire_source', 'fixed_split': 'test',
            'review_note': ('Publisher ear-rot category needs expert differential review against smuts; '
                            'not confirmed Fusarium/Gibberella and not healthy.' if organ == 'ear' else '')})
    if len(records) != 600 or len({r['path'] for r in records}) != 600:
        raise ValueError('Incomplete or repeated IDADP annotations')
    result = {'source': 'idadp', 'images': len(records), 'records': records,
              'archive': receipt, 'role': 'whole-source holdout; unresolved categories excluded from training',
              'description_policy': 'Image descriptions are not imported into the agronomic knowledge base.'}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'idadp.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'images': len(records), 'southern_leaf_blight': 100, 'unresolved_ear_images': 100}))


if __name__ == '__main__':
    prepare()
