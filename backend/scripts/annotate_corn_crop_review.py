"""Record visually reviewed visible-leaf rectangles for the fixed 45-photo pilot.

Coordinates were read from crop_0..3 contact sheets before model predictions.
They bound the visible leaf, including symptoms, rather than individual lesions.
This is a geometry annotation, not an expert review of disease labels.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from scripts.prepare_corn_comparison import write_json

BOXES = [
    [0,0,1,1], [.12,0,.86,1], [.20,0,.72,1], [.06,0,.90,1],
    [0,.23,1,.86], [0,0,1,.96], [0,.03,1,.94], [0,.08,1,.92],
    [.10,.25,1,1], [0,0,1,.91], [.20,.02,.80,1], [0,0,1,1],
    [.30,0,.71,1], [0,0,1,1], [0,0,1,1], [0,0,1,1],
    [.20,0,.73,1], [.10,0,1,1], [0,.10,1,1], [.04,0,1,1],
    [0,.13,1,.90], [0,0,.92,1], [.04,0,1,1], [0,0,1,.95],
    [0,.31,1,.78], [.06,0,.87,1], [.12,.09,.89,1], [0,.15,1,.91],
    [0,.05,1,.76], [.08,0,.93,1], [.10,0,.82,1], [.03,0,.90,1],
    [.13,0,.82,1], [0,0,.91,1], [.20,0,.87,1], [0,0,1,1],
    [.13,0,.96,1], [.22,0,.88,1], [0,0,.94,1], [.13,0,.99,1],
    [.12,0,.98,1], [.28,0,.99,1], [.26,0,.84,1], [.12,0,.92,1], [.12,0,.88,1],
]


def main():
    directory = ROOT/'backend/data/datasets/corn_comparison_v1'
    source = directory/'crop_selection.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'db966242b417993191e52c73e21e3d025842d9e13f272ab414516b21439b5727':
        raise ValueError('Selection differs from the exact images used for manual annotation')
    rows = json.loads(source.read_text(encoding='utf-8'))
    if len(rows) != 45 or [r['source'] for r in rows] != ['idadp']*16+['pretoria']*16+['tari']*13:
        raise ValueError('This manual review belongs to the original fixed selection only')
    output = directory/'crop_annotations.json'
    if output.exists():
        raise ValueError('Do not overwrite frozen crop annotations')
    write_json(output, {
        'selection_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'reviewer':'Codex visual geometry review', 'coordinate_system':'normalized xyxy after EXIF transpose',
        'context_margin':.05,
        'purpose':'Visible leaf rectangle; keep disease labels unchanged; never used to train classifiers.',
        'detector_gate':{'minimum_net_corrected':5, 'minimum_healthy_false_positive_reduction':2,
                        'maximum_additional_false_healthy':0, 'minimum_improved_sources':2},
        'limitations':['Small, previously inspected pilot with approximate axis-aligned rectangles.',
            'Many leaves already fill the frame; unchanged rectangles are retained in paired metrics.',
            'A ground-truth crop experiment is an upper-bound intervention, not a measured automatic detector.',
            'No pathology labels were inferred or changed during geometry annotation.'],
        'records':[{**r,'bbox_xyxy':box,'annotation_basis':'visual_visible_leaf_extent'} for r,box in zip(rows,BOXES)],
    })


if __name__=='__main__':
    main()
