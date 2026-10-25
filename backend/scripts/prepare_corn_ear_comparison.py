"""Apply explicit organ triage to new licensed observations and freeze ear splits.

The negative class means 'no visible smut', NOT a healthy ear or absence of rot.
Only the publisher/community smut label is retained; no rot pathogen is invented.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from scripts.build_corn_field import split_rows
from scripts.prepare_corn_comparison import read_rows, write_json
from scripts.train_corn_field import verify_splits


def main():
    cv2.setNumThreads(1)
    directory = ROOT/'backend/data/datasets/corn_comparison_v1'
    output = directory/'ears'
    if (output/'protocol.json').exists():
        raise ValueError('Ear experiment already frozen')
    output.mkdir(exist_ok=True)
    receipt_path = directory/'ear_acquisition.json'
    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != '91e6ead7d498e7bd5f2654397d9356b1de85bc0babc04f562f28c03ad420f539':
        raise ValueError('Ear receipt differs from the exact images visually reviewed')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    records = receipt['records']
    if len(records)!=94 or records[34]['source']!='inat_smut_recent':
        raise ValueError('Visual index belongs to the reviewed 94-image receipt only')
    # Full-resolution negatives 11,12,13 inspected separately; the two remaining
    # negatives have exposed kernels in the fixed contact sheets.
    negatives = {11,12,13,15,23}
    # Exclude tassels, detached galls and anatomically ambiguous tissues.
    positives = set(range(34,94)) - {35,38,42,44,50,52,62,77,83}
    reviews = {}
    decisions = []
    for i,row in enumerate(records):
        label = 'common_smut' if i in positives else 'healthy_ear' if i in negatives else None
        decision = {'sha256':row['sha256'],'path':row['path'],'index':i,
                    'status':'accepted_research_only' if label else 'pending',
                    'reviewer':'Codex visual organ/smut triage',
                    'label':label,'expert_verified':False,
                    'evidence':'Retained community research-grade smut identification; ear visible.' if i in positives else
                               'Exposed kernels; no visible smut galls. Other damage/rot not excluded.' if label else
                               'Closed husk, whole plant, tassel, detached tissue or unclear organ; no health inference.'}
        decisions.append(decision)
        if label:
            reviews[row['sha256']] = {'status':'accepted','labels':[label],'organ':'ear',
                'reviewer':decision['reviewer'],'evidence':decision['evidence']}
    all_rows = []
    for split in ['train','validation','test','quarantine','external_review']:
        for row in read_rows(ROOT/f'.cache/datasets/corn_field/manifests/{split}.jsonl'):
            if split=='test':
                row['protected'] = True
            if split=='validation':
                row['previous_split'] = 'validation_candidate'
            all_rows.append(row)
    old_hashes = {r['sha256'] for r in all_rows}
    for row in records:
        if row['sha256'] in old_hashes:
            continue
        row = dict(row)
        row['split_group_hint'] = row.pop('capture_group_hint')
        row['protected'] = False
        all_rows.append(row)
    # Same-observer series and perceptual duplicates propagate all old holdouts.
    grouped = split_rows(all_rows,reviews)
    splits = {s:[r for r in grouped if r['split']==s and r['organ']=='ear'
                 and r['labels'] in [['common_smut'],['healthy_ear']]] for s in ['train','validation','test']}
    verify_splits(splits)
    hashes = {}
    for split, rows in splits.items():
        path = output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = {s:dict(Counter(r['labels'][0] for r in rows)) for s,rows in splits.items()}
    write_json(output/'protocol.json',{'classes':['common_smut','healthy_ear'],'organ':'ear',
        'manifest_hashes':hashes,'counts':counts,'epochs':4,'seed':42,
        'selection':'Validation macro-F1 only; identical splits/augmentation for both architectures.',
        'limitations':['Research-only community labels and non-expert visual triage.',
            'healthy_ear means no visible smut, not absence of other diseases.',
            'Very few independent negative ears; no release eligibility.',
            'Rot, head smut and specific fungal species are not represented by validated classes.',
            'Existing protected test and validation groups preserved; sources were previously inspected.']})
    write_json(output/'visual_reviews.json',{'receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'decisions':decisions,'counts':dict(Counter(d['status'] for d in decisions))})
    known_ear = {r['sha256'] for split in ['train','validation','test']
                 for r in read_rows(ROOT/f'.cache/datasets/corn_field/manifests/{split}.jsonl') if r['organ']=='ear'}
    write_json(output/'data_audit.json',{'counts':counts,
        'new_usable':sum(r['sha256'] not in known_ear for rows in splits.values() for r in rows),
        'new_usable_by_class':dict(Counter(r['labels'][0] for rows in splits.values() for r in rows if r['sha256'] not in known_ear)),
        'new_quarantined_after_review':[{'sha256':r['sha256'],'reasons':r['exclusion_reasons']} for r in grouped
            if r['sha256'] in reviews and r['split'] in {'quarantine','external_review'}],
        'additional_detail_photos':35,
        'detail_review':'Mostly closed husks/whole plants/tassels or same ears already reviewed. Retained separately, not health-labelled.',
        'kernel_counting_source':'Not imported: terms restrict use to internal non-commercial academic research; project intended use not established.',
        'activated':False})
    print(json.dumps({'ear_counts':counts}),flush=True)


if __name__=='__main__':
    main()
