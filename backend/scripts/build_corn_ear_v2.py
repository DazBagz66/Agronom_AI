"""Build a reproducible ear corpus with explicit provenance/review gates.

Keep research examples and unsupported diagnoses separate from training manifests.
No web folder or visual appearance is promoted to a pathogen-species label.
"""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.prepare_corn_comparison import read_rows, write_json

OUT = ROOT / 'backend/data/datasets/corn_ear_v2'
RECEIPT_HASH = 'ea26a093a6ebd25fa6c06c64a19c5d8551f729332b9d3c22d357821de4cbbba4'
# Explicit contact-sheet triage, indexed in the pinned acquisition receipt.
# All remaining rot photographs show an exposed ear with discoloration/mold-like
# lesions. These are broad visual observations, NOT species diagnoses.
ROT_EXCLUDE = {19:'labelled_collage', 26:'labelled_collage', 27:'labelled_collage',
    36:'smut_like_or_deformed_ear', 37:'whole_plant', 38:'whole_plant', 46:'detached_kernels',
    48:'collage', 50:'labelled_collage', 52:'field_view', 57:'labelled_collage',
    61:'labelled_collage', 62:'labelled_collage', 72:'smut_like_galls', 79:'collage',
    83:'closed_husk', 84:'specimen_plate', 88:'labelled_collage', 92:'whole_plant',
    95:'labelled_collage', 99:'single_kernel', 100:'smut_like_or_deformed_ear',
    101:'collage', 102:'labelled_collage', 114:'single_kernel'}


def write_rows(name, rows):
    path = OUT / name
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = OUT / 'acquisition.json'
    if hashlib.sha256(receipt.read_bytes()).hexdigest() != RECEIPT_HASH:
        raise ValueError('New acquisition receipt requires visual review')
    fresh = json.loads(receipt.read_text(encoding='utf-8'))['records']
    original = json.loads((OUT / 'bugwood_acquisition.json').read_text(encoding='utf-8'))['records']
    records = []
    protected = {split: read_rows(ROOT / f'backend/data/datasets/corn_comparison_v1/ears/{split}.jsonl')
                 for split in ['train', 'validation', 'test']}
    for split, rows in protected.items():
        for row in rows:
            row = dict(row, role='research_training' if split == 'train' else 'protected_' + split,
                disease_expert_verified=False, training_eligible=split == 'train', review_status='prior_visual_review',
                intended_label=row['labels'][0], label_scope='healthy_ear means no visible smut ONLY; rot not excluded')
            records.append(row)
    for i, row in enumerate(fresh):
        row = dict(row, disease_expert_verified=False, training_eligible=False, split='quarantine', review_index=i)
        if row['source'] == 'hf_ricemaize_ear_rot':
            reason = ROT_EXCLUDE.get(i)
            row.update(role='excluded' if reason else 'license_and_expert_review',
                intended_label='unknown' if reason else 'ear_rot',
                visual_label='excluded' if reason else 'mold_or_rot_like_ear',
                review_status='visual_review_completed', organ='unknown' if reason else 'ear',
                exclusion_reasons=['original_image_provenance_unverified'] + ([reason] if reason else ['pathogen_unverified']),
                review_note='Codex contact-sheet visual triage. Visible rot-like changes only; no fungal species assigned. Some source pictures reproduce NC Bugwood images despite the uploader MIT declaration.')
        else:
            row.update(role='auxiliary_development', intended_label=row['raw_label'], review_status='visual_review_completed',
                domain='unwrapped_ear_scan', exclusion_reasons=['domain_shift_from_phone_photographs', 'development_is_not_health'],
                review_note='Codex visual review: mechanically unwrapped ear scans, not ordinary phone images. Retain publisher development label only; do not use as a disease/healthy class.')
        records.append(row)
    for i, row in enumerate(original):
        microscope = row['source_id'] in {'5405281', '5405282', '5405283', '5405284'}
        row = dict(row, role='excluded' if microscope else 'noncommercial_reference' if 'NC' in row['license'] else 'licensed_pathogen_example',
            organ='microscopy' if microscope else 'ear', intended_label=row['labels'][0],
            training_eligible=False, disease_expert_verified=False, review_status='visual_review_completed',
            exclusion_reasons=['not_a_phone_ear_image'] if microscope else ['noncommercial_license'] if 'NC' in row['license'] else ['insufficient_independent_class_support'],
            review_note='Original catalog host and pathogen attribution restored; organ manually checked. Single licensed original is insufficient to train a disease class.')
        records.append(row)
    # The already reserved IDADP source remains a holdout review queue.
    for row in json.loads((ROOT / 'backend/data/datasets/corn_field/idadp.json').read_text(encoding='utf-8'))['records']:
        if row['organ'] == 'ear':
            records.append(dict(row, role='protected_external_review', split='external_review',
                intended_label='ear_rot_unverified', training_eligible=False, disease_expert_verified=False,
                review_status='expert_differential_required',
                exclusion_reasons=['reserved_external_source', 'smut_vs_rot_differential_unresolved']))
    # Exact and perceptual duplicate components, including transitive matches,
    # are recorded across ALL roles. Existing protected splits never move.
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        parent[root(b)] = root(a)
    duplicate_pairs = []
    for i, row in enumerate(records):
        for j in range(i):
            other = records[j]
            exact = row['sha256'] == other['sha256'] or (row.get('pixel_sha256') and row.get('pixel_sha256') == other.get('pixel_sha256'))
            near = row.get('phash') and other.get('phash') and (int(row['phash'], 16) ^ int(other['phash'], 16)).bit_count() <= 6
            if exact or near:
                union(j, i)
                duplicate_pairs.append({'left':j, 'right':i, 'kind':'exact' if exact else 'perceptual_candidate'})
    for i, row in enumerate(records):
        row['corpus_id'] = f'ear-v2-{i:04d}'
        row['review_duplicate_group'] = f'ear-v2-group-{root(i):04d}'
    hashes = {'inventory.jsonl': write_rows('inventory.jsonl', records)}
    for split, rows in protected.items():
        hashes[split + '.jsonl'] = write_rows(split + '.jsonl', rows)
    for role in sorted({r['role'] for r in records} - {'research_training', 'protected_validation', 'protected_test'}):
        hashes[role + '.jsonl'] = write_rows(role + '.jsonl', [r for r in records if r['role'] == role])
    fields = ['corpus_id','path','source','source_url','license','attribution','intended_label','role','review_status',
              'review_duplicate_group','expert_label','expert_name','expert_notes']
    with (OUT / 'expert_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    summary = {'records':len(records), 'unique_sha256':len({r['sha256'] for r in records}),
        'new_downloads':len(fresh)+len(original), 'roles':dict(Counter(r['role'] for r in records)),
        'proposed_visual_rot_examples':sum(r.get('visual_label') == 'mold_or_rot_like_ear' for r in records),
        'sources':dict(Counter(r['source'] for r in records)), 'duplicate_pairs':duplicate_pairs,
        'multiclass_disease_training_ready':False, 'activated':False, 'hashes':hashes,
        'blocked_by':['Independent expert labels for rot subtypes and original image usage rights.',
                      'Natural phone photos of ears without visible disease; normal-development scans are not substitutes.',
                      'Independent test specimens per class; current no-smut test contains only four examples.'],
        'split_policy':'Prior ear train/validation/test retained exactly; new sources never enter them automatically. IDADP remains external review. Split future accepted data by photographer, plot/specimen and duplicate component.'}
    write_json(OUT / 'dataset_card.json', summary)
    cards = []
    for row in records:
        relative = '../../../../' + row['path']
        cards.append(f'<article><img loading="lazy" src="{html.escape(relative)}"><p><b>{row["corpus_id"]}</b> — {html.escape(row["intended_label"])}</p><p>{html.escape(row["role"])} · {html.escape(row["license"])}</p><p>{html.escape(row.get("attribution", ""))}</p><a href="{html.escape(row["source_url"])}">Первоисточник</a></article>')
    (OUT / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Початки: проверка датасета v2</title><style>body{font:15px system-ui;background:#eee;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}article{background:white;padding:12px}img{width:100%;height:220px;object-fit:contain}p{overflow-wrap:anywhere}</style><h1>Початки: датасет v2</h1><p>Предварительная разметка. Категории гнилей требуют экспертной проверки. Заполняйте expert_review.csv; изображения и источники сохранены.</p><main>' + ''.join(cards) + '</main>', encoding='utf-8')
    print(json.dumps({key:value for key,value in summary.items() if key not in {'duplicate_pairs','hashes'}}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
