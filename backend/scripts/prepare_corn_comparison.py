"""Acquire original Bangladesh leaves and freeze an isolated comparison dataset.

Publisher augmented images are excluded. The entire new source is development
data (train only), avoiding an unverifiable random split of capture sequences.
Previously protected validation/test photos never move into training.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import cv2
import httpx
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
from scripts.acquire_corn_field import fetch_image
from scripts.audit_corn_sources import inspect_image, HashTree
from scripts.train_corn_field import verify_splits

DEST = ROOT/'backend/data/datasets/corn_comparison_v1'
CACHE = ROOT/'.cache/datasets/corn_comparison_v1'
FIELD = ROOT/'.cache/datasets/corn_field/manifests'
CLASSES = sorted(['common_rust', 'gray_leaf_spot', 'healthy', 'curvularia_leaf_spot',
                  'maize_lethal_necrosis', 'maize_streak_virus',
                  'northern_leaf_blight', 'southern_leaf_blight'])
MAPPING = {'Healthy_Leaf':'healthy', 'Common_Rust':'common_rust', 'Gray_Leaf_Spot':'gray_leaf_spot'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def sheet(rows, path, columns=6, size=200):
    canvas = Image.new('RGB', (columns*size, ((len(rows)+columns-1)//columns)*(size+24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        with Image.open(ROOT/row['path']) as raw:
            preview = ImageOps.contain(ImageOps.exif_transpose(raw).convert('RGB'), (size-4, size-4))
        x, y = (i%columns)*size, (i//columns)*(size+24)
        canvas.paste(preview, (x, y))
        draw.text((x+2, y+size), f'{i}: {row["labels"][0]}', fill='black')
    canvas.save(path)


def acquire():
    source_cache = CACHE/'bangladesh'
    source_cache.mkdir(parents=True, exist_ok=True)
    metadata_file = source_cache/'metadata.json'
    folders_file = source_cache/'folders.json'
    for url, path in [
        ('https://data.mendeley.com/public-api/datasets/sb4wwjs8bz', metadata_file),
        ('https://data.mendeley.com/public-api/datasets/sb4wwjs8bz/folders/1', folders_file),
    ]:
        if not path.exists():
            response = httpx.get(url, follow_redirects=True, timeout=60)
            response.raise_for_status(); write_json(path, response.json())
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    folders = json.loads(folders_file.read_text(encoding='utf-8'))
    if metadata['version'] != 1 or metadata['data_licence']['short_name'] != 'CC BY 4.0':
        raise ValueError('Publisher version/license changed; inspect before import')
    parents = {f['id']:f for f in folders}
    mapping = {f['id']:MAPPING[f['name']] for f in folders if f['name'] in MAPPING
               and parents.get(f.get('parent_id'), {}).get('name') == 'RAW_Image'}
    attribution = '; '.join(' '.join([c.get('first_name',''), c.get('last_name','')]).strip()
                            for c in metadata['contributors'])
    records = []
    for entry in metadata['files']:
        label = mapping.get(entry.get('folder_id'))
        if not label or Path(entry['filename']).suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        detail = entry['content_details']
        records.append({'path':(source_cache/'images'/label/(entry['id']+Path(entry['filename']).suffix)).relative_to(ROOT).as_posix(),
            'source':'bangladesh_raw_v1', 'source_id':entry['id'], 'original_name':entry['filename'],
            'source_url':'https://data.mendeley.com/datasets/sb4wwjs8bz/1',
            'download_url':detail['download_url'], 'expected_sha256':detail['sha256_hash'],
            'license':'CC BY 4.0', 'attribution':attribution, 'labels':[label], 'organ':'leaf',
            'raw_label':parents[entry['folder_id']]['name'], 'review_required':False,
            'label_basis':'publisher_original_class_not_independent_expert_confirmation',
            'protected':False, 'split':'train', 'exclusion_reasons':[],
            'partition_group':'bangladesh_raw_v1_entire_source'})
    results, errors = [], []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_image, r):r for r in records}
        for job in as_completed(futures):
            try:
                results.append(inspect_image(job.result()))
            except Exception as error:
                errors.append({'source_id':futures[job]['source_id'], 'error':str(error)})
            if (len(results)+len(errors))%25 == 0:
                print(f'Downloaded {len(results)}/{len(records)}; errors {len(errors)}', flush=True)
    results.sort(key=lambda r:r['path'])
    receipt = {'source':'bangladesh_raw_v1', 'source_url':'https://data.mendeley.com/datasets/sb4wwjs8bz/1',
               'version':1, 'license':metadata['data_licence'], 'attribution':attribution,
               'metadata_sha256':hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
               'requested':len(records), 'errors':errors, 'records':results,
               'excluded':'Augmented/converted/HEIC images and ambiguous disease names excluded.'}
    write_json(DEST/'acquisition.json', receipt)
    if errors:
        raise ValueError('Incomplete acquisition; rerun to retry missing files')
    return results


def prepare(new_rows):
    if (DEST/'protocol.json').exists():
        raise ValueError('Dataset already frozen; do not overwrite a running comparison')
    originals = {s:read_rows(FIELD/f'{s}.jsonl') for s in ['train','validation','test','quarantine','external_review']}
    audit = json.loads((ROOT/'backend/data/datasets/corn_field/audit.json').read_text(encoding='utf-8'))
    for split in ['train','validation','test']:
        if hashlib.sha256((FIELD/f'{split}.jsonl').read_bytes()).hexdigest() != audit['manifests'][split]:
            raise ValueError('Original dataset audit mismatch')
    existing = [r for rows in originals.values() for r in rows]
    tree = HashTree(); hashes = set(); pixels = set()
    for i, row in enumerate(existing):
        if 'phash' in row:
            tree.add(int(row['phash'],16), i)
        hashes.add(row['sha256']); pixels.add(row.get('pixel_sha256'))
    accepted, rejected = [], []
    for row in new_rows:
        reasons = []
        if 'error' in row or min(row.get('width',0),row.get('height',0)) < 96 or row.get('gray_std',0) < 5:
            reasons.append('image_quality')
        if row['sha256'] in hashes or row.get('pixel_sha256') in pixels:
            reasons.append('exact_or_decoded_duplicate')
        if 'phash' in row and list(tree.neighbors(int(row['phash'],16), radius=6)):
            reasons.append('perceptual_duplicate_of_existing_or_new_image')
        if reasons:
            rejected.append({**row, 'exclusion_reasons':reasons})
            continue
        row['duplicate_group'] = row['sha256'][:24]
        accepted.append(row)
        hashes.add(row['sha256']); pixels.add(row['pixel_sha256'])
        tree.add(int(row['phash'],16),len(existing)+len(accepted))
    splits = {s:[r for r in originals[s] if r['organ']=='leaf' and len(r['labels'])==1
                 and r['labels'][0] in CLASSES] for s in ['train','validation','test']}
    splits['train'].extend(accepted)
    verify_splits(splits)
    manifest_hashes = {}
    for split, rows in splits.items():
        path = DEST/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        manifest_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(DEST/'new_source_audit.json', {'accepted':len(accepted),'rejected':len(rejected),
        'accepted_classes':dict(Counter(r['labels'][0] for r in accepted)),
        'rejected_records':rejected, 'grouping':'Entire new source stays in train; duplicates excluded against all existing partitions.'})
    protocol = {'classes':CLASSES, 'manifest_hashes':manifest_hashes,
        'counts':{s:dict(Counter(r['labels'][0] for r in rows)) for s,rows in splits.items()},
        'sources':{s:dict(Counter(r['source'] for r in rows)) for s,rows in splits.items()},
        'architectures':['mobilenet_v3_large','efficientnet_v2_s'], 'seed':42, 'epochs':4,
        'selection':'Validation macro-F1; identical training manifests and augmentation for both models.',
        'evaluation':'Report external sources separately; legacy_kaggle is not independent of deployed baseline.',
        'crop_experiment':'Deterministically selected held-out photos; boxes drawn before reading model predictions. Same weights and image pairs.',
        'limitations':['Publisher labels and visual triage are not laboratory confirmation.',
            'Older test sources have been inspected in previous experiments; this is not a new blind benchmark.',
            'Legacy Kaggle test images may have been used in the deployed model training.',
            'Four CPU epochs compare a fixed recipe, not the maximum achievable performance of either architecture.',
            'New raw source is train-only because capture/specimen identities are unavailable.',
            'No automatic activation; diagnosis and detection must be independently reviewed.']}
    write_json(DEST/'protocol.json', protocol)
    # Stratified, fixed geometry experiment: 8 healthy and 8 diseased per source.
    selected = []
    for source in ['idadp','pretoria','tari']:
        for healthy in [True,False]:
            pool = [r for r in splits['test'] if r['source']==source and (r['labels'][0]=='healthy')==healthy]
            selected.extend(sorted(pool,key=lambda r:r['sha256'])[:8])
    write_json(DEST/'crop_selection.json', selected)
    for offset in range(0,len(selected),12):
        sheet(selected[offset:offset+12], CACHE/f'crop_{offset//12}.jpg',columns=4,size=300)
    for label in MAPPING.values():
        subset = sorted([r for r in accepted if r['labels']==[label]], key=lambda r:r['sha256'])[:18]
        if subset:
            sheet(subset,CACHE/f'new_{label}.jpg')
    print(json.dumps({'counts':protocol['counts'],'new_accepted':len(accepted),'crop_pairs':len(selected)}),flush=True)


if __name__ == '__main__':
    DEST.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    prepare(acquire())
