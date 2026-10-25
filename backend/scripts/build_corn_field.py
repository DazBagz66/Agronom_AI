"""Build versioned training manifests; protected tests and unresolved labels stay out.

Reviews are CSV: sha256,labels (semicolon separated),organ,reviewer,evidence,
status (accepted/rejected). Labels changing a publisher's diagnosis require
expert evidence; visual consistency alone is not a pathogen identification.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
from app.corn_taxonomy import CORN_LABELS
from scripts.audit_corn_sources import inspect_image, assign_groups

REPORTS = ROOT/'backend/data/datasets/corn_field'
OUTPUT = ROOT/'.cache/datasets/corn_field/manifests'
ANHUI_LABELS = {'Bipolaris_Maydis': 'southern_leaf_blight', 'Common_Rust': 'common_rust',
    'Curvularia_Lunata(wakker)boed_Spot': 'curvularia_leaf_spot', 'Northern-Leaf-Blight': 'northern_leaf_blight'}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def read_reviews(path):
    if path is None:
        return {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        if row.get('status') not in {'accepted','rejected'}:
            continue
        sha = row.get('sha256','')
        if not re.fullmatch('[0-9a-f]{64}', sha) or sha in result:
            raise ValueError('Invalid or repeated review checksum')
        if not row.get('reviewer','').strip() or not row.get('evidence','').strip():
            raise ValueError('Review needs a named reviewer and evidence')
        labels = row.get('labels','').split(';')
        if row['status']=='accepted' and (not set(labels) <= set(CORN_LABELS) or row.get('organ') not in {'leaf','ear','stem','whole'}):
            raise ValueError('Unknown reviewed label or organ')
        result[sha] = {**row, 'labels': sorted(set(labels))}
    return result


def prepare_rows(reviews):
    inspected={}
    for split in ['train','validation','test','external_review','quarantine']:
        old=OUTPUT/f'{split}.jsonl'
        if old.exists():
            for row in read_jsonl(old):inspected[row['path']]=row
    metric_keys=['width','height','phash','pixel_sha256','gray_std','laplacian_variance','bytes','error']
    reserved_test = {r['sha256'] for r in inspected.values() if r.get('split')=='test' or r.get('protected')}
    reserved_validation = {r['sha256'] for r in inspected.values() if r.get('split')=='validation'}
    # Completed and running candidates keep immutable copies of their holdouts.
    for candidate in (ROOT/'backend/data/models/corn_candidates').glob('*'):
        for split, reserved in [('test',reserved_test),('validation',reserved_validation)]:
            snapshot = candidate/f'{split}_manifest.jsonl'
            if snapshot.exists():reserved.update(r['sha256'] for r in read_jsonl(snapshot))
    def preserve_partition(row):
        previous = inspected.get(row['path'], {})
        if row['sha256'] in reserved_test or previous.get('protected'):
            row['protected'] = True
        elif row['sha256'] in reserved_validation:
            row['previous_split'] = 'validation_candidate'
        return row
    base = ROOT/'.cache/datasets/corn_expansion/review/manifest.jsonl'
    for row in read_jsonl(base):
        row = dict(row)
        row['labels'] = ['healthy' if x=='healthy_leaf' else x for x in row['labels']]
        row['previous_split'] = row['split']
        row['protected'] = row['split']=='external_review' or 'reserved_by_existing_model_validation_or_test' in row.get('exclusion_reasons',[])
        row['review_required'] = bool(set(row['labels'])-set(CORN_LABELS))
        yield preserve_partition(row)
    csv_path=ROOT/'.cache/datasets/corn_field/ecuador/Data.csv'
    sites = {}
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            sites = {r['filename'].strip(): r for r in csv.DictReader(stream, delimiter=';') if r.get('filename','').strip()}
    for source in ['ecuador','anhui','inat_smut','inat_maize','inat_ears','inat_cobs','idadp']:
        path = REPORTS/f'{source}.json'
        if not path.exists():
            continue
        receipt = json.loads(path.read_text(encoding='utf-8'))
        for raw in receipt['records']:
            row = dict(raw)
            if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Acquired file changed')
            if source=='anhui' and row['raw_label'] in ANHUI_LABELS:
                row.update(labels=[ANHUI_LABELS[row['raw_label']]], review_required=False,
                           label_basis='publisher_species_mapping')
            # Entire Anhui source is a cross-source test, not a validation set.
            row['protected'] = source in {'anhui', 'idadp'}
            if source=='ecuador':
                site=sites.get(Path(row['path']).name)
                if not site:raise ValueError('Ecuador image has no site annotation')
                row['split_group_hint']='ecuador:'+site['location_canton']+':'+site['plantation_block']
                row['fixed_split']={'BOLIVAR':'train','TOSAGUA':'validation','CHONE':'test'}[site['location_canton']]
            if source.startswith('inat_'):
                row['split_group_hint']=row.pop('capture_group_hint')
            cached=inspected.get(row['path'])
            if cached and cached['sha256']==row['sha256']:
                row={**{k:cached[k] for k in metric_keys if k in cached},**row}
            else:
                row = inspect_image(row)
            yield preserve_partition(row)
    feedback=ROOT/'.cache/datasets/corn_field/user_feedback/pending.jsonl'
    if feedback.exists():
        for raw in read_jsonl(feedback):
            row={**raw,'source':'user_feedback','review_required':True,'protected':False,
                 'split_group_hint':'user:'+str(raw.get('capture_group_hint',raw['check_id']))}
            row.pop('capture_group_hint',None)
            yield preserve_partition(inspect_image(row))


def split_rows(rows, reviews):
    seen_reviews = set()
    for row in rows:
        review = reviews.get(row['sha256'])
        if review:
            seen_reviews.add(row['sha256'])
            row['review'] = review
            if review['status']=='accepted':
                row.update(labels=review['labels'], organ=review['organ'], review_required=False)
        row['exclusion_reasons'] = []
    if set(reviews)-seen_reviews:
        raise ValueError('Review contains images absent from this dataset version')
    duplicate_groups=assign_groups(rows)
    for members in duplicate_groups:
        fixed={r['fixed_split'] for r in members if r.get('fixed_split')}
        for row in members:
            row['cross_site_duplicate']=len(fixed)>1
    # Site/observer grouping affects partitions, not duplicate-label conflicts.
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    hints={}
    for i,row in enumerate(rows):
        for key in [('duplicate',row['duplicate_group']),('capture',row.get('split_group_hint'))]:
            if row['cross_site_duplicate'] and key[0]=='capture':continue
            if not key[1]:continue
            if key in hints:parent[root(i)]=root(hints[key])
            hints[key]=i
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[root(i)].append(row)
    for members in groups.values():
        protected = any(r.get('protected') for r in members)
        validation = any(r.get('previous_split')=='validation_candidate' for r in members)
        group_hash = min(r['sha256'] for r in members)
        bucket = int(group_hash[:8],16) % 20
        fixed={r.get('fixed_split') for r in members}
        split = 'test' if protected or 'test' in fixed else 'validation' if validation or 'validation' in fixed else 'train' if 'train' in fixed else ('test' if bucket<3 else 'validation' if bucket<6 else 'train')
        used_hashes = set()
        for row in members:
            row['partition_group']=group_hash[:24]
            reasons = row['exclusion_reasons']
            if 'error' in row or min(row.get('width',0),row.get('height',0))<96 or row.get('gray_std',0)<5:
                reasons.append('image_quality')
            if row.get('review',{}).get('status')=='rejected':
                reasons.append('review_rejected')
            if row.get('review_required') or not set(row['labels'])<=set(CORN_LABELS):
                reasons.append('taxonomy_or_organ_review_required')
            if row['organ'] not in {'leaf','ear'}:
                reasons.append('organ_not_supported_for_training')
            if len(row['labels'])!=1:
                reasons.append('multiple_labels_require_multilabel_model')
            if row['label_conflict']:
                reasons.append('conflicting_duplicate_group')
            if row['cross_site_duplicate']:
                reasons.append('duplicate_across_fixed_sites')
            if row['sha256'] in used_hashes:
                reasons.append('exact_duplicate')
            used_hashes.add(row['sha256'])
            # Existing external records stay visible for open-set evaluation, never train.
            row['split'] = split if not reasons else 'external_review' if protected else 'quarantine'
            row['protected'] = protected
    return rows


def write_review_queue(rows, path):
    fields = ['sha256','path','source','raw_label','labels','organ','reasons','reviewer','evidence','status']
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            if row['split'] not in {'quarantine','external_review'}:
                continue
            writer.writerow({**{k:row.get(k,'') for k in fields}, 'labels':';'.join(row['labels']),
                             'reasons':';'.join(row['exclusion_reasons']), 'status':'pending'})


def main(review_path=None):
    cv2.setNumThreads(2)
    reviews = read_reviews(review_path)
    rows=[]
    for row in prepare_rows(reviews):
        rows.append(row)
        if len(rows)%500==0:print(f'Inspected {len(rows)} images',flush=True)
    print('Grouping images and protecting partitions',flush=True)
    rows = split_rows(rows, reviews)
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    for split in ['train','validation','test','external_review','quarantine']:
        with (OUTPUT/f'{split}.jsonl').open('w',encoding='utf-8') as stream:
            for row in rows:
                if row['split']==split:
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_review_queue(rows,REPORTS/'review_queue.csv')
    counts = {s:dict(Counter(r['labels'][0] for r in rows if r['split']==s and len(r['labels'])==1)) for s in ['train','validation','test']}
    report = {'images':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
              'classes':counts,'sources':dict(Counter(r['source'] for r in rows)),
              'reviews_applied':len(reviews),'excluded':dict(Counter(x for r in rows for x in r['exclusion_reasons'])),
              'manifests':{s:hashlib.sha256((OUTPUT/f'{s}.jsonl').read_bytes()).hexdigest() for s in ['train','validation','test','external_review']},
              'limitations':['Ecuador split by publisher plantation/site: BOLIVAR train, TOSAGUA validation, CHONE test; not every class occurs at every site.',
                  'Original Pretoria review was already inspected; it is not a new blind test.',
                  'Community observation labels require organ/visual review and are not laboratory confirmations.',
                  'Unresolved and mixed diagnoses are excluded from single-label training.']}
    (REPORTS/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews',type=Path)
    main(parser.parse_args().reviews)
