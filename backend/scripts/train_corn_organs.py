"""Train a coarse corn photo router; disease labels never define organ health.

Uses existing licensed images, reviewed scene labels and group-preserving splits.
The user photos are never used to fit the model or its calibration.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, log_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.diagnostics.tuber_model import image_blob, softmax
from scripts.audit_corn_sources import inspect_image, assign_groups

# Visual review of the existing maize_extra_0..3 contact sheets (index 0..299).
# These are organ/scene annotations only; they assert no disease or health.
EARS = {0,1,2,7,15,35,58,60,71,84,89,91,93,101,102,119,125,126,130,133,
        138,145,146,157,171,172,175,234,247,248,253,254,274,275,277,281,284,287,293}
OTHER = {4,6,74,75,86,103,105,116,118,128,129,137,140,144,160,169,176,199,249,282}
SKIP = {80,87,96,113,120,131,143,155,166,168,170,179,233,237,251,267,276}
BARE_FIELDS = set(range(180,230)) - {193,225}
WHOLE = set(range(300)) - EARS - OTHER - SKIP - BARE_FIELDS - {255}


def load_rows():
    rows = []
    for split in ['train','validation','test','external_review','quarantine']:
        rows.extend(json.loads(line) for line in (ROOT/f'.cache/datasets/corn_field/manifests/{split}.jsonl').read_text(encoding='utf-8').splitlines())
    by_path = {r['path']: r for r in rows}
    selected = []
    # Existing validated organ labels; unclear disease labels can still depict an ear.
    for row in rows:
        if row['source'] == 'user_feedback':
            continue
        if row['split'] in {'train','validation','test'} and row['organ'] in {'leaf','ear'}:
            selected.append({**row, 'organ_label': row['organ'], 'organ_basis': 'publisher_scope_or_existing_organ_review'})
        elif row['source'] == 'idadp' and row['organ'] == 'ear':
            selected.append({**row, 'organ_label':'ear', 'organ_basis':'publisher_ear_category; sampled_visual_organ_review; disease_unresolved', 'split':'test'})
    index = json.loads((ROOT/'.cache/datasets/corn_field/review_sheets/maize_extra_index.json').read_text(encoding='utf-8'))
    reviews = []
    for i, raw in enumerate(index):
        label = 'whole' if i in WHOLE else 'ear' if i in EARS else 'other' if i in OTHER else 'leaf' if i == 255 else None
        if label is None:
            continue
        row = by_path[raw['path']]
        bucket = int(row['partition_group'][:8],16) % 20
        split = row['split'] if row['split'] in {'train','validation','test'} else (
            'test' if row.get('protected') or bucket < 3 else 'validation' if bucket < 6 else 'train')
        selected.append({**row, 'split':split, 'organ_label':label, 'organ_basis':'Codex visual organ triage; not a disease annotation'})
        reviews.append({'sha256':row['sha256'],'path':row['path'],'organ':label,'sheet_index':i,
                        'reviewer':'Codex visual organ triage','purpose':'Organ only; no disease/health diagnosis'})
    # Negative examples from an existing licensed dataset: preserve its split groups.
    report = json.loads((ROOT/'backend/data/models/sunflower/training_report.json').read_text(encoding='utf-8'))
    for split, members in report['splits'].items():
        for raw in sorted(members,key=lambda r:r['sha256'])[:{'train':180,'validation':45,'test':60}[split]]:
            path = ROOT/'.cache/datasets/sunflower/images/Original Image'/raw['file']
            row = inspect_image({'path':path.relative_to(ROOT).as_posix(),'source':'sunflower_negative',
                'sha256':raw['sha256'],'labels':['other'],'organ':'other','organ_label':'other',
                'organ_basis':'different crop; negative for corn routing', 'split':split,
                'partition_group':f"sunflower:{raw['group']}"})
            selected.append(row)
    unique = {}
    for row in selected:
        if row.get('error') or min(row['width'],row['height']) < 96:
            continue
        key = row['sha256']
        if key in unique and unique[key]['organ_label'] != row['organ_label'] and not row['organ_basis'].startswith('Codex visual organ triage'):
            raise ValueError('Conflicting organ review')
        unique[key] = row
    selected = list(unique.values())
    # Recheck visual duplicates across sources and preserve the most restrictive fold.
    for row in selected:
        row['labels'] = [row['organ_label']]
    groups = assign_groups(selected)
    parents = list(range(len(selected)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(selected):
        for field in ['duplicate_group','partition_group']:
            key = (field, row[field])
            if key in seen: parents[root(i)] = root(seen[key])
            seen[key] = i
    joined = defaultdict(list)
    for i,row in enumerate(selected): joined[root(i)].append(row)
    for members in joined.values():
        folds = {r['split'] for r in members}
        split = 'test' if 'test' in folds else 'validation' if 'validation' in folds else 'train'
        group = min(r['sha256'] for r in members)
        for row in members: row.update(split=split, organ_group=group)
    selected = [r for r in selected if not r['label_conflict']]
    # Deterministic cap keeps the router small; limits repeated views per group.
    capped = []; counts = Counter(); group_counts = Counter()
    for row in sorted(selected,key=lambda r:r['sha256']):
        key = (row['split'],row['organ_label']); g = (row['organ_group'],row['organ_label'])
        cap = {'train':300,'validation':100,'test':220}[row['split']]
        group_cap = 120 if row['source'] in {'idadp','ecuador'} else 12
        if counts[key] < cap and group_counts[g] < group_cap:
            capped.append(row); counts[key] += 1; group_counts[g] += 1
    return capped, reviews


def train(output):
    if output.exists(): raise ValueError('Choose a new output directory')
    cv2.setNumThreads(2)
    print('Preparing reviewed organ labels and duplicate groups',flush=True)
    fixed = ROOT/'backend/data/datasets/corn_organs/dataset.jsonl'
    if fixed.exists():
        rows = [json.loads(line) for line in fixed.read_text(encoding='utf-8').splitlines()]
        reviews = []
    else:
        rows, reviews = load_rows()
    classes = ['ear','leaf','other','whole']
    counts = {s:dict(Counter(r['organ_label'] for r in rows if r['split']==s)) for s in ['train','validation','test']}
    print(json.dumps(counts),flush=True)
    for split in counts:
        if any(counts[split].get(label,0) < 15 for label in classes):
            raise ValueError('Insufficient independent groups for an organ class')
    output.mkdir(parents=True)
    (output/'organ_reviews.json').write_text(json.dumps(reviews,indent=2),encoding='utf-8')
    (output/'dataset.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    import torch
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    torch.set_num_threads(2)
    pretrained = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
    extractor = torch.nn.Sequential(pretrained.features, pretrained.avgpool, torch.nn.Flatten()).eval()
    backbone = output/'features.onnx'
    torch.onnx.export(extractor, torch.zeros(1,3,224,224), str(backbone),
                      opset_version=17, dynamo=False, input_names=['image'], output_names=['features'])
    backbone_sha = hashlib.sha256(backbone.read_bytes()).hexdigest()
    net = cv2.dnn.readNetFromONNX(str(backbone))
    cache = ROOT/'.cache/features/corn_organs'/f'{backbone_sha}-trim-square224-v1'; cache.mkdir(parents=True,exist_ok=True)
    x = []
    for i,row in enumerate(rows):
        payload = (ROOT/row['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']: raise ValueError('Image changed')
        saved = cache/(row['sha256']+'.npy')
        if saved.exists(): features = np.load(saved,allow_pickle=False)
        else:
            image = cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
            blob = image_blob(image)
            net.setInput(blob)
            features = net.forward().reshape(-1)
            if features.shape != (960,) or not np.isfinite(features).all():
                raise ValueError('Invalid MobileNetV3 features')
            if i == 0:
                with torch.no_grad(): expected = extractor(torch.from_numpy(blob)).numpy().reshape(-1)
                if not np.allclose(features, expected, atol=1e-4, rtol=1e-4):
                    raise ValueError('ONNX differs from training backbone')
            np.save(saved,features)
        x.append(features)
        if (i+1)%100 == 0: print('features',i+1,flush=True)
    x = np.asarray(x); y = np.asarray([classes.index(r['organ_label']) for r in rows])
    masks = {s:np.asarray([r['split']==s for r in rows]) for s in counts}
    scaler = StandardScaler().fit(x[masks['train']])
    model = LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',random_state=42)
    model.fit(scaler.transform(x[masks['train']]),y[masks['train']])
    val_logits = model.decision_function(scaler.transform(x[masks['validation']]))
    temperature = float(minimize_scalar(lambda t:log_loss(y[masks['validation']],softmax(val_logits/t),labels=range(4)),bounds=(.5,10),method='bounded').x)
    # The threshold is fixed in advance; test cannot tune it or choose checkpoints.
    threshold = .8
    probabilities = softmax(model.decision_function(scaler.transform(x[masks['test']]))/temperature)
    actual = y[masks['test']]; predicted = probabilities.argmax(axis=1); accepted = probabilities.max(axis=1)>=threshold
    metrics = classification_report(actual,predicted,labels=range(4),target_names=classes,output_dict=True,zero_division=0)
    accepted_metrics = {}
    for i,label in enumerate(classes):
        positive = accepted & (predicted==i)
        tp = int(((actual==i)&positive).sum())
        accepted_metrics[label] = {'support':int((actual==i).sum()),'accepted':int(positive.sum()),
            'precision':tp/int(positive.sum()) if positive.any() else 0.,'recall':tp/int((actual==i).sum())}
    eligible = all(v['precision']>=.9 and v['recall']>=.6 and v['support']>=15 for v in accepted_metrics.values())
    np.savez_compressed(output/'classifier.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,intercept=model.intercept_,temperature=np.asarray(temperature))
    manifest = {'format':'corn_organ_features_v2','architecture':'mobilenet_v3_large',
        'feature_size':960,'backbone_file':'features.onnx',
        'classes':classes,'threshold':threshold,'temperature':temperature,
        'backbone_sha256':backbone_sha,'classifier_sha256':hashlib.sha256((output/'classifier.npz').read_bytes()).hexdigest(),
        'release_status':'approved' if eligible else 'candidate',
        'limitations':'Coarse corn organ routing; whole includes general plant/field views. Not a crop, disease or health diagnosis. Mixed and unknown objects may be misclassified.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report = {'counts':counts,'threshold':threshold,'temperature':temperature,'classification_report':metrics,
        'accepted_metrics':accepted_metrics,'confusion_matrix':confusion_matrix(actual,predicted,labels=range(4)).tolist(),
        'eligible':eligible,'rule':'At fixed 0.8 score: each class precision >=0.9, recall >=0.6, test support >=15.',
        'predictions':[{'path':r['path'],'source':r['source'],'actual':r['organ_label'],'predicted':classes[int(p.argmax())],'score':float(p.max())} for r,p in zip([r for r in rows if r['split']=='test'],probabilities)]}
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'eligible':eligible,'accepted_metrics':accepted_metrics}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    train(parser.parse_args().output)
