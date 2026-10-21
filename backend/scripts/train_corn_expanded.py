"""Train a seven-class corn candidate from audited, immutable split manifests.

Frozen ImageNet features; training-only augmentation and class/source balancing.
C and temperature use validation only. External images are evaluated after fitting.
This script writes a candidate directory and never overwrites the active model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.diagnostics.tuber_model import extract_features, softmax

LABELS = {'common_rust': 'Обыкновенная ржавчина кукурузы',
          'curvularia_leaf_spot': 'Курвуляриозная пятнистость кукурузы',
          'gray_leaf_spot': 'Серая пятнистость листьев кукурузы',
          'healthy': 'Здоровый лист',
          'maize_lethal_necrosis': 'Летальный некроз кукурузы',
          'maize_streak_virus': 'Вирус полосатости кукурузы (Maize streak virus)',
          'northern_leaf_blight': 'Северный гельминтоспориоз кукурузы'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def canonical(label):
    return 'healthy' if label == 'healthy_leaf' else label


def validate_splits(train, validation, external):
    for rows in [train, validation]:
        for row in rows:
            if row['source'] not in {'legacy_kaggle', 'tari', 'tom2024'}:
                raise ValueError('Unapproved training source')
            if len(row['labels']) != 1 or canonical(row['labels'][0]) not in LABELS:
                raise ValueError('Unsupported or ambiguous training label')
            if row.get('label_conflict') or row.get('overlaps_external') or row.get('exclusion_reasons'):
                raise ValueError('Quarantined record in training/validation')
        if {canonical(r['labels'][0]) for r in rows} != set(LABELS):
            raise ValueError('Every supported class must occur in train and validation')
    for field in ['sha256', 'duplicate_group']:
        groups = [{r[field] for r in rows} for rows in [train, validation, external]]
        if any(groups[i] & groups[j] for i,j in [(0,1),(0,2),(1,2)]):
            raise ValueError(f'Split leakage by {field}')


def augmented(image, seed):
    rng = np.random.default_rng(seed)
    h,w = image.shape[:2]
    scale = min(1., 768/max(h,w))
    image = cv2.resize(image, (round(w*scale), round(h*scale)))
    h,w = image.shape[:2]
    fraction = rng.uniform(.82, 1.)
    ch,cw = max(1, round(h*fraction)), max(1, round(w*fraction))
    top,left = rng.integers(0,h-ch+1), rng.integers(0,w-cw+1)
    image = image[top:top+ch,left:left+cw]
    if rng.random() < .5:
        image = cv2.flip(image, 1)
    image = np.rot90(image, int(rng.integers(0,4))).copy()
    return np.clip(image.astype(np.float32)*rng.uniform(.85,1.15)+rng.uniform(-8,8),0,255).astype(np.uint8)


def features(rows, net, backbone_sha, views, feature_cache):
    xs, ys, expanded = [], [], []
    for index,row in enumerate(rows,1):
        path = (ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != row['sha256']:
            raise ValueError(f'Image changed or outside project: {row["path"]}')
        key = f'{row["sha256"]}_{backbone_sha[:12]}_aug1_v{views}.npz'
        cached = feature_cache/key
        if cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                vectors = data['features']
            if vectors.shape != (views,1280) or not np.isfinite(vectors).all():
                raise ValueError('Invalid cached features')
        else:
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Unreadable image: {path}')
            vectors = [extract_features(net,image)]
            for view in range(1,views):
                seed = int(row['sha256'][:8],16)+view
                vectors.append(extract_features(net,augmented(image,seed)))
            vectors = np.array(vectors)
            np.savez_compressed(cached, features=vectors)
        xs.extend(vectors)
        ys.extend([canonical(row['labels'][0])]*views)
        expanded.extend([row]*views)
        if index % 100 == 0:
            print(f'Features: {index}/{len(rows)}, views={views}',flush=True)
    return np.array(xs), np.array(ys), expanded


def class_source_weights(rows):
    counts = Counter((canonical(r['labels'][0]),r['source']) for r in rows)
    sources = Counter(label for label,source in counts)
    weights = np.array([1/(counts[(canonical(r['labels'][0]),r['source'])]*sources[canonical(r['labels'][0])]) for r in rows])
    return weights/weights.mean()


def main(output, views):
    active = ROOT/'backend/data/models/corn'
    if output.resolve() == active.resolve():
        raise ValueError('Train to a candidate directory; review before activation')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; preserve previous candidate results')
    cv2.setNumThreads(2)
    review = ROOT/'.cache/datasets/corn_expansion/review'
    train = read_rows(review/'train_candidate.jsonl')
    validation = read_rows(review/'validation_candidate.jsonl')
    external = read_rows(review/'external_review.jsonl')
    validate_splits(train,validation,external)
    backbone = active/'mobilenetv2_features.onnx'
    backbone_sha = sha256(backbone)
    net = cv2.dnn.readNetFromONNX(str(backbone))
    cache = ROOT/'.cache/features/corn_expanded'
    cache.mkdir(parents=True, exist_ok=True)
    xt,yt,expanded = features(train,net,backbone_sha,views,cache)
    xv,yv,_ = features(validation,net,backbone_sha,1,cache)
    weights = class_source_weights(expanded)
    scaler = StandardScaler().fit(xt, sample_weight=weights)
    xt,xv = scaler.transform(xt), scaler.transform(xv)
    fits = []
    for c in [.01,.1,1.]:
        head = LogisticRegression(C=c,max_iter=2000,random_state=42)
        with threadpool_limits(limits=4):
            head.fit(xt,yt,sample_weight=weights)
        scores = head.predict_proba(xv)
        predicted = head.classes_[scores.argmax(axis=1)]
        metrics = {'C':c, 'macro_f1':f1_score(yv,predicted,average='macro'),
                   'accuracy':accuracy_score(yv,predicted),'log_loss':log_loss(yv,scores,labels=head.classes_)}
        print('Validation: '+json.dumps(metrics),flush=True)
        fits.append((metrics,head))
    selected,head = max(fits,key=lambda pair:(pair[0]['macro_f1'],-pair[0]['log_loss']))
    classes = head.classes_.tolist()
    logits = head.decision_function(xv)
    fit = minimize_scalar(lambda t:log_loss(yv,softmax(logits/t),labels=classes),bounds=(.5,10.),method='bounded')
    temperature = float(fit.x)
    probabilities = softmax(logits/temperature)
    predicted = head.classes_[probabilities.argmax(axis=1)]
    val_metrics = {'accuracy':accuracy_score(yv,predicted), 'balanced_accuracy':balanced_accuracy_score(yv,predicted),
                   'macro_f1':f1_score(yv,predicted,average='macro'), 'log_loss':log_loss(yv,probabilities,labels=classes),
                   'confusion_matrix':confusion_matrix(yv,predicted,labels=classes).tolist(),
                   'classification_report':classification_report(yv,predicted,labels=classes,output_dict=True,zero_division=0)}
    print('Selected model fixed; starting external evaluation.',flush=True)
    external_results = []
    for row in external:
        x,_,_ = features([row],net,backbone_sha,1,cache)
        scores = softmax(head.decision_function(scaler.transform(x))[0]/temperature)
        key = classes[int(scores.argmax())]
        external_results.append({'file':row['path'], 'sha256':row['sha256'], 'group':row['duplicate_group'],
            'source':row['source'], 'actual_labels':[canonical(x) for x in row['labels']],
            'predicted':key, 'confidence':float(scores.max()), 'probabilities':dict(zip(classes,map(float,scores)))})
    # Same 24 images as the legacy benchmark, plus coverage of new supported classes.
    field = [r for r in external_results if r['source']=='pretoria']
    common = [r for r in field if len(r['actual_labels'])==1 and r['actual_labels'][0] in {'common_rust','gray_leaf_spot','healthy'}]
    supported = [r for r in field if len(r['actual_labels'])==1 and r['actual_labels'][0] in classes]
    def metrics(rows):
        accepted = [r for r in rows if r['confidence']>=.65]
        return {'count':len(rows), 'correct':sum(r['predicted']==r['actual_labels'][0] for r in rows),
                'accepted_at_65':len(accepted),'correct_accepted':sum(r['predicted']==r['actual_labels'][0] for r in accepted)}
    external_metrics = {'common_three_classes':metrics(common),'all_supported_single_labels':metrics(supported),
        'raw_healthy_on_nonhealthy_at_65':sum('healthy' not in r['actual_labels'] and r['predicted']=='healthy' and r['confidence']>=.65 for r in field)}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'classifier.npz',mean=scaler.mean_,scale=scaler.scale_,coef=head.coef_,
                        intercept=head.intercept_,temperature=np.array(temperature))
    for name in ['mobilenetv2_features.onnx','BACKBONE_LICENSE.txt','BACKBONE_README.md']:
        shutil.copyfile(active/name,output/name)
    def split_records(rows):
        return [{'file':r['path'],'sha256':r['sha256'],'group':r['duplicate_group'],
                 'class':canonical(r['labels'][0]),'source':r['source']} for r in rows]
    report = {'schema_version':2, 'path_basis':'project_root','seed':42,
        'method':'Frozen ImageNet MobileNetV2 + source/class-balanced LogisticRegression; train-only augmentation',
        'augmentation_views':views, 'augmentation':'Original + deterministic crop/flip/90-degree rotations/brightness; train only',
        'backbone_finetuned':False,'classes':classes, 'counts':dict(Counter(yt[::views])),
        'selection':'Validation macro-F1; validation log-loss tie-break. External not used to select C or temperature.',
        'validation_candidates':[m for m,h in fits], 'selected_C':selected['C'],
        'temperature':temperature, 'temperature_fit':'validation only', 'validation_metrics':val_metrics,
        'split_counts':{'train':len(train),'validation':len(validation),'test':len(supported)},
        'splits':{'train':split_records(train),'validation':split_records(validation),
                  'test':[{'file':r['file'],'sha256':r['sha256'],'group':r['group'],'class':r['actual_labels'][0],'source':r['source']} for r in supported]},
        'test_predictions':[{'file':r['file'],'actual':r['actual_labels'][0],'predicted':r['predicted'],'confidence':r['confidence']} for r in supported],
        'external_metrics':external_metrics,
        'input_manifests_sha256':{n:sha256(review/n) for n in ['train_candidate.jsonl','validation_candidate.jsonl','external_review.jsonl']},
        'limitations':['Small stratified external sample; already reviewed for legacy model, not a new blind field trial.',
            'Unknown diseases not represented by a trained rejection class.', 'Southern leaf blight and common smut unsupported.',
            'Class/source background confounding remains possible. Healthy is not proof of disease absence.']}
    (output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'external_review.json').write_text(json.dumps(external_results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest = {'classes':classes,'labels':LABELS,'supported_organ':'leaf',
        'input':'RGB 224x224 letterbox, ImageNet mean/std; frozen 1280-dimensional features',
        'files':{name:sha256(output/name) for name in ['mobilenetv2_features.onnx','classifier.npz']},
        'training':report['method'],'limitations':report['limitations'],
        'dataset':'Audited corn expansion: Kaggle original + TARI + TOM2024',
        'backbone':'https://huggingface.co/onnxmodelzoo/mobilenetv2-12'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(ROOT/'backend/data/datasets/corn_expansion/sources.json',output/'dataset_provenance.json')
    print(json.dumps({'output':str(output),'validation':selected,'external':external_metrics,'temperature':temperature},indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'.cache/models/corn_expanded_v1')
    parser.add_argument('--views',type=int,default=3)
    args = parser.parse_args()
    if args.views<1 or args.views>5:
        parser.error('--views must be 1..5')
    main(args.output,args.views)
