"""Compare frozen candidates with the deployed baseline and a paired crop pilot.

Never selects a checkpoint on test or activates a model. Shared-class accuracy
uses only healthy/rust/GLS; predictions outside that vocabulary count as errors.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.diagnostics.crop_model import classify_crop, load_model
from scripts.train_corn_comparison import (crop_box, infer, make_model, normalized,
    probabilities, summarize, write_json)

DATASET = ROOT/'backend/data/datasets/corn_comparison_v1'
SHARED = {'healthy','common_rust','gray_leaf_spot'}


def prediction_summary(predictions):
    healthy = [r for r in predictions if r['actual']=='healthy']
    sick = [r for r in predictions if r['actual']!='healthy']
    shared = [r for r in predictions if r['actual'] in SHARED]
    return {'count':len(predictions), 'shared_class_count':len(shared),
        'shared_class_correct':sum(r['actual']==r['predicted'] for r in shared),
        'shared_class_accuracy':sum(r['actual']==r['predicted'] for r in shared)/len(shared) if shared else None,
        'healthy_count':len(healthy),
        'healthy_called_diseased_argmax':sum(r['predicted']!='healthy' for r in healthy),
        'healthy_called_diseased_accepted':sum(r['predicted']!='healthy' and r['confidence']>=.65 for r in healthy),
        'healthy_correct_accepted':sum(r['predicted']=='healthy' and r['confidence']>=.65 for r in healthy),
        'healthy_abstained':sum(r['confidence']<.65 for r in healthy),
        'diseased_count':len(sick),
        'diseased_called_healthy_accepted':sum(r['predicted']=='healthy' and r['confidence']>=.65 for r in sick)}


def baseline(rows, crop=False, variation=None):
    _, manifest = load_model('Кукуруза')
    if set(manifest['classes']) != {'healthy','blight','common_rust','gray_leaf_spot'}:
        raise ValueError('Deployed baseline changed; freeze a new comparison protocol')
    predictions = []
    for i, row in enumerate(rows):
        with Image.open(ROOT/row['path']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            if crop:
                image = crop_box(image,row['bbox_xyxy'])
            if variation:
                image = perturb(image,variation)
        p = classify_crop('Кукуруза',cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
        label = max(p,key=p.get)
        predictions.append({'path':row['path'],'sha256':row['sha256'],'source':row['source'],
            'actual':row['labels'][0],'predicted':label,'confidence':p[label],'probabilities':p})
        if (i+1)%300 == 0:
            print(f'Baseline {i+1}/{len(rows)}',flush=True)
    return {'model_manifest_sha256':manifest['_loaded_manifest_sha256'], 'predictions':predictions}


def perturb(image, mode):
    # Sensitivity experiment only: synthetic gray padding is not a field photo.
    image = image.copy(); image.thumbnail((480,480),Image.Resampling.LANCZOS)
    if mode == 'rotation_25':
        return image.rotate(25,expand=True,resample=Image.Resampling.BILINEAR,fillcolor=(127,127,127))
    if mode == 'scale_60':
        w,h = image.size
        canvas = Image.new('RGB',(round(w/.6),round(h/.6)),(127,127,127))
        canvas.paste(image,((canvas.width-w)//2,(canvas.height-h)//2))
        return canvas
    raise ValueError('Unknown perturbation')


def candidate_variation(model, rows, classes, temperature, mode):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            with Image.open(ROOT/row['path']) as original:
                image = perturb(ImageOps.exif_transpose(original).convert('RGB'),mode)
                image.thumbnail((320,320),Image.Resampling.LANCZOS)
                image = ImageOps.pad(image,(224,224),color=(127,127,127))
            logits = model(normalized(image)[None].to(memory_format=torch.channels_last)).numpy()
            p = probabilities(logits,temperature)[0]
            predictions.append({'path':row['path'],'sha256':row['sha256'],'source':row['source'],
                'actual':row['labels'][0],'predicted':classes[int(p.argmax())],'confidence':float(p.max())})
    return predictions


def pair_metrics(full, cropped):
    if len(full)!=len(cropped) or any(a['sha256']!=b['sha256'] or a['actual']!=b['actual'] for a,b in zip(full,cropped)):
        raise ValueError('Crop comparison needs identical ordered image/label pairs')
    # Blight is deliberately NOT mapped to either pathogen-specific class.
    shared_full = [r for r in full if r['actual'] in SHARED]
    shared_crop = [r for r in cropped if r['actual'] in SHARED]
    corrected = sum(a['predicted']!=a['actual'] and b['predicted']==b['actual'] for a,b in zip(shared_full,shared_crop))
    broken = sum(a['predicted']==a['actual'] and b['predicted']!=b['actual'] for a,b in zip(shared_full,shared_crop))
    native_corrected = sum(a['predicted']!=a['actual'] and b['predicted']==b['actual'] for a,b in zip(full,cropped))
    native_broken = sum(a['predicted']==a['actual'] and b['predicted']!=b['actual'] for a,b in zip(full,cropped))
    return {'count':len(full),'shared_count':len(shared_full),'shared_corrected':corrected,'shared_broken':broken,
        'native_corrected':native_corrected,'native_broken':native_broken,
        'changed_predictions':sum(a['predicted']!=b['predicted'] for a,b in zip(full,cropped)),
        'full':prediction_summary(full),'crop':prediction_summary(cropped)}


def load_candidate(directory):
    config = json.loads((directory/'run_config.json').read_text(encoding='utf-8'))
    report = json.loads((directory/'training_report.json').read_text(encoding='utf-8'))
    if config['protocol_sha256'] != hashlib.sha256((DATASET/'protocol.json').read_bytes()).hexdigest():
        raise ValueError('Candidate used a different data protocol')
    model = make_model(config['architecture'],report['classes'],pretrained=False)
    model.load_state_dict(torch.load(directory/'weights.pt',weights_only=True));model.eval()
    return model,report


def run(args):
    torch.set_num_threads(args.threads);cv2.setNumThreads(1)
    test = [json.loads(x) for x in (DATASET/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol = json.loads((DATASET/'protocol.json').read_text(encoding='utf-8'))
    training_sources = set(protocol['sources']['train'])
    annotations = json.loads((DATASET/'crop_annotations.json').read_text(encoding='utf-8'))
    if annotations['selection_sha256'] != hashlib.sha256((DATASET/'crop_selection.json').read_bytes()).hexdigest():
        raise ValueError('Crop selection changed after annotation')
    crop_rows = annotations['records']
    # Verify exact images for baseline and geometry, including original pixels.
    for row in test:
        if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Evaluation photo changed')
    baseline_path = DATASET/'baseline_evaluation.json'
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text(encoding='utf-8'))
        _, manifest = load_model('Кукуруза')
        if base['model_manifest_sha256'] != manifest['_loaded_manifest_sha256'] or base['test_sha256'] != hashlib.sha256((DATASET/'test.jsonl').read_bytes()).hexdigest():
            raise ValueError('Stale baseline cache')
    else:
        base = baseline(test)
        base['test_sha256'] = hashlib.sha256((DATASET/'test.jsonl').read_bytes()).hexdigest()
        write_json(baseline_path,base)
    if args.baseline_only:
        print(json.dumps(prediction_summary(base['predictions'])),flush=True)
        return
    predictions = {'deployed_mobilenet_v2':base['predictions']}
    geometry = {}
    base_by_sha = {r['sha256']:r for r in base['predictions']}
    full = [base_by_sha[r['sha256']] for r in crop_rows]
    cropped = baseline(crop_rows,crop=True)['predictions']
    geometry['deployed_mobilenet_v2'] = {'paired':pair_metrics(full,cropped),
        'full_predictions':full,'crop_predictions':cropped,
        'by_source':{s:pair_metrics([r for r in full if r['source']==s],[r for r in cropped if r['source']==s])
                     for s in sorted({r['source'] for r in crop_rows})},
        'sensitivity':{mode:pair_metrics(full,baseline(crop_rows,variation=mode)['predictions'])
                       for mode in ['rotation_25','scale_60']}}
    candidates = {}
    user_spotcheck = {}
    feedback_path = ROOT/'.cache/datasets/corn_field/user_feedback/pending.jsonl'
    feedback = [json.loads(x) for x in feedback_path.read_text(encoding='utf-8').splitlines()
                if x.strip()] if feedback_path.exists() else []
    feedback = [r for r in feedback if r['organ']=='leaf']
    training_hashes = {json.loads(x)['sha256'] for x in (DATASET/'train.jsonl').read_text(encoding='utf-8').splitlines()}
    if any(r['sha256'] in training_hashes for r in feedback):
        raise ValueError('User spot-check photo found in training')
    for directory in args.candidates:
        model, report = load_candidate(directory)
        name, classes, temperature = report['architecture'], report['classes'], report['temperature']
        if name in candidates:
            raise ValueError('One frozen candidate per architecture is required')
        candidates[name] = {'directory':directory.relative_to(ROOT).as_posix(),
            'validation_macro_f1':report['validation']['macro_f1'],
            'test_macro_f1':report['test']['macro_f1'],
            'weights_sha256':hashlib.sha256((directory/'weights.pt').read_bytes()).hexdigest()}
        predictions[name] = report['test']['predictions']
        by_sha = {r['sha256']:r for r in predictions[name]}
        full = [by_sha[r['sha256']] for r in crop_rows]
        logits,_ = infer(model,crop_rows,classes,args.batch,crop=True)
        cropped = summarize(crop_rows,logits,classes,temperature)['predictions']
        geometry[name] = {'paired':pair_metrics(full,cropped), 'full_predictions':full,'crop_predictions':cropped,
            'by_source':{s:pair_metrics([r for r in full if r['source']==s],[r for r in cropped if r['source']==s])
                         for s in sorted({r['source'] for r in crop_rows})},
            'sensitivity':{mode:pair_metrics(full,candidate_variation(model,crop_rows,classes,temperature,mode))
                           for mode in ['rotation_25','scale_60']}}
        print(json.dumps({'architecture':name,'crop':geometry[name]['paired']}),flush=True)
        if feedback:
            # Dummy target serves only the Dataset loader. User labels are kept
            # separately as unverified proposals, never scored as ground truth.
            items = [{**r,'labels':[classes[0]]} for r in feedback]
            user_logits,_ = infer(model,items,classes,args.batch)
            p = probabilities(user_logits,temperature)
            user_spotcheck[name] = [{'check_id':r['check_id'],'sha256':r['sha256'],
                'user_proposed_labels':r['labels'],'verified':False,
                'predicted':classes[int(s.argmax())],'confidence':float(s.max()),
                'accepted_at_existing_threshold':bool(s.max()>=.65)} for r,s in zip(feedback,p)]
        del model
    preferred = max(candidates,key=lambda name:candidates[name]['validation_macro_f1'])
    chosen = geometry[preferred]; paired = chosen['paired']; gate = annotations['detector_gate']
    detector_eligible = (
        paired['native_corrected']-paired['native_broken'] >= gate['minimum_net_corrected']
        and paired['full']['healthy_called_diseased_accepted']-paired['crop']['healthy_called_diseased_accepted'] >= gate['minimum_healthy_false_positive_reduction']
        and paired['crop']['diseased_called_healthy_accepted']-paired['full']['diseased_called_healthy_accepted'] <= gate['maximum_additional_false_healthy']
        and sum(m['native_corrected']>m['native_broken'] for m in chosen['by_source'].values()) >= gate['minimum_improved_sources'])
    summary = {'models':{},'candidates':candidates,'preferred_by_validation':preferred,
        'unseen_sources':sorted({r['source'] for r in test}-training_sources),
        'activated':False,'crop_annotations_sha256':hashlib.sha256((DATASET/'crop_annotations.json').read_bytes()).hexdigest(),
        'detector_pilot':{'eligible_for_training':bool(detector_eligible),'rule':gate,
                          'evaluated_candidate':preferred,'paired':paired},
        'limitations':['Shared-class accuracy covers healthy, common rust and gray leaf spot only.',
            'External metrics exclude legacy_kaggle; Ecuador/TARI/TOM are image/group holdouts within candidate training sources.',
            'Unseen-source metrics exclude every source present in candidate training. Old-source tests have previously been inspected.',
            'Source labels are not independently confirmed diagnoses.',
            'Crop experiment uses approximate manually reviewed boxes, not automatic detections.',
            'Rotation/scale sensitivity is synthetic and not a field-accuracy estimate.',
            'No model was selected using test performance or activated.']}
    for name, rows in predictions.items():
        if [r['sha256'] for r in rows] != [r['sha256'] for r in test]:
            raise ValueError('Models evaluated different test photographs or ordering')
        summary['models'][name] = {'all':prediction_summary(rows),
            'external':prediction_summary([r for r in rows if r['source']!='legacy_kaggle']),
            'unseen_sources':prediction_summary([r for r in rows if r['source'] not in training_sources]),
            'by_source':{s:prediction_summary([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})}}
    write_json(DATASET/'crop_comparison.json',geometry)
    write_json(DATASET/'comparison.json',summary)
    write_json(DATASET/'user_spotcheck.json',{'purpose':'Previously inspected user photos; not training, selection or an accuracy estimate.',
                                            'models':user_spotcheck})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,nargs='*',default=[])
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--batch',type=int,default=16)
    args = parser.parse_args()
    args.candidates = [p.resolve() for p in args.candidates]
    if not args.baseline_only and len(args.candidates)!=2:
        parser.error('Pass exactly two frozen candidates, or --baseline-only')
    run(args)
