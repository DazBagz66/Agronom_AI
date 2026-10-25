"""Compare frozen and partially fine-tuned MobileNetV3 on protected manifests.

Validation selects the checkpoint and temperature. Test is evaluated once after
selection. Outputs a separate ONNX candidate; does not replace the active model.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.corn_taxonomy import CORN_LABELS
from scripts.build_corn_field import OUTPUT, REPORTS


class Images(Dataset):
    def __init__(self,rows,classes,train=False):
        self.rows,self.classes=rows,classes
        extra=[transforms.RandomHorizontalFlip(),transforms.RandomRotation(20),
               transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.1)] if train else []
        self.transform=transforms.Compose([*extra,transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(ROOT/row['path']) as original:
            image=ImageOps.pad(ImageOps.exif_transpose(original).convert('RGB'),(224,224),color=(127,127,127))
        return self.transform(image),self.classes.index(row['labels'][0])


def evaluate(model,loader):
    model.eval();scores=[];actual=[]
    with torch.inference_mode():
        for images,labels in loader:
            scores.append(model(images).numpy());actual.extend(labels.tolist())
    return np.concatenate(scores),np.array(actual)


def probabilities(logits,temperature=1.):
    a=logits/temperature;a=a-a.max(axis=1,keepdims=True)
    p=np.exp(a);return p/p.sum(axis=1,keepdims=True)


def metrics(actual,p,classes):
    prediction=p.argmax(axis=1)
    return {'count':len(actual),'accuracy':float(accuracy_score(actual,prediction)),
        'macro_f1':float(f1_score(actual,prediction,labels=list(range(len(classes))),average='macro',zero_division=0)),
        'classification_report':classification_report(actual,prediction,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(actual,prediction,labels=list(range(len(classes)))).tolist()}


def verify_splits(splits):
    keys={}
    for name,rows in splits.items():
        for row in rows:
            path=(ROOT/row['path']).resolve()
            if not path.is_relative_to(ROOT.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Unsafe or changed training image')
            if row.get('exclusion_reasons') or row.get('review_required') or (name=='train' and row.get('protected')):
                raise ValueError('Quarantined/protected image in training manifests')
            for field in ['sha256','duplicate_group','partition_group']:
                key=(field,row[field])
                if key in keys and keys[key]!=name:raise ValueError('Split leakage')
                keys[key]=name


def main(output,organ,epochs,batch,threads,min_train):
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new candidate directory')
    torch.set_num_threads(threads);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    audit=json.loads((REPORTS/'audit.json').read_text(encoding='utf-8'))
    raw_splits={s:(OUTPUT/f'{s}.jsonl').read_bytes() for s in ['train','validation','test']}
    manifest_hashes={s:hashlib.sha256(raw).hexdigest() for s,raw in raw_splits.items()}
    if any(audit['manifests'][s]!=digest for s,digest in manifest_hashes.items()):
        raise ValueError('Dataset is being rebuilt or differs from its audit; retry after a complete build')
    all_splits={s:[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()] for s,raw in raw_splits.items()}
    splits={s:[r for r in rows if r['organ']==organ and len(r['labels'])==1] for s,rows in all_splits.items()}
    counts={s:Counter(r['labels'][0] for r in rows) for s,rows in splits.items()}
    targets={'common_rust','gray_leaf_spot','healthy','curvularia_leaf_spot','maize_lethal_necrosis',
             'maize_streak_virus','northern_leaf_blight','southern_leaf_blight'} if organ=='leaf' else {
             'common_smut','healthy_ear','head_smut','ear_rot','fusarium_ear_rot','gibberella_ear_rot','aspergillus_ear_rot'}
    classes=sorted(k for k,n in counts['train'].items() if k in targets and n>=min_train and counts['validation'][k]>=5)
    if len(classes)<2:raise ValueError('Need at least two adequately represented classes for this organ')
    splits={s:[r for r in rows if r['labels'][0] in classes] for s,rows in splits.items()}
    if not splits['test']:raise ValueError('Independent test manifest is empty')
    verify_splits(splits)
    output.mkdir(parents=True)
    for s,rows in splits.items():
        (output/f'{s}_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    train=Images(splits['train'],classes,True)
    class_source=Counter((r['labels'][0],r['source']) for r in splits['train'])
    source_counts=Counter(label for label,source in class_source)
    weights=[1/(class_source[(r['labels'][0],r['source'])]*source_counts[r['labels'][0]]) for r in splits['train']]
    loader=DataLoader(train,batch_size=batch,sampler=WeightedRandomSampler(weights,len(train),replacement=True),num_workers=0)
    validation=DataLoader(Images(splits['validation'],classes),batch_size=batch,num_workers=0)
    test=DataLoader(Images(splits['test'],classes),batch_size=batch,num_workers=0)
    model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(classes))
    for parameter in model.features.parameters():parameter.requires_grad=False
    criterion=nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=.001,weight_decay=.001)
    best=None;best_score=-1.;history=[]
    for epoch in range(epochs):
        if epoch==2:
            for parameter in model.features[-4:].parameters():parameter.requires_grad=True
            optimizer=torch.optim.AdamW([
                {'params':model.features[-4:].parameters(),'lr':.0001},
                {'params':model.classifier.parameters(),'lr':.0003}],weight_decay=.001)
        model.train()
        # Keep pretrained batch-normalization statistics stable on small datasets.
        for layer in model.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        start=time.monotonic();loss_sum=0.
        for images,labels in loader:
            optimizer.zero_grad();loss=criterion(model(images),labels);loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(labels)
        logits,actual=evaluate(model,validation);m=metrics(actual,probabilities(logits),classes)
        row={'epoch':epoch+1,'stage':'frozen' if epoch<2 else 'finetune','loss':loss_sum/len(train),
             'validation_macro_f1':m['macro_f1'],'validation_accuracy':m['accuracy'],'seconds':time.monotonic()-start}
        history.append(row);print(json.dumps(row),flush=True)
        (output/'progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if m['macro_f1']>best_score:
            best_score=m['macro_f1'];best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(best);model.eval()
    val_logits,val_actual=evaluate(model,validation)
    temperature=float(minimize_scalar(lambda t:log_loss(val_actual,probabilities(val_logits,t),labels=list(range(len(classes)))),bounds=(.5,10),method='bounded').x)
    print('Checkpoint selected on validation; evaluating untouched test.',flush=True)
    logits,actual=evaluate(model,test);p=probabilities(logits,temperature)
    result=metrics(actual,p,classes)
    by_source={}
    for source in sorted({r['source'] for r in splits['test']}):
        indexes=[i for i,r in enumerate(splits['test']) if r['source']==source]
        by_source[source]=metrics(actual[indexes],p[indexes],classes)
    # A conservative readiness report, not a guarantee of real-world reliability.
    failures=[]
    for label in classes:
        scores=result['classification_report'][label]
        if scores['support']<15 or min(scores['precision'],scores['recall'])<.7:
            failures.append(label)
    readiness={'eligible':not failures and result['macro_f1']>=.75,
               'failed_classes':failures,'rule':'Each class: >=15 test images, precision/recall >=0.70; macro-F1 >=0.75. Further independent field review required.'}
    torch.save(best,output/'weights.pt')
    torch.onnx.export(model,torch.zeros(1,3,224,224),str(output/'model.onnx'),opset_version=17,
                      input_names=['images'],output_names=['logits'],dynamo=False)
    import cv2
    cv2.setNumThreads(2)
    exported=cv2.dnn.readNetFromONNX(str(output/'model.onnx'))
    sample=Images(splits['validation'],classes)[0][0].unsqueeze(0)
    exported.setInput(sample.numpy())
    with torch.inference_mode():reference=model(sample).numpy()
    difference=float(np.max(np.abs(exported.forward()-reference)))
    if difference>0.001:raise ValueError(f'ONNX/runtime parity failed: {difference}')
    manifest={'format':'onnx_logits_v1','classes':classes,'labels':{k:CORN_LABELS[k] for k in classes},
        'supported_organ':organ,'temperature':temperature,'input':'RGB 224x224 gray letterbox; ImageNet normalization',
        'files':{'model.onnx':hashlib.sha256((output/'model.onnx').read_bytes()).hexdigest()},
        'readiness':readiness,'onnx_max_abs_error':difference,'release_status':'candidate'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'seed':42,'classes':classes,'organ':organ,'minimum_training_examples':min_train,'history':history,'selected_epoch':best_epoch,
        'backbone_finetuned':best_epoch>2,'temperature':temperature,'selection':'validation macro-F1 only',
        'counts':{s:dict(Counter(r['labels'][0] for r in rows)) for s,rows in splits.items()},
        'manifest_hashes':manifest_hashes,
        'validation':metrics(val_actual,probabilities(val_logits,temperature),classes),
        'test':result,'test_by_source':by_source,'readiness':readiness,
        'predictions':[{'path':r['path'],'source':r['source'],'actual':r['labels'][0],
            'predicted':classes[int(scores.argmax())],'confidence':float(scores.max())} for r,scores in zip(splits['test'],p)],
        'limitations':['Test may include legacy training images; per-source metrics must be considered separately.',
            'Ecuador split by plantation; some classes are absent from the held-out site.',
            'Small observational classes need expert confirmation before deployment.']}
    (output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'test':result['accuracy'],'macro_f1':result['macro_f1'],'readiness':readiness}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--organ',choices=['leaf','ear'],default='leaf')
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--min-train',type=int,default=20,
                        help='Exploratory training minimum only; release evidence requirements remain unchanged')
    args=parser.parse_args()
    if args.epochs<3 or args.batch_size<2 or args.threads<1 or args.min_train<5:parser.error('Need >=3 epochs, batch >=2, >=1 thread and >=5 training examples/class')
    main(args.output,args.organ,args.epochs,args.batch_size,args.threads,args.min_train)
