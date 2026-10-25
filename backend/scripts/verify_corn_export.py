"""Verify a candidate's ONNX and inference preprocessing against its saved weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from scripts.train_corn_field import Images


def verify(directory):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    onnx=directory/'model.onnx'
    if hashlib.sha256(onnx.read_bytes()).hexdigest()!=manifest['files']['model.onnx']:
        raise ValueError('ONNX checksum mismatch')
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model=models.mobilenet_v3_small(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(manifest['classes']))
    model.load_state_dict(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
    model.eval()
    rows=[json.loads(line) for line in (directory/'validation_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    selected={}
    for row in rows:selected.setdefault(row['labels'][0],row)
    examples=list(selected.values())
    if not examples:raise ValueError('No validation examples')
    images=Images(examples,manifest['classes'])
    net=cv2.dnn.readNetFromONNX(str(onnx));errors=[]
    for i,row in enumerate(examples):
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Validation image checksum mismatch')
        sample=images[i][0].unsqueeze(0)
        # Reproduce application preprocessing independently of torchvision.
        bgr=cv2.imread(str(path))
        if bgr is None:raise ValueError('OpenCV cannot read validation image')
        rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        array=np.asarray(ImageOps.pad(rgb,(224,224),color=(127,127,127)),dtype=np.float32)/255.
        array=((array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        net.setInput(array)
        with torch.inference_mode():reference=model(sample).numpy()
        errors.append(float(np.max(np.abs(net.forward()-reference))))
    maximum=max(errors)
    if maximum>.001:raise ValueError(f'Export/runtime mismatch: {maximum}')
    manifest['onnx_max_abs_error']=maximum
    manifest.pop('active',None)
    manifest.setdefault('release_status','candidate')
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'samples':len(examples),'maximum_logit_error':maximum,'passed':True}
    (directory/'export_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('candidate',type=Path)
    verify(parser.parse_args().candidate)
