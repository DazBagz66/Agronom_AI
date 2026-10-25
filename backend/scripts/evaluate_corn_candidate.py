"""Evaluate a completed ONNX candidate on the fixed, previously reviewed field set."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.diagnostics.tuber_model import softmax


def evaluate(directory):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    model_path=directory/'model.onnx'
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=manifest['files']['model.onnx']:
        raise ValueError('Candidate checksum mismatch')
    net=cv2.dnn.readNetFromONNX(str(model_path));cv2.setNumThreads(2)
    old=json.loads((ROOT/'backend/data/datasets/corn_expansion/current_model_review.json').read_text(encoding='utf-8'))
    results=[]
    for row in old['predictions']:
        if row['source']!='pretoria':continue
        with Image.open(ROOT/row['path']) as raw:
            image=np.array(ImageOps.pad(ImageOps.exif_transpose(raw).convert('RGB'),(224,224),color=(127,127,127)),dtype=np.float32)/255.
        blob=((image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        net.setInput(blob);p=softmax(net.forward().reshape(-1)/manifest['temperature'])
        labels=['healthy' if x=='healthy_leaf' else x for x in row['expected_labels']]
        results.append({'path':row['path'],'actual':labels,'predicted':manifest['classes'][int(p.argmax())],
            'confidence':float(p.max()),'legacy_prediction':row['raw_prediction'],
            'comparable':row['comparable_single_label']})
    comparable=[r for r in results if r['comparable']]
    report={'model_sha256':manifest['files']['model.onnx'],'reviewed':len(results),
        'common_three_classes':{'count':len(comparable),
            'candidate_correct':sum(r['actual']==[r['predicted']] for r in comparable),
            'legacy_correct':sum(r['actual']==[r['legacy_prediction']] for r in comparable)},
        'limitations':['Small stratified, previously inspected field set; not a blind validation.',
                       'Evaluation happens after checkpoint selection; not used for training.'],
        'predictions':results}
    (directory/'field_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('candidate',type=Path)
    evaluate(parser.parse_args().candidate)
