"""Offline coarse organ router with independent weights and checksum validation."""
from functools import lru_cache
import hashlib
import json
import threading

import cv2
import numpy as np

from app.crops import ROOT
from app.diagnostics.tuber_model import image_blob, softmax

MODEL_DIR = ROOT / 'backend/data/models/corn_organ'
LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_organ_model():
    manifest_bytes = (MODEL_DIR/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('release_status') != 'approved' or manifest.get('format') != 'corn_organ_features_v2':
        raise ValueError('Organ router has not passed validation')
    if manifest['classes'] != ['ear','leaf','other','whole'] or not .5 <= manifest['threshold'] <= 1:
        raise ValueError('Invalid organ model metadata')
    backbone = MODEL_DIR/'features.onnx'
    classifier = MODEL_DIR/'classifier.npz'
    if hashlib.sha256(backbone.read_bytes()).hexdigest() != manifest['backbone_sha256']:
        raise ValueError('Organ backbone checksum mismatch')
    if hashlib.sha256(classifier.read_bytes()).hexdigest() != manifest['classifier_sha256']:
        raise ValueError('Organ classifier checksum mismatch')
    with np.load(classifier,allow_pickle=False) as data:
        head = {name:data[name] for name in data.files}
    if not np.isfinite(head['temperature']) or head['temperature'] <= 0:
        raise ValueError('Invalid organ calibration')
    manifest['sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    return cv2.dnn.readNetFromONNX(str(backbone)),head,manifest


def extract_features(net, image):
    net.setInput(image_blob(image))
    features = net.forward().reshape(-1)
    if features.shape != (960,) or not np.isfinite(features).all():
        raise ValueError('Invalid MobileNetV3 organ features')
    return features


def detect_corn_organ(image):
    with LOCK:
        net,head,manifest = load_organ_model()
        features = extract_features(net,image)
    scaled = (features-head['mean'])/head['scale']
    probabilities = softmax((scaled@head['coef'].T+head['intercept'])/head['temperature'])
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid organ prediction')
    index = int(probabilities.argmax())
    score = float(probabilities[index])
    candidate = manifest['classes'][index]
    # part — решение по порогу; candidate — лучший класс даже ниже порога,
    # чтобы маршрутизация могла отличить «скорее лист» от «посторонний объект».
    part = 'unknown' if score < manifest['threshold'] or candidate == 'other' else candidate
    return {'part': part, 'candidate': candidate, 'score': score,
            'probabilities': dict(zip(manifest['classes'], map(float, probabilities))),
            'model_sha256': manifest['sha256']}
