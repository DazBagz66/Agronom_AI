"""Independent, checksum-verified classifiers for the active crops."""
from functools import lru_cache
import hashlib
import json
import threading
import cv2
import numpy as np
from app.crops import ROOT, CROPS
from app.diagnostics.tuber_model import extract_features, softmax

LOCK = threading.Lock()


@lru_cache(maxsize=2)
def load_model(crop):
    config = CROPS[crop]
    directory = ROOT / "backend/data/models" / config["id"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["classes"]) != set(config["labels"]):
        raise ValueError("Model classes do not match the selected crop")
    for filename in ["mobilenetv2_features.onnx", "classifier.npz"]:
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != manifest["files"][filename]:
            raise ValueError("Crop model checksum mismatch")
    net = cv2.dnn.readNetFromONNX(str(directory / "mobilenetv2_features.onnx"))
    with np.load(directory / "classifier.npz", allow_pickle=False) as data:
        head = {name: data[name] for name in data.files}
    return net, head, manifest


def classify_crop(crop, image):
    with LOCK:
        net, head, manifest = load_model(crop)
        features = extract_features(net, image)
    scaled = (features - head["mean"]) / head["scale"]
    probabilities = softmax((scaled @ head["coef"].T + head["intercept"]) / head["temperature"])
    if probabilities.shape != (len(manifest["classes"]),) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid crop classifier output")
    return dict(zip(manifest["classes"], map(float, probabilities)))
