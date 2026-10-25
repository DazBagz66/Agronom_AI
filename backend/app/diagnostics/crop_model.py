"""Классификаторы болезней активных культур: выбор, проверка и запуск.

Активные модели перечислены в ``backend/data/models/registry.json``::

    {"corn": {"leaf": "backend/data/models/corn_candidates/leaf_v3"},
     "sunflower": {"leaf": "backend/data/models/sunflower_candidates/leaf_v2"}}

Каждый каталог содержит ``manifest.json`` формата ``onnx_logits_v1`` и
``model.onnx`` (см. ``backend/scripts/train_classifier.py``). Манифест хранит
классы, температуру калибровки, список выпущенных классов и контрольную сумму
файла. Модель без записи в реестре, без статуса ``approved`` или с неверной
суммой не запускается: сервер честно отвечает, что модели нет.
"""
from functools import lru_cache
import hashlib
import json
import threading

import cv2
import numpy as np

from app.crops import ROOT, CROPS
from app.corn_taxonomy import CORN_LABELS
from app.diagnostics import preprocess

MODELS = ROOT / "backend/data/models"
REGISTRY = MODELS / "registry.json"
LOCK = threading.Lock()


class ModelNotInstalledError(ValueError):
    """Для органа нет выпущенной модели — в отличие от повреждённого файла."""


def registry():
    if not REGISTRY.exists():
        return {}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def model_directory(crop_id, organ):
    """Каталог активной модели; путь обязан лежать внутри хранилища моделей."""
    entry = registry().get(crop_id, {}).get(organ)
    if not entry:
        raise ModelNotInstalledError(f"No released {crop_id}/{organ} model")
    directory = (ROOT / entry).resolve()
    if not directory.is_relative_to(MODELS.resolve()):
        raise ValueError("Model path outside model storage")
    return directory


def known_labels(config):
    """Допустимые ключи классов и их русские названия для культуры."""
    labels = dict(config["labels"])
    if config["id"] == "corn":
        labels = {**CORN_LABELS, **labels}
    return labels


def validate_manifest(manifest, config, organ):
    if manifest.get("format") != "onnx_logits_v1":
        raise ValueError("Unsupported model format")
    if manifest.get("release_status") != "approved":
        raise ModelNotInstalledError("Model has not been approved for release")
    if organ not in manifest.get("supported_organ", "").split(","):
        raise ValueError("Model organ mismatch")
    classes, released = manifest["classes"], manifest.get("release_classes") or []
    if not set(classes) <= set(known_labels(config)):
        raise ValueError("Model classes outside the crop taxonomy")
    if not released or not set(released) <= set(classes):
        raise ValueError("Release classes are missing or outside the model classes")
    if not np.isfinite(manifest["temperature"]) or manifest["temperature"] <= 0:
        raise ValueError("Invalid calibration temperature")


@lru_cache(maxsize=4)
def load_model(crop, organ="leaf"):
    """Сеть OpenCV и манифест; обе проверки — целостность и статус — здесь."""
    config = CROPS[crop]
    directory = model_directory(config["id"], organ)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest, config, organ)
    model_path = directory / "model.onnx"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["files"]["model.onnx"]:
        raise ValueError("Model checksum mismatch")
    manifest["_loaded_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return cv2.dnn.readNetFromONNX(str(model_path)), manifest


def crop_capabilities(crop):
    """Интерфейсу показываем только органы и классы установленных моделей."""
    config = CROPS[crop]
    parts, labels = [], {}
    for organ in config["organs"]:
        try:
            _, manifest = load_model(crop, organ)
        except (OSError, ValueError, KeyError, cv2.error):
            continue
        parts.extend(config["organs"][organ])
        labels.update({key: known_labels(config)[key] for key in manifest["release_classes"]})
    return {"parts": parts, "labels": labels}


def classify_crop(crop, image, organ="leaf"):
    """Вероятности выпущенных классов; масса невыпущенных попадает в ``unvalidated``."""
    with LOCK:
        net, manifest = load_model(crop, organ)
        net.setInput(preprocess.model_input(preprocess.from_bgr(image)))
        logits = net.forward().reshape(-1)
    if logits.shape != (len(manifest["classes"]),) or not np.isfinite(logits).all():
        raise ValueError("Invalid model output")
    scaled = logits / manifest["temperature"]
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    released = set(manifest["release_classes"])
    result = {label: float(p) for label, p in zip(manifest["classes"], probabilities) if label in released}
    result["unvalidated"] = float(sum(p for label, p in zip(manifest["classes"], probabilities) if label not in released))
    return result
