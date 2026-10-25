"""Запуск модели клубней с местного диска. Обучение выполняется отдельным скриптом."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading
import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "data/models/tuber"
LOCK = threading.Lock()


def image_blob(image):
    # При обучении и диагностике фото должно проходить одинаковую подготовку.
    from app.diagnostics.vision import trim_screenshot_frame, square_image
    image = square_image(trim_screenshot_frame(image), 224)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return rgb.transpose(2, 0, 1)[None].copy()


def extract_features(net, image):
    """Получаем 1280 чисел-признаков из замороженной нейросети."""
    net.setInput(image_blob(image))
    features = net.forward().reshape(-1)
    if features.shape != (1280,) or not np.isfinite(features).all():
        raise ValueError("Invalid tuber backbone output")
    return features


def softmax(logits):
    """Преобразуем оценки классов в числа с суммой 1."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=1)
def load_model():
    """Загружаем основу и обученную голову, сверяя файлы с manifest."""
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get('release_status') == 'retired':
        raise ValueError('The MobileNetV2 tuber model was retired')
    for filename, expected in manifest["files"].items():
        if hashlib.sha256((MODEL_DIR / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Tuber model checksum mismatch")
    net = cv2.dnn.readNetFromONNX(str(MODEL_DIR / "mobilenetv2_features.onnx"))
    with np.load(MODEL_DIR / "classifier.npz", allow_pickle=False) as data:
        head = {key: data[key] for key in data.files}
    return net, head, manifest


def classify_tuber(image):
    # Общий объект OpenCV обрабатывает по одному изображению за раз.
    with LOCK:
        net, head, manifest = load_model()
        features = extract_features(net, image)
    # Используем параметры нормализации и веса, сохранённые при обучении.
    scaled = (features - head["mean"]) / head["scale"]
    logits = (scaled @ head["coef"].T + head["intercept"]) / head["temperature"]
    probabilities = softmax(logits)
    if len(probabilities) != len(manifest["classes"]) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid tuber classifier output")
    return dict(zip(manifest["classes"], map(float, probabilities)))
