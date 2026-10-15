"""Подготовка фото, наблюдения поверхности и выбор модели по органу."""
import base64
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import threading

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models" / "potato"
MODEL_HASH = "cff894c2d718db1a11647aab4829a39b1c3ab272f177537087e40f393daa6624"
MODEL_LOCK = threading.Lock()


def trim_screenshot_frame(image):
    """Обрезаем простые однотонные поля скриншота, сохраняя крупное изображение."""
    height, width = image.shape[:2]
    if min(height, width) < 96:
        return image
    scale = min(1., 900 / max(height, width))
    preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    border = np.concatenate([preview[0], preview[-1], preview[:, 0], preview[:, -1]])
    background = np.median(border, axis=0)
    # Цветные и неоднородные края обычно относятся к самой фотографии.
    if not (background.mean() > 235 or background.mean() < 30):
        return image
    if np.mean(np.linalg.norm(border.astype(float) - background, axis=1) < 22) < .9:
        return image
    difference = np.linalg.norm(preview.astype(float) - background, axis=2)
    foreground = (difference > 28).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(foreground)
    if count < 2:
        return image
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    if largest < preview.shape[0] * preview.shape[1] * .015:
        return image
    substantial = [stat for stat in stats[1:] if stat[cv2.CC_STAT_AREA] >= largest * .25]
    left = min(stat[cv2.CC_STAT_LEFT] for stat in substantial)
    top = min(stat[cv2.CC_STAT_TOP] for stat in substantial)
    right = max(stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH] for stat in substantial)
    bottom = max(stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT] for stat in substantial)
    box_area = (right - left) * (bottom - top)
    if box_area > preview.shape[0] * preview.shape[1] * .85:
        return image
    # Небольшой отступ помогает сохранить края объекта.
    margin = int(max(right - left, bottom - top) * .025)
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(preview.shape[1], right + margin), min(preview.shape[0], bottom + margin)
    return image[int(top / scale):int(bottom / scale), int(left / scale):int(right / scale)]


def square_image(image, size):
    """Вписываем фото в квадрат с полями, не растягивая объект."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    corners = np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])
    background = tuple(map(int, np.median(corners, axis=0)))
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    return cv2.copyMakeBorder(resized, y, size - resized.shape[0] - y,
                             x, size - resized.shape[1] - x, cv2.BORDER_CONSTANT, value=background)


class VisionTriageReport(BaseModel):
    """Промежуточные наблюдения по фото до формирования диагноза."""
    is_plant_visible: bool = False
    is_healthy: bool = False
    object_type: str = "текст"
    image_quality_score: int = Field(default=0, ge=0, le=100)
    green_pct: float = 0
    yellow_pct: float = 0
    brown_pct: float = 0
    spot_count: int = 0
    extracted_visual_symptoms: str = ""
    preliminary_visual_hypotheses: list[str] = Field(default_factory=list)
    has_contradictions_or_lack_of_data: bool = True
    triage_notes: str = ""
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    healthy_confidence: float = 0
    lesion_pct: float = 0
    model_available: bool = False
    no_visible_damage: bool = False
    surface_varied: bool = False


@lru_cache(maxsize=1)
def leaf_model():
    """Проверяем файл весов и загружаем CNN один раз на процесс."""
    path = MODEL_DIR / "potato_leaf.onnx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_HASH:
        raise ValueError("Model checksum mismatch")
    return cv2.dnn.readNetFromONNX(str(path))


def classify_leaf(image):
    # Формат листовой CNN: RGB 256×256, пиксели / 255; на выходе три оценки.
    blob = cv2.dnn.blobFromImage(square_image(image, 256), 1 / 255., swapRB=True)
    with MODEL_LOCK:
        net = leaf_model()
        net.setInput(blob)
        probabilities = np.asarray(net.forward()).reshape(-1)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model output")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=.01):
        raise ValueError("Expected normalized probabilities")
    labels = json.loads((MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    expected = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
    if [labels[str(i)] for i in range(3)] != expected:
        raise ValueError("Class mapping mismatch")
    return dict(zip(["alternaria", "late_blight", "healthy"], map(float, probabilities)))


def decode_image(encoded):
    """Превращаем строку Base64 из запроса в изображение OpenCV."""
    try:
        data = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, cv2.error):
        return None


def analyze_image(request):
    if not request.image_base64:
        return VisionTriageReport(triage_notes="Фото не приложено.")
    image = decode_image(request.image_base64)
    if image is None:
        return VisionTriageReport(triage_notes="Не удалось прочитать фото. Прикрепите изображение повторно.")
    if min(image.shape[:2]) < 96:
        return VisionTriageReport(triage_notes="Изображение слишком маленькое. Прикрепите более крупное фото.")
    image = trim_screenshot_frame(image)
    if max(image.shape[:2]) < 96:
        return VisionTriageReport(triage_notes="Фотография внутри скриншота слишком маленькая для оценки деталей.")
    small = square_image(image, 400)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Резкость проверяем до изменения размера, чтобы сами не ухудшили оценку.
    sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    quality = int(min(100, sharpness / 1.5))
    # Отделяем цветной объект от нейтрального фона; коричневая кожура сама по себе нормальна.
    mask = cv2.inRange(hsv, np.array([0, 28, 30]), np.array([100, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    coverage = np.count_nonzero(mask) / mask.size
    if coverage < .08 or sharpness < 12:
        return VisionTriageReport(image_quality_score=quality,
            triage_notes="На фото мало различимых деталей. Снимите картофель ближе, при ровном освещении.")
    foreground = mask > 0
    green = cv2.inRange(hsv, np.array([35, 35, 30]), np.array([90, 255, 255])) > 0
    yellow = cv2.inRange(hsv, np.array([15, 30, 45]), np.array([34, 255, 255])) > 0
    brown = cv2.inRange(hsv, np.array([0, 35, 25]), np.array([22, 255, 155])) > 0
    denominator = max(1, foreground.sum())
    green_pct = float((green & foreground).sum() / denominator * 100)
    yellow_pct = float((yellow & foreground).sum() / denominator * 100)
    brown_pct = float((brown & foreground).sum() / denominator * 100)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main = max(contours, key=cv2.contourArea)
    _, _, width, height = cv2.boundingRect(main)
    elongation = max(width, height) / max(1, min(width, height))
    selected = request.plant_part
    if selected == "auto":
        if green_pct < 12 and yellow_pct + brown_pct > 55:
            selected = "tuber"
        elif elongation > 3.2 and green_pct > 55:
            selected = "stem"
        else:
            selected = "leaf"
    names = {"leaf": "лист", "stem": "стебель", "tuber": "клубень", "whole": "растение целиком"}
    organ = names[selected]
    interior = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    # Ищем локальные потемнения относительно соседней ткани, исключая влияние фона.
    tissue = foreground.astype(np.float32)
    baseline = cv2.GaussianBlur(gray.astype(np.float32) * tissue, (41, 41), 0)
    baseline /= np.maximum(cv2.GaussianBlur(tissue, (41, 41), 0), .001)
    depressions = ((baseline - gray) > (30 if selected == "tuber" else 25)) & interior
    spots, _ = cv2.findContours(depressions.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = sum(1 for c in spots if cv2.contourArea(c) > 30)
    lesion_pct = float(depressions.sum() / max(1, interior.sum()) * 100)
    report = VisionTriageReport(is_plant_visible=True, object_type=organ,
        image_quality_score=quality, green_pct=green_pct, yellow_pct=yellow_pct,
        brown_pct=brown_pct, spot_count=spot_count, lesion_pct=lesion_pct,
        has_contradictions_or_lack_of_data=False,
        triage_notes=f"Рассмотрен {organ}. " + ("Есть неоднородные тёмные участки." if lesion_pct > 3 else "Выраженных локальных потемнений не видно."))
    # Дополнительно проверяем широкие изменения цвета, которые не выглядят как отдельные пятна.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for index in range(1, component_count):
        if stats[index, cv2.CC_STAT_AREA] < .05 * mask.size:
            continue
        pixels = lab[:, :, 1:][(labels == index) & interior].astype(np.float32)
        if len(pixels) < 100:
            continue
        distance = np.linalg.norm(pixels - np.median(pixels, axis=0), axis=1)
        if np.mean(distance > 15) > .15:
            report.surface_varied = True
    if selected == "tuber" and report.surface_varied:
        report.triage_notes = "Рассмотрен клубень. Видны участки неоднородной окраски; по ним нельзя отличить особенности кожуры от повреждения."
    if selected == "leaf":
        try:
            report.class_probabilities = classify_leaf(image)
            report.model_available = True
            report.healthy_confidence = report.class_probabilities["healthy"]
        except (OSError, ValueError, cv2.error):
            logger.exception("Local leaf model unavailable")
            report.triage_notes += " Автоматическая оценка листа сейчас недоступна."
        # Общий вид зелёной ботвы может не подходить модели отдельного листа.
        if green_pct > 85 and brown_pct < 3:
            if report.class_probabilities.get("healthy", 0) < .65:
                report.class_probabilities = {}
                report.healthy_confidence = 0
                report.no_visible_damage = True
                report.triage_notes += " Для оценки отдельного листа нужен крупный план."
    elif selected == "tuber":
        from app.diagnostics.tuber_model import classify_tuber
        try:
            report.class_probabilities = classify_tuber(image)
            report.model_available = True
            report.healthy_confidence = report.class_probabilities["healthy"]
            if max(report.class_probabilities, key=report.class_probabilities.get) == "unknown":
                report.triage_notes += " Изменения не удалось отнести к поддерживаемым болезням клубней."
        except (OSError, ValueError, KeyError, cv2.error):
            logger.exception("Local tuber model unavailable")
            report.triage_notes += " Автоматическая оценка клубня сейчас недоступна."
        # Неуверенный ответ модели не заменяем выводом «здоров» только по цвету.
    elif selected == "stem":
        backdrop = hsv[:, :, 1] < 28
        clear_backdrop = backdrop.sum() / mask.size > .12
        normal_colour = (yellow_pct + brown_pct > 60) if selected == "tuber" else green_pct > 80
        if clear_backdrop and normal_colour and lesion_pct < 1.5 and spot_count <= 4 and quality >= 35 and not report.surface_varied:
            report.no_visible_damage = True
            report.triage_notes += " Поверхность выглядит равномерной, заметных повреждений не обнаружено."
    elif selected == "whole":
        report.triage_notes += " Для проверки отдельных болезней снимите лист, основание стебля или клубень крупнее."
    report.is_healthy = report.healthy_confidence >= .65
    report.extracted_visual_symptoms = report.triage_notes
    return report
