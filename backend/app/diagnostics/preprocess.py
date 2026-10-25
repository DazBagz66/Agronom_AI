"""Единая подготовка фотографии для обучения и диагностики.

Обучение и сервер обязаны готовить изображение одинаково, иначе сеть видит
на сервере не то, на чём училась. Поэтому оба пути вызывают функции отсюда.

Шаги: EXIF-поворот → RGB → уменьшение до 320 px по длинной стороне (LANCZOS)
→ вписывание в серый квадрат 224×224 без растяжения → нормализация ImageNet.
"""
import numpy as np
from PIL import Image, ImageOps

THUMBNAIL_SIDE = 320
INPUT_SIDE = 224
FILL = (127, 127, 127)
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)
DESCRIPTION = ("EXIF transpose, RGB, thumbnail max side 320 (LANCZOS), "
               "pad to 224x224 gray (127,127,127), ImageNet mean/std")


def load_rgb(path):
    """Читаем файл с диска: поворот по EXIF, RGB и уменьшенная копия."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE), Image.Resampling.LANCZOS)
        return image


def from_bgr(array):
    """Кадр OpenCV (BGR, уже повёрнутый по EXIF при декодировании) → такая же уменьшенная копия."""
    image = Image.fromarray(np.ascontiguousarray(array[:, :, ::-1]))
    image.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE), Image.Resampling.LANCZOS)
    return image


def letterbox(image):
    """Вписываем фото в квадрат с серыми полями, не искажая пропорции."""
    return ImageOps.pad(image, (INPUT_SIDE, INPUT_SIDE), color=FILL)


def to_tensor(image):
    """PIL 224×224 → массив (1, 3, 224, 224) float32 с нормализацией ImageNet."""
    pixels = np.asarray(image, dtype=np.float32) / 255.
    pixels = (pixels - MEAN) / STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def model_input(image):
    """Полный путь для сервера: уменьшенная PIL-копия → входной тензор."""
    return to_tensor(letterbox(image))
