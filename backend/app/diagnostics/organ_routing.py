"""Автоматический выбор органа кукурузы по фото.

Определитель органа — грубый фильтр, обученный на 737 снимках. На полевых
крупных планах он нередко сомневается, поэтому его неуверенность не должна
блокировать диагностику:

  * уверенный ответ «лист» или «початок» — идём к модели этого органа;
  * «общий вид» или уверенно посторонний объект — просим крупный план;
  * ниже порога, но похоже на часть растения — спрашиваем модель листьев:
    если она уверенно узнаёт выпущенный класс, считаем снимок листом,
    иначе честно просим переснять.
Модель листьев никогда не применяется вместо отсутствующей модели початков.
"""
import logging
import math

import cv2

from app.diagnostics.corn_organ import detect_corn_organ
from app.diagnostics.crop_advisor import insufficient, ACCEPTANCE_THRESHOLD
from app.diagnostics.crop_model import classify_crop
from app.diagnostics.vision import decode_image

logger = logging.getLogger(__name__)
MINIMUM_SIDE = 96


def route_corn_photo(request):
    """Возвращает уточнённый запрос, метаданные ответа и ранний ответ (или None)."""
    if request.crop != 'Кукуруза' or request.plant_part != 'auto':
        return request, {}, None
    metadata = {'organ_detection_method': 'automatic'}
    image, problem = _usable_photo(request)
    if problem:
        return request, metadata, problem
    try:
        observation = detect_corn_organ(image)
    except (OSError, ValueError, KeyError, cv2.error):
        logger.exception('Corn organ router unavailable')
        return request, metadata, insufficient(
            'Автоматическое определение объекта временно недоступно. Повторите запрос после восстановления модели.')
    part = observation['part']
    if part == 'unknown' and observation.get('candidate', 'other') != 'other':
        part = _leaf_if_leaf_model_is_confident(request, image, observation)
    metadata.update(detected_plant_part=part, organ_confidence=_percent(observation, part),
                    organ_model_sha256=observation['model_sha256'])
    if part == 'unknown':
        result = insufficient('Не удалось уверенно определить объект на фото. '
                              'Снимите один лист или один початок крупнее, без посторонних предметов.')
        result.recommendations = [
            'Поднесите камеру ближе: один лист или початок должен занимать примерно ⅔–⅘ длины кадра. '
            'Оставьте немного фона, используйте рассеянный дневной свет и резкий фокус. '
            'Это ориентир для съёмки, размер объекта автоматически не измерялся.',
        ]
        return request, metadata, result
    resolved = request.model_copy(update={'plant_part': part})
    if part == 'whole':
        result = insufficient('На фото общий вид растения или посадки. '
                              'Для определения болезни нужен отдельный крупный снимок листа, початка или изменённого участка.')
        result.recommendations = ['Сохраните общий вид для сравнения и добавьте крупный план органа, на котором видны изменения.']
        return resolved, metadata, result
    return resolved, metadata, None


def _usable_photo(request):
    if not request.image_base64:
        return None, insufficient('Для автоматического определения органа нужна фотография растения.')
    image = decode_image(request.image_base64)
    if image is None:
        return None, insufficient('Не удалось прочитать изображение. Загрузите JPG или PNG повторно.')
    if min(image.shape[:2]) < MINIMUM_SIDE:
        height, width = image.shape[:2]
        return None, insufficient(
            f'Фото {width} × {height} пикселей слишком маленькое. Загрузите исходный снимок; '
            'желательно от 600 пикселей по короткой стороне. Увеличение этого файла не добавит деталей.')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5 or cv2.Laplacian(gray, cv2.CV_64F).var() < 10:
        return None, insufficient('На фото мало различимых деталей. Сделайте более резкий снимок при дневном свете.')
    return image, None


def _leaf_if_leaf_model_is_confident(request, image, observation):
    """Определитель сомневается: уверенный выпущенный класс модели листьев перевешивает."""
    try:
        probabilities = classify_crop(request.crop, image)
    except (OSError, ValueError, KeyError, cv2.error):
        logger.info('Leaf model unavailable for uncertain organ; keeping unknown', exc_info=True)
        return 'unknown'
    label, score = max(probabilities.items(), key=lambda pair: pair[1])
    if label != 'unvalidated' and score >= ACCEPTANCE_THRESHOLD:
        logger.info('Organ router uncertain (%s %.2f); leaf model confident (%s %.2f)',
                    observation['candidate'], observation['score'], label, score)
        return 'leaf'
    return 'unknown'


def _percent(observation, part):
    """Уверенность в выбранном органе; для листа после запасной проверки — доля «лист» у определителя."""
    probabilities = observation.get('probabilities')
    if probabilities and part in probabilities:
        return math.floor(probabilities[part] * 100)
    return math.floor(observation['score'] * 100)
