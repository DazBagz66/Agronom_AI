"""Image-based advice restricted to the classes of the selected crop dataset."""
import logging
import math
import re
import cv2
from app.crops import CROPS
from app.schemas import DiagnosisResponse, ProbableCause
from app.diagnostics.vision import decode_image
from app.diagnostics.crop_model import classify_crop

logger = logging.getLogger(__name__)


def insufficient(message, confidence=0):
    return DiagnosisResponse(
        status="insufficient_data", overall_confidence=confidence,
        assessment_basis="visual_observation", visual_observations=message,
        human_explanation=message, recommendations=["Приложите чёткий снимок одного поражённого органа при естественном освещении."],
        clarifying_questions=["Какие изменения видны и когда они появились?"],
    )


def apply_crop_display_policy(result):
    """A closed-set Healthy prediction has not passed external disease screening.

    Also applied to historical responses without changing the stored records.
    Raising a softmax threshold does not solve confidently wrong predictions.
    """
    if result.status != "healthy":
        return result
    revised = insufficient(
        f"Модель выбрала класс «Здоровый лист» (оценка {result.overall_confidence}%). "
        "Это сходство с примерами обучающего набора, а не вероятность здоровья растения. "
        "Отсутствие заболевания не подтверждено: на внешних фотографиях модель может пропускать поражения. "
        "Если видны изменения, нужен крупный план изменённого участка и осмотр растения."
    )
    revised.recommendations = [
        "Если видны пятна, наросты или другие повреждения, приложите их крупный план.",
        "Если видимых изменений нет, наблюдайте за растением; результат модели не исключает другие заболевания.",
    ]
    revised.clarifying_questions = ["Есть ли видимые изменения на листьях, стебле или початке?"]
    revised.check_id = result.check_id
    revised.created_at = result.created_at
    return revised


def diagnose_crop(request):
    config = CROPS[request.crop]
    if request.plant_part not in config["parts"]:
        return insufficient("Выберите поддерживаемую часть растения: " +
                            ("лист." if config["id"] == "corn" else "лист или корзинку. Отдельной модели стебля и всего растения пока нет."))
    if not request.image_base64:
        return insufficient(f"Для проверки культуры «{request.crop}» нужна фотография. По одному описанию болезнь не определяется.")
    image = decode_image(request.image_base64)
    if image is None:
        return insufficient("Не удалось прочитать изображение. Загрузите JPG или PNG повторно.")
    if min(image.shape[:2]) < 96:
        return insufficient("Изображение слишком маленькое для оценки. Нужен снимок не менее 96 пикселей по каждой стороне.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5 or cv2.Laplacian(gray, cv2.CV_64F).var() < 10:
        return insufficient("На фото недостаточно различимых деталей. Сделайте более резкий снимок.")
    try:
        probabilities = classify_crop(request.crop, image)
    except (OSError, ValueError, KeyError, cv2.error):
        logger.exception("Crop model unavailable: %s", request.crop)
        return insufficient("Модель выбранной культуры недоступна. Проверьте установку её файлов и повторите запрос.")
    label, score = max(probabilities.items(), key=lambda pair: pair[1])
    confidence = math.floor(score * 100)
    if score < .65:
        return insufficient("Ни один класс не достиг порога 65%. Нужны дополнительные снимки и осмотр растения.", confidence)
    if request.plant_part == "flower" and label != "gray_mold":
        return insufficient("Для корзинки недостаточно подтверждения: получен класс, относящийся к листьям. Приложите отдельный снимок листа или обратитесь к специалисту.", confidence)
    # Healthy may win even on visibly diseased out-of-distribution photos.
    # Do not certify health until independently validated screening is available.
    if label == "healthy":
        if re.search(r"пят|нал[её]т|гнил|увяда|желт|жёлт|поврежд", request.symptoms or "", re.I):
            return insufficient("Фото похоже на класс здорового листа, но описание содержит признаки повреждения. Нужен крупный план изменённого участка.", confidence)
        return apply_crop_display_policy(DiagnosisResponse(status="healthy", overall_confidence=confidence, assessment_basis="model",
            visual_observations="Снимок похож на изображения здоровых листьев выбранной культуры.",
            human_explanation="В рамках поддерживаемых классов признаков болезни не обнаружено. Другие болезни и скрытые повреждения не исключены.",
            recommendations=["Наблюдайте за растением; при появлении изменений сделайте новый снимок."]))
    name = config["labels"][label]
    return DiagnosisResponse(status="success", overall_confidence=confidence, assessment_basis="model",
        visual_observations=f"Изображение похоже на класс «{name}» в наборе выбранной культуры.",
        probable_causes=[ProbableCause(name=name, category="неопределено" if label == "leaf_scars" else "болезнь",
            probability=confidence, reasoning="Результат классификации фотографии моделью выбранной культуры; требуется проверка при осмотре.")],
        human_explanation=f"Предварительная гипотеза: {name}. Оценка модели не является подтверждённым диагнозом."
            + (" В наборе нет отдельного класса здоровых корзинок: результат может зависеть от вида органа, а не только от поражения." if request.plant_part == "flower" else "")
            + (" Метка Leaf scars описывает повреждения и не устанавливает возбудителя." if label == "leaf_scars" else ""),
        recommendations=["Сравните несколько поражённых и непоражённых растений на участке.",
                         "Сделайте снимки поражённого участка с обеих сторон и зафиксируйте дату появления симптомов.",
                         "Подтвердите причину у агронома перед выбором обработки."])
