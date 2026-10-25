"""Диагноз по фотографии в пределах классов модели выбранной культуры.

Порядок работы ``diagnose_crop``:
  1. проверяем, что выбранная часть растения поддерживается и фото пригодно;
  2. запускаем классификатор нужного органа (лист или початок);
  3. если модели нет — честно сообщаем об этом (для початка дополнительно
     запускаем экспериментальную проверку на головню);
  4. применяем порог 65 % и правила отображения: «здоровый» класс никогда не
     выдаётся как подтверждённое здоровье растения.
"""
import logging
import math
import re

import cv2

from app.crops import CROPS
from app.corn_taxonomy import CORN_LABELS
from app.schemas import DiagnosisResponse, ProbableCause
from app.diagnostics.vision import decode_image
from app.diagnostics.crop_model import classify_crop, ModelNotInstalledError
from app.diagnostics.corn_care import care_for, reported_damage, LEAF_PHOTO_TIPS, EAR_PHOTO_TIPS
from app.knowledge.corn_history import refresh_corn_references

logger = logging.getLogger(__name__)

ACCEPTANCE_THRESHOLD = .65
MINIMUM_SIDE = 96
NOT_A_DIAGNOSIS = {"unvalidated", "other_damage", "unknown_disease"}
HEALTHY_LABELS = {"healthy", "healthy_ear"}
DAMAGE_WORDS = re.compile(r"пят|нал[её]т|гнил|увяда|желт|жёлт|поврежд", re.I)
GENERIC_RECOMMENDATIONS = [
    "Сравните несколько поражённых и непоражённых растений на участке.",
    "Сделайте снимки поражённого участка с обеих сторон и зафиксируйте дату появления симптомов.",
    "Подтвердите причину у агронома перед выбором обработки.",
]
LEGACY_BLIGHT_NAMES = ["Ожог листьев кукурузы (Blight)", "Поражение листьев типа Blight — вид требует уточнения"]


def insufficient(message, confidence=0):
    return DiagnosisResponse(
        status="insufficient_data", overall_confidence=confidence,
        assessment_basis="visual_observation", visual_observations=message,
        human_explanation=message,
        recommendations=["Приложите чёткий снимок одного органа растения при естественном освещении."],
        clarifying_questions=["Какие изменения видны и когда они появились?"],
    )


# ----------------------------------------------------------------------------
# Правила отображения (применяются и к сохранённой истории)
# ----------------------------------------------------------------------------
def apply_crop_display_policy(result, crop=None):
    """Правила отображения, в том числе для сохранённой истории.

    Ответ «здоровый» от текущих моделей (``model_label`` заполнен) показывается
    как статус «здоров»: их класс «здоровый» проверен на чужих источниках.
    Старые записи без ``model_label`` получены прежними моделями, которые
    называли здоровыми и больные листья, поэтому для них «здоровый» остаётся
    наблюдением без вывода. Исторические записи при этом не переписываются.
    """
    if crop == "Кукуруза":
        result = refresh_corn_references(result)
        if _is_confident_single_diagnosis(result):
            return _rename_legacy_corn_label(result)
    if result.status != "healthy" or result.model_label in HEALTHY_LABELS:
        return result
    return _healthy_as_observation(result, crop)


def _is_confident_single_diagnosis(result):
    return (result.status == "success" and result.overall_confidence >= 65
            and len(result.probable_causes) == 1 and result.probable_causes[0].probability >= 65)


def _rename_legacy_corn_label(result):
    """Старые записи называли классы иначе; показываем актуальные названия и карточки."""
    config = CROPS["Кукуруза"]
    keys_by_name = {name: key for key, name in config["labels"].items()}
    keys_by_name.update({name: "blight" for name in LEGACY_BLIGHT_NAMES})
    old_name = result.probable_causes[0].name
    label = keys_by_name.get(old_name)
    if not label or label == "healthy":
        return result
    result = result.model_copy(deep=True)
    name = config["labels"][label]
    result.probable_causes[0].name = name
    result.visual_observations = result.visual_observations.replace(old_name, name)
    result.human_explanation = result.human_explanation.replace(old_name, name)
    if label == "blight" and "солнечный" not in result.human_explanation:
        result.human_explanation += (" Ожогоподобные пятна не подтверждают солнечный или химический ожог; "
                                     "вид поражения требует уточнения.")
    if not result.knowledge_matches:
        result.recommendations, result.sources = care_for(label)
    return result


def _healthy_as_observation(result, crop):
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
    if crop == "Кукуруза":
        revised.model_observation = "healthy_leaf_like"
        revised.recommendations = [
            "Если видимых изменений нет, наблюдайте за растением. "
            "Обработка от болезни только по этому результату не требуется.",
            *LEAF_PHOTO_TIPS,
        ]
    return revised


# ----------------------------------------------------------------------------
# Диагностика по фото
# ----------------------------------------------------------------------------
def diagnose_crop(request):
    config = CROPS[request.crop]
    supported_parts = {part for parts in config["organs"].values() for part in parts}
    if request.plant_part not in supported_parts:
        hint = "лист." if config["id"] == "corn" else "лист или корзинку. Отдельной модели стебля и всего растения пока нет."
        return insufficient("Выберите поддерживаемую часть растения: " + hint)
    image, problem = _usable_photo(request)
    if problem:
        return problem
    organ = "ear" if request.plant_part == "ear" else "leaf"
    try:
        probabilities = classify_crop(request.crop, image, organ) if organ == "ear" else classify_crop(request.crop, image)
    except ModelNotInstalledError:
        return _no_model_response(organ)
    except (OSError, ValueError, KeyError, cv2.error):
        logger.exception("Crop model unavailable: %s", request.crop)
        return insufficient("Модель выбранной культуры недоступна. Проверьте установку её файлов и повторите запрос.")
    return _interpret(request, config, probabilities)


def _usable_photo(request):
    """Декодируем фото и отсеиваем снимки, по которым модель не может работать."""
    if not request.image_base64:
        return None, insufficient(f"Для проверки культуры «{request.crop}» нужна фотография. "
                                  "По одному описанию болезнь не определяется.")
    image = decode_image(request.image_base64)
    if image is None:
        return None, insufficient("Не удалось прочитать изображение. Загрузите JPG или PNG повторно.")
    if min(image.shape[:2]) < MINIMUM_SIDE:
        return None, insufficient("Изображение слишком маленькое для оценки. "
                                  f"Нужен снимок не менее {MINIMUM_SIDE} пикселей по каждой стороне.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5 or cv2.Laplacian(gray, cv2.CV_64F).var() < 10:
        return None, insufficient("На фото недостаточно различимых деталей. Сделайте более резкий снимок.")
    return image, None


def _no_model_response(organ):
    if organ == "leaf":
        result = insufficient("Прежняя модель болезней листьев удалена. Новая модель ещё не прошла проверку; "
                              "диагноз по этому фото сейчас не выставляется. "
                              "Добавьте описание изменений для разбора наблюдений.")
        result.recommendations = list(LEAF_PHOTO_TIPS)
        return result
    result = insufficient(
        "Автоматическая диагностика початков пока недоступна: модель початков не установлена. "
        "Модель листьев к этому фото не применяется. Ни головня, ни гниль, ни здоровье початка по нему не установлены."
    )
    result.recommendations = list(EAR_PHOTO_TIPS[:2])
    result.clarifying_questions = ["Что изменилось на початке: форма, цвет зёрен, налёт или наросты?"]
    return result


def _interpret(request, config, probabilities):
    """Порог, особые классы и итоговый ответ по вероятностям модели."""
    label, score = max(probabilities.items(), key=lambda pair: pair[1])
    confidence = math.floor(score * 100)
    if score < ACCEPTANCE_THRESHOLD:
        return insufficient("Ни один класс не достиг порога 65%. Нужны дополнительные снимки и осмотр растения.",
                            confidence)
    if label in HEALTHY_LABELS:
        return _healthy_response(request, config, confidence, label)
    if label in NOT_A_DIAGNOSIS:
        return insufficient("По этому снимку нельзя подтвердить причину. "
                            "Нужны дополнительные фотографии и осмотр специалиста.", confidence)
    if request.plant_part == "flower" and label != "gray_mold":
        return insufficient("Для корзинки недостаточно подтверждения: получен класс, относящийся к листьям. "
                            "Приложите отдельный снимок листа или обратитесь к специалисту.", confidence)
    return _disease_response(request, config, label, confidence)


def _healthy_response(request, config, confidence, label):
    """Зелёный статус «здоров» с оговоркой; описание повреждений в тексте его отменяет."""
    if config["id"] == "corn":
        has_damage = reported_damage(request.symptoms)
    else:
        has_damage = DAMAGE_WORDS.search(request.symptoms or "")
    organ = "початка" if label == "healthy_ear" else "листа"
    if has_damage:
        return insufficient(f"Фото похоже на класс здорового {organ}, но описание содержит признаки повреждения. "
                            "Нужен крупный план изменённого участка.", confidence)
    if label == "healthy_ear":
        observation = "Снимок похож на початки без видимых признаков головни и гнили."
        tips = list(EAR_PHOTO_TIPS[:2])
    else:
        observation = "Снимок похож на изображения здоровых листьев выбранной культуры."
        tips = list(LEAF_PHOTO_TIPS) if config["id"] == "corn" else []
    result = DiagnosisResponse(
        status="healthy", overall_confidence=confidence, assessment_basis="model", model_label=label,
        visual_observations=observation,
        human_explanation="В рамках поддерживаемых классов признаков болезни не обнаружено. "
                          "Другие болезни и скрытые повреждения не исключены; это оценка по фото, а не осмотр.",
        recommendations=["Наблюдайте за растением; при появлении изменений сделайте новый снимок.", *tips])
    if config["id"] == "corn" and label == "healthy":
        result.model_observation = "healthy_leaf_like"
    return apply_crop_display_policy(result, request.crop)


def _disease_response(request, config, label, confidence):
    name = config["labels"].get(label) or (CORN_LABELS.get(label) if config["id"] == "corn" else None)
    if name is None:
        return insufficient("Получена неподдерживаемая категория. Нужен осмотр специалиста.")
    recommendations, sources = (care_for(label) if config["id"] == "corn" else (GENERIC_RECOMMENDATIONS, []))
    explanation = f"Предварительная гипотеза: {name}. Оценка модели не является подтверждённым диагнозом."
    if request.plant_part == "flower":
        explanation += (" В наборе нет отдельного класса здоровых корзинок: результат может зависеть "
                        "от вида органа, а не только от поражения.")
    if label == "blight":
        explanation += (" Ожогоподобные пятна не подтверждают солнечный или химический ожог "
                        "и не различают северный и южный гельминтоспориоз.")
    if label == "leaf_scars":
        explanation += " Метка Leaf scars описывает повреждения и не устанавливает возбудителя."
    if label == "ear_rot":
        explanation += (" Вид гнили по фото не определяется: фузариоз, гиббереллёз и аспергиллёз "
                        "различаются только при осмотре и лабораторном анализе.")
    return DiagnosisResponse(
        status="success", overall_confidence=confidence, assessment_basis="model", model_label=label,
        visual_observations=f"Изображение похоже на класс «{name}» в наборе выбранной культуры.",
        probable_causes=[ProbableCause(
            name=name, category="неопределено" if label == "leaf_scars" else "болезнь", probability=confidence,
            reasoning="Результат классификации фотографии моделью выбранной культуры; требуется проверка при осмотре.")],
        human_explanation=explanation, recommendations=recommendations, sources=sources)
