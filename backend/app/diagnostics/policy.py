"""Правило показа гипотез от 65%, включая старые записи истории."""

MIN_DIAGNOSIS_PROBABILITY = 65
from app.diagnostics.factors import possible_factors
INSUFFICIENT_EXPLANATION = (
    "Недостаточно данных: ни одна гипотеза не достигла порога 65%. "
    "Уточните симптомы и приложите чёткие фотографии поражённых частей картофеля."
)
NEXT_STEPS = [
    "Сфотографируйте поражённую часть крупным планом и растение целиком.",
    "Опишите цвет, форму пятен, наличие налёта и скорость развития симптомов.",
]


def apply_diagnosis_policy(result):
    """Не даём скрытой слабой гипотезе попасть в объяснение или рекомендации."""
    result.sources = []  # Происхождение статей остаётся внутри базы знаний.
    result.possible_factors = []
    if result.status == "healthy" and (result.overall_confidence >= MIN_DIAGNOSIS_PROBABILITY or result.assessment_basis == "visual_observation"):
        result.probable_causes = []
        return result
    result.probable_causes = [
        cause for cause in result.probable_causes
        if cause.probability >= MIN_DIAGNOSIS_PROBABILITY
    ]
    if not result.probable_causes or result.overall_confidence < MIN_DIAGNOSIS_PROBABILITY:
        result.probable_causes = []
        result.status = "insufficient_data"
        result.human_explanation = INSUFFICIENT_EXPLANATION
        result.recommendations = list(NEXT_STEPS)
        result.sources = []
        if not result.clarifying_questions:
            result.clarifying_questions = ["Какая часть картофеля поражена: листья, стебель или клубни?"]
    else:
        result.status = "success"
        result.possible_factors = possible_factors(result.probable_causes[0].name)
        if any(token in result.human_explanation for token in ("RAG", "OpenCV", "векторн", "базе «")):
            result.human_explanation = f"Предположительная причина — {result.probable_causes[0].name}."
        for cause in result.probable_causes:
            if any(token in cause.reasoning for token in ("RAG", "CV:", "OpenCV", "векторн")):
                cause.reasoning = "Сопоставлены видимые изменения и описание симптомов."
    return result
