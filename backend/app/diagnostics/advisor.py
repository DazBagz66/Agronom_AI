"""Объединяем фото, описание и статьи в предварительный вывод."""
import re
from typing import List, Dict, Any

from app.diagnostics.policy import apply_diagnosis_policy
from app.schemas import DiagnosisRequest, DiagnosisResponse, ProbableCause
from app.diagnostics.symptoms import disease_key, DISEASE_NAMES, symptom_evidence, positive_symptoms
from app.diagnostics.vision import VisionTriageReport, analyze_image, decode_image

DEFAULT_PLACEHOLDERS = {
    "определи проблему по фотографии листа или ростка.",
    "определи проблему по фотографии листа/ростка.",
    "пользователь прислал фото растения для визуальной диагностики без текстового описания.",
    "📷 [отправлено только фото растения без текста]",
    "📷 [отправлено фото растения на визуальную диагностику]",
}


def normalize_text(text):
    """Убираем служебные подписи к фото, чтобы не считать их симптомами."""
    text = (text or "").casefold().replace("ё", "е").strip()
    for placeholder in DEFAULT_PLACEHOLDERS:
        text = text.replace(placeholder, "")
    return text.strip()


def normalize_category(category):
    return {"disease": "болезнь", "pest": "вредитель", "nutrition": "дефицит питания"}.get(category, category) if category in {
        "disease", "pest", "nutrition", "болезнь", "вредитель", "дефицит питания"} else "неопределено"


class PotatoAdvisor:
    _decode_cv2_image = staticmethod(decode_image)

    def analyze_photo(self, request):
        return analyze_image(request)

    def _calculate_cv_match_score(self, article, triage, full_symptoms):
        # Несмотря на имя функции, здесь проверяются слова, а не пиксели.
        words = set(re.findall(r"[а-яa-z]{4,}", positive_symptoms(full_symptoms)))
        if len(words) < 4:
            return 0.
        article_text = (article.get("title", "") + " " + article.get("symptoms_text", "")).casefold().replace("ё", "е")
        hits = sum(word[:5] in article_text for word in words)
        return min(.95, hits / max(4, len(words) * .7))

    def assess(self, request: DiagnosisRequest,
            triage_report: VisionTriageReport, rag_articles: List[Dict[str, Any]]) -> DiagnosisResponse:
        """Выбираем гипотезы по признакам и готовим понятное объяснение."""
        text = normalize_text(request.symptoms)
        triage = triage_report
        candidates = {}
        strongest_text = 0.
        for article in rag_articles:
            key = disease_key(article["title"])
            if key == "tuber_rot" and not (request.plant_part == "tuber" or triage.object_type == "клубень" or "клубн" in text):
                continue
            explicit_score, signs = symptom_evidence(text, key)
            strongest_text = max(strongest_text, explicit_score)
            retrieval = max(0., min(1., float(article.get("similarity_score", 0))))
            lexical = self._calculate_cv_match_score(article, triage, text)
            fallback = .55 * lexical + .45 * retrieval if text else 0.
            # Одного сходства со статьёй недостаточно для уверенной гипотезы.
            if key in DISEASE_NAMES:
                fallback = min(fallback, .64)
            visual = triage.class_probabilities.get(key, 0.)
            if triage.image_quality_score < 30:
                visual = 0.
            # Берём наиболее сильное свидетельство; проценты не складываем.
            score = max(explicit_score, fallback, visual)
            reasoning = []
            if explicit_score >= .65:
                reasoning.append("В описании совпали признаки: " + ", ".join(signs) + ".")
            if visual >= .65:
                reasoning.append(f"Модель для объекта «{triage.object_type}» нашла сходство с признаками этой болезни.")
            candidate = {"key": key, "score": score, "text_score": explicit_score,
                         "article": article, "reasoning": " ".join(reasoning)}
            if key not in candidates or candidates[key]["score"] < score:
                candidates[key] = candidate
        ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
        # Сообщённые симптомы могут противоречить внешне здоровому виду.
        has_reported_problem = bool(re.search(r"пят|налет|гнил|язв|вян|увяда|черне|желте|пожелт|трещин|мягк|запах", positive_symptoms(text)))
        if (triage.healthy_confidence >= .65 or triage.no_visible_damage) and not has_reported_problem and strongest_text < .65:
            return apply_diagnosis_policy(DiagnosisResponse(
                status="healthy", overall_confidence=int(triage.healthy_confidence * 100),
                assessment_basis="visual_observation" if triage.no_visible_damage else "model",
                visual_observations=triage.triage_notes,
                human_explanation=f"На фото ({triage.object_type}) видимых признаков болезни не обнаружено. "
                    "Это оценка видимой поверхности: скрытые повреждения по одному снимку исключить нельзя.",
                recommendations=["Продолжайте наблюдение за растением.",
                                 "Если изменения появятся, сделайте новый снимок при том же освещении."]))
        eligible = [item for item in ranked if item["score"] >= .65]
        # При конфликте фото и сильного описания явно указываем опору на текст.
        conflict = triage.class_probabilities.get("healthy", 0) >= .90 and strongest_text >= .80
        overall = int(min(.99, ranked[0]["score"]) * 100) if ranked else 0
        causes = [ProbableCause(name=DISEASE_NAMES.get(item["key"], item["article"]["title"]),
            category=normalize_category(item["article"].get("category", "болезнь")),
            probability=int(min(.99, item["score"]) * 100),
            reasoning=item["reasoning"] or "Совпали несколько признаков из описания.") for item in eligible[:3]]
        explanation = (f"Наиболее вероятная причина — {causes[0].name}. " + causes[0].reasoning) if causes else ""
        recommendations = eligible[0]["article"].get("recommendations", []) if eligible else []
        result = apply_diagnosis_policy(DiagnosisResponse(
            status="success" if causes else "insufficient_data", overall_confidence=overall,
            assessment_basis="model" if eligible and triage.class_probabilities.get(eligible[0]["key"], 0) >= eligible[0]["text_score"] else "symptoms",
            visual_observations=triage.triage_notes, probable_causes=causes,
            human_explanation=explanation, recommendations=recommendations,
            clarifying_questions=[] if causes else ["Какая часть картофеля изменилась и как быстро развиваются изменения?"]))
        if conflict:
            result.assessment_basis = "symptoms"
            result.human_explanation = (
                f"По описанию предварительная гипотеза — {causes[0].name}. "
                "Фото не подтверждает описанные признаки, поэтому гипотеза опирается на текст. "
                "Снимите изменённый участок отдельно для проверки."
            )
            result.clarifying_questions = ["Описанные изменения находятся на той же части растения, которая видна на фото?"]
        elif request.image_base64 and not triage.is_plant_visible and not text:
            result.human_explanation = triage.triage_notes
        elif request.image_base64 and not causes and triage.is_plant_visible:
            result.assessment_basis = "visual_observation"
            result.human_explanation = (
                f"{triage.triage_notes} По фото нельзя уверенно установить конкретную болезнь. "
                "Для уточнения опишите, остаётся ли ткань твёрдой, есть ли размягчение, запах или изменения на разрезе."
            )
            result.clarifying_questions = [
                "Пятно только на кожуре или уходит в мякоть?",
                "Клубень остаётся твёрдым или размягчается? Есть ли запах?",
            ] if triage.object_type == "клубень" else ["Какие изменения вы заметили и как быстро они появились?"]
        return result
