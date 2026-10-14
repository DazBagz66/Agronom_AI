from typing import List, Dict, Any
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image


from schemas import (
    DiagnosisRequest,
    DiagnosisResponse,
    ProbableCause,
    SourceReference,
)


SYMPTOM_RULES = {
    "Дефицит азота": {
        "category": "дефицит питания",
        "keywords": [
            "желтеют",
            "желтые листья",
            "пожелтение",
            "хлороз",
            "слабый рост",
            "медленный рост",
            "нижние листья",
        ],
        "recommendations": [
            "Проверить содержание азота в почве.",
            "Оценить необходимость внесения удобрений "
            "по результатам анализа почвы.",
        ],
    },
    "Фитофтороз": {
        "category": "болезнь",
        "keywords": [
            "бурые пятна",
            "темные пятна",
            "гниль",
            "увядание",
            "поражение листьев",
        ],
        "recommendations": [
            "Осмотреть растения на характерные признаки заболевания.",
            "Проверить распространение симптомов по полю.",
            "Подбирать средства защиты только после уточнения диагноза.",
        ],
    },
    "Повреждение вредителями": {
        "category": "вредитель",
        "keywords": [
            "дырки",
            "объеденные листья",
            "личинки",
            "насекомые",
            "жуки",
        ],
        "recommendations": [
            "Осмотреть листья с обеих сторон.",
            "Проверить наличие вредителей и оценить их численность.",
        ],
    },
    "Грибковое заболевание": {
        "category": "болезнь",
        "keywords": [
            "налет",
            "белый налет",
            "пятна",
            "грибок",
        ],
        "recommendations": [
            "Осмотреть пораженные участки растений.",
            "Уточнить возможного возбудителя перед выбором обработки.",
        ],
    },
}


CATEGORY_MAP = {
    "disease": "болезнь",
    "болезнь": "болезнь",
    "pest": "вредитель",
    "вредитель": "вредитель",
    "nutrition": "дефицит питания",
    "дефицит питания": "дефицит питания",
    "unknown": "неопределено",
    "неопределено": "неопределено",
}


class VisionTriageReport(BaseModel):
    is_plant_visible: bool = False
    is_healthy: bool = False
    image_quality_score: int = Field(default=0, ge=0, le=100)
    extracted_visual_symptoms: str = ""
    preliminary_visual_hypotheses: List[str] = Field(
        default_factory=list
    )
    has_contradictions_or_lack_of_data: bool = True
    triage_notes: str = ""


def normalize_text(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")

    replacements = {
        "желтеют": "желтые листья",
        "желтеет": "желтые листья",
        "пожелтели": "пожелтение",
        "вянут": "увядание",
        "сохнут": "сухость",
        "пятнышки": "пятна",
        "темнеют": "темные",
        "объедены": "объеденные",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def calculate_rule_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0

    matches = sum(1 for word in keywords if word in text)

    # Одного случайного совпадения недостаточно,
    # чтобы считать диагноз подтвержденным.
    return matches / len(keywords)


def normalize_category(category: str) -> str:
    return CATEGORY_MAP.get(
        (category or "").strip().lower(),
        "неопределено",
    )


class AgroAgentOrchestrator:

    def __init__(self):
        print(
            "Локальный экспертный движок запущен. "
            "Внешние LLM/VLM отключены."
        )

    def run_vision_triage_agent(
            self,
            request: DiagnosisRequest,
    ) -> VisionTriageReport:
        symptoms = (request.symptoms or "").strip()
        has_image = bool(request.image_base64)

        extracted_symptoms = symptoms
        image_analysis_notes = ""

        # ЛОКАЛЬНЫЙ АНАЛИЗ ПИКСЕЛЕЙ КАРТИНКИ (Без внешних API!)
        if has_image:
            try:
                # Декодируем base64 картинку
                img_data = request.image_base64
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                image_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")

                # Уменьшаем для быстрого анализа
                img_thumb = img.resize((100, 100))
                pixels = list(img_thumb.getdata())

                # Считаем средние значения RGB и отклонения (желтизна / темнота)
                total_pixels = len(pixels)
                yellow_ish = 0
                dark_spots = 0

                for r, g, b in pixels:
                    # Признак желтизны (много красного и зеленого, мало синего)
                    if r > 120 and g > 120 and b < 90:
                        yellow_ish += 1
                    # Признак темных/бурых пятен (низкая яркость)
                    elif r < 80 and g < 80 and b < 80:
                        dark_spots += 1

                yellow_pct = (yellow_ish / total_pixels) * 100
                dark_pct = (dark_spots / total_pixels) * 100

                image_analysis_notes = f"Анализ пикселей фото: желтизна/хлороз ~{yellow_pct:.1f}%, темные пятна/некроз ~{dark_pct:.1f}%."

                # Автоматически формируем симптомы по пикселям, если текст не был указан
                auto_symptoms = []
                if dark_pct > 12:
                    auto_symptoms.append("темные пятна бурые пятна поражение листьев")
                if yellow_pct > 15:
                    auto_symptoms.append("желтые листья пожелтение хлороз")

                if auto_symptoms and (not symptoms or len(symptoms) < 10):
                    extracted_symptoms = " ".join(auto_symptoms)

            except Exception as e:
                image_analysis_notes = f"Ошибка обработки изображения: {str(e)}"

        no_description = not extracted_symptoms or len(extracted_symptoms.strip()) < 5

        return VisionTriageReport(
            is_plant_visible=has_image,
            is_healthy=False,
            image_quality_score=80 if has_image else 0,
            extracted_visual_symptoms=extracted_symptoms,
            preliminary_visual_hypotheses=[],
            has_contradictions_or_lack_of_data=no_description,
            triage_notes=image_analysis_notes or "Использовано текстовое описание или локальный анализ фото.",
        )

    def run_agronomist_advisor_agent(
        self,
        request: DiagnosisRequest,
        triage_report: VisionTriageReport,
        rag_articles: List[Dict[str, Any]],
    ) -> DiagnosisResponse:

        symptoms = normalize_text(
            request.symptoms or ""
        )

        # Если нет текстовых симптомов, не выдаем диагноз
        # только на основании наличия фотографии.
        if len(symptoms) < 5 and not triage_report.extracted_visual_symptoms:
            return DiagnosisResponse(
                status="insufficient_data",
                overall_confidence=0,
                visual_observations=(
                    "Локальный движок не умеет анализировать "
                    "пиксели фотографии. Нужны текстовые симптомы."
                ),
                probable_causes=[],
                human_explanation=(
                    "Опишите наблюдаемые признаки: цвет и форму "
                    "пятен, состояние краев листьев, наличие налета "
                    "или следов вредителей."
                ),
                recommendations=[
                    "Добавьте текстовое описание симптомов.",
                    "При необходимости обратитесь к агроному.",
                ],
                clarifying_questions=[
                    "Какие изменения заметны на листьях?",
                    "На каких частях растения появились симптомы?",
                ],
                sources=[],
            )

        candidates = {}

        # 1. Сопоставление с локальными правилами.
        for name, rule in SYMPTOM_RULES.items():
            score = calculate_rule_score(
                symptoms,
                rule["keywords"],
            )

            if score > 0:
                candidates[name] = {
                    "name": name,
                    "category": rule["category"],
                    "rule_score": score,
                    "rag_score": 0.0,
                    "recommendations": rule["recommendations"],
                    "article": None,
                }

        # 2. Объединение результатов с RAG.
        for article in rag_articles or []:
            name = article.get("title")

            if not name:
                continue

            similarity = float(
                article.get("similarity_score", 0.0) or 0.0
            )
            similarity = max(0.0, min(1.0, similarity))

            category = normalize_category(
                article.get("category", "")
            )

            if name in candidates:
                candidates[name]["rag_score"] = similarity
                candidates[name]["article"] = article
                candidates[name]["category"] = category
            else:
                candidates[name] = {
                    "name": name,
                    "category": category,
                    "rule_score": 0.0,
                    "rag_score": similarity,
                    "recommendations": article.get(
                        "recommendations", []
                    ),
                    "article": article,
                }

        # 3. Рейтинг гипотез.
        ranked = []

        for item in candidates.values():
            rule_score = item["rule_score"]
            rag_score = item["rag_score"]

            if rule_score > 0 and rag_score > 0:
                confidence = 0.6 * rag_score + 0.4 * rule_score
            elif rag_score > 0:
                confidence = rag_score
            else:
                confidence = rule_score

            item["confidence"] = max(
                0.0, min(1.0, confidence)
            )
            ranked.append(item)

        ranked.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        if not ranked:
            return DiagnosisResponse(
                status="insufficient_data",
                overall_confidence=0,
                visual_observations=symptoms,
                probable_causes=[],
                human_explanation=(
                    "По заданным симптомам подходящие причины "
                    "не найдены в локальных правилах и справочнике."
                ),
                recommendations=[
                    "Уточните описание симптомов.",
                    "Проверьте культуру и фазу ее развития.",
                ],
                clarifying_questions=[
                    "Когда впервые появились симптомы?",
                    "Какие части растения поражены?",
                ],
                sources=[],
            )

        # 4. Структурированный ответ.
        probable_causes = []

        for item in ranked[:3]:
            probable_causes.append(
                ProbableCause(
                    name=item["name"],
                    category=item["category"],
                    probability=round(item["confidence"] * 100),
                    reasoning=(
                        "Совпадение с текстовыми признаками: "
                        f"{item['rule_score']:.2f}; "
                        "сходство со справочником: "
                        f"{item['rag_score']:.2f}. "
                        "Это оценка соответствия, а не "
                        "статистически калиброванная вероятность."
                    ),
                )
            )

        # 5. Источники из найденных статей.
        sources = []
        seen_sources = set()

        for item in ranked[:3]:
            article = item.get("article")

            if not article:
                continue

            source_data = article.get("source") or {}

            source_key = (
                source_data.get("title", ""),
                source_data.get("section", ""),
                source_data.get("url_or_code", ""),
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append(
                SourceReference(
                    title=source_key[0] or "Источник не указан",
                    section=source_key[1] or "Раздел не указан",
                    url_or_code=source_key[2] or "Ссылка не указана",
                )
            )

        top = ranked[0]

        recommendations = top["recommendations"] or [
            "Сверьте признаки с официальным агрономическим справочником.",
            "При сомнениях запросите очную консультацию специалиста.",
        ]

        return DiagnosisResponse(
            status="success",
            overall_confidence=round(top["confidence"] * 100),
            visual_observations=(
                "Автоматический анализ фотографии не выполнялся. "
                "Учтены только текстовые симптомы."
            ),
            probable_causes=probable_causes,
            human_explanation=(
                f"Наиболее подходящее совпадение: {top['name']}. "
                "Результат получен локальным сопоставлением "
                "симптомов и справочных статей. "
                "Перед обработкой растений проверьте диагноз."
            ),
            recommendations=recommendations,
            clarifying_questions=[
                "Когда появились первые симптомы?",
                "Поражены отдельные растения или все поле?",
            ],
            sources=sources,
        )
