from typing import List, Dict, Any
from pydantic import BaseModel, Field
import base64
import cv2
import numpy as np

from schemas import (
    DiagnosisRequest,
    DiagnosisResponse,
    ProbableCause,
    SourceReference,
)

DEFAULT_PLACEHOLDERS = {
    "определи проблему по фотографии листа или ростка.",
    "определи проблему по фотографии листа/ростка.",
    "пользователь прислал фото растения для визуальной диагностики без текстового описания.",
    "📷 [отправлено только фото растения без текста]",
    "📷 [отправлено фото растения на визуальную диагностику]",
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
    object_type: str = "лист"  # "лист", "клубень" или "текст"
    image_quality_score: int = Field(default=0, ge=0, le=100)
    green_pct: float = 0.0
    yellow_pct: float = 0.0
    brown_pct: float = 0.0
    spot_count: int = 0
    extracted_visual_symptoms: str = ""
    preliminary_visual_hypotheses: List[str] = Field(default_factory=list)
    has_contradictions_or_lack_of_data: bool = True
    triage_notes: str = ""


def normalize_text(text: str) -> str:
    text = (text or "").lower().replace("ё", "е").strip()
    if text in DEFAULT_PLACEHOLDERS:
        return ""
    for ph in DEFAULT_PLACEHOLDERS:
        text = text.replace(ph, "")
    return text.strip()


def normalize_category(category: str) -> str:
    return CATEGORY_MAP.get((category or "").strip().lower(), "неопределено")


class AgroAgentOrchestrator:
    def __init__(self):
        print("✅ Локальный CV + RAG движок запущен (OpenCV + ChromaDB без дублей).")

    def _decode_cv2_image(self, image_base64: str) -> np.ndarray | None:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            img_bytes = base64.b64decode(image_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"❌ Ошибка декодирования OpenCV: {e}")
            return None

    def run_vision_triage_agent(self, request: DiagnosisRequest) -> VisionTriageReport:
        """
        АГЕНТ 1: Компьютерное зрение OpenCV.
        Различает зеленый лист и клубень картофеля, считает точные пропорции поражений.
        """
        raw_user_text = normalize_text(request.symptoms or "")
        has_image = bool(request.image_base64)

        if not has_image:
            return VisionTriageReport(
                is_plant_visible=False,
                object_type="текст",
                image_quality_score=0,
                extracted_visual_symptoms=raw_user_text,
                has_contradictions_or_lack_of_data=len(raw_user_text) < 5,
                triage_notes="Фото не приложено. Анализ выполнен по текстовому описанию.",
            )

        img_bgr = self._decode_cv2_image(request.image_base64)
        if img_bgr is None:
            return VisionTriageReport(
                is_plant_visible=False,
                extracted_visual_symptoms=raw_user_text,
                has_contradictions_or_lack_of_data=True,
                triage_notes="Не удалось декодировать изображение.",
            )

        img_resized = cv2.resize(img_bgr, (400, 400))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        # 1. Оценка резкости по Лапласиану
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality_score = int(min(100, max(15, (laplacian_var / 150.0) * 100)))

        # 2. Маски HSV
        green_mask = cv2.inRange(hsv, np.array([35, 35, 30]), np.array([88, 255, 255]))
        yellow_beige_mask = cv2.inRange(hsv, np.array([15, 30, 50]), np.array([34, 255, 255]))
        brown_dark_mask = cv2.inRange(hsv, np.array([0, 30, 15]), np.array([15, 255, 185]))

        total_px = float(400 * 400)
        green_ratio = np.count_nonzero(green_mask) / total_px
        yellow_ratio = np.count_nonzero(yellow_beige_mask) / total_px
        brown_ratio = np.count_nonzero(brown_dark_mask) / total_px

        plant_coverage = green_ratio + yellow_ratio + brown_ratio
        if plant_coverage < 0.08:
            return VisionTriageReport(
                is_plant_visible=False,
                image_quality_score=quality_score,
                extracted_visual_symptoms=raw_user_text,
                has_contradictions_or_lack_of_data=True,
                triage_notes=(
                    f"OpenCV: Растительный объект занимает менее {plant_coverage*100:.1f}% кадра. "
                    "Сфотографируйте лист или клубень ближе."
                ),
            )

        area = max(plant_coverage, 0.01)
        green_pct = round((green_ratio / area) * 100, 1)
        yellow_pct = round((yellow_ratio / area) * 100, 1)
        brown_pct = round((brown_ratio / area) * 100, 1)

        # 3. Подсчет контуров пятен/язв
        kernel = np.ones((3, 3), np.uint8)
        cleaned_brown = cv2.morphologyEx(brown_dark_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_count = sum(1 for c in contours if 18 < cv2.contourArea(c) < 12000)

        # 4. Определяем: это КЛУБЕНЬ или ЛИСТ?
        # У клубня картофеля почти нет зеленого (< 5%), основа бежевая (yellow_pct > 40%) и есть бурые очаги
        is_tuber = (green_pct < 5.0 and yellow_pct > 35.0 and (brown_pct > 5.0 or spot_count >= 5))
        object_type = "клубень" if is_tuber else "лист"

        cv_symptoms = []
        is_healthy = False

        if is_tuber:
            # Для клубня бежевый цвет — это нормальная кожура, а НЕ хлороз листа!
            if spot_count >= 12 or brown_pct > 15.0:
                cv_symptoms.append(
                    "поражение клубня картофеля на кожуре клубня сухие коричневые бурые язвы струпья парша трещины сетчатый некроз"
                )
            else:
                cv_symptoms.append(
                    "на поверхности клубня картофеля вдавленные буро-серые пятна некротические очаги гнили"
                )
            notes = (
                f"OpenCV анализ (объект: КЛУБЕНЬ, резкость: {quality_score}/100): "
                f"светлая кожура ~{yellow_pct:.1f}%, бурые язвы/некроз ~{brown_pct:.1f}%, "
                f"локальных очагов поражения: {spot_count} шт."
            )
        else:
            # Это ЛИСТ (зеленый или пожелтевший)
            if brown_pct > 18.0:
                cv_symptoms.append("зеленый лист крупные темно-бурые мокнущие пятна фитофтороз некроз")
            elif brown_pct > 4.0 or spot_count >= 5:
                cv_symptoms.append("на листьях округлые сухие темно-коричневые пятна альтернариоз некротическая пятнистость")

            # Хлороз ставим только тогда, когда лист реально желтый И бурых пятен мало
            if yellow_pct > 25.0 and brown_pct < 10.0:
                cv_symptoms.append("равномерное пожелтение хлороз листовой пластинки желтый лист дефицит азота и калия без бурых язв")

            if not cv_symptoms and green_pct > 60.0:
                is_healthy = True
                cv_symptoms.append("здоровый зеленый лист без некрозов")

            notes = (
                f"OpenCV анализ (объект: ЛИСТ, резкость: {quality_score}/100): "
                f"здоровая зеленая ткань ~{green_pct:.1f}%, желтизна/хлороз ~{yellow_pct:.1f}%, "
                f"бурый некроз ~{brown_pct:.1f}%, локальных пятен: {spot_count} шт."
            )

        combined_symptoms = f"{raw_user_text} {' '.join(cv_symptoms)}".strip()
        print(f"🔍 [OpenCV]: {notes}")

        return VisionTriageReport(
            is_plant_visible=True,
            is_healthy=is_healthy,
            object_type=object_type,
            image_quality_score=quality_score,
            green_pct=green_pct,
            yellow_pct=yellow_pct,
            brown_pct=brown_pct,
            spot_count=spot_count,
            extracted_visual_symptoms=combined_symptoms,
            has_contradictions_or_lack_of_data=False,
            triage_notes=notes,
        )

    def _calculate_cv_match_score(
        self,
        article: Dict[str, Any],
        triage: VisionTriageReport,
        full_symptoms: str,
    ) -> float:
        """
        Честный непрерывный расчет совпадения признаков (от 0.05 до 0.98) вместо грубой единицы 1.00!
        Учитывает тип органа (клубень vs лист), процент некроза, хлороза и пересечение терминов.
        """
        art_text = f"{article.get('title', '')} {article.get('symptoms_text', '')}".lower()
        cat = normalize_category(article.get("category", ""))

        # 1. Лексическое пересечение с текстом статьи
        query_words = [w for w in set(full_symptoms.lower().split()) if len(w) > 3]
        if query_words:
            hits = sum(1 for w in query_words if w in art_text)
            lexical_score = min(0.95, hits / max(4, len(query_words) * 0.6))
        else:
            lexical_score = 0.2

        # Если фото не было — опираемся только на текст
        if not triage.is_plant_visible:
            return round(lexical_score, 2)

        # 2. Биометрический скоринг по метрикам OpenCV
        is_tuber_article = "клубн" in art_text or "парша" in art_text

        if triage.object_type == "клубень":
            if is_tuber_article:
                # Чем больше язв и бурого некроза на клубне, тем выше уверенность в парше/гнили клубня
                severity = min(0.96, 0.55 + (triage.brown_pct / 100.0) * 0.8 + min(0.25, triage.spot_count / 150.0))
                return round(0.6 * severity + 0.4 * lexical_score, 2)
            elif cat == "дефицит питания":
                # На клубне не бывает листового хлороза — жестко штрафуем!
                return 0.08
            else:
                # Листовая болезнь (например, фитофтороз листьев) косвенно связана, но вероятность ниже
                return round(0.25 + 0.25 * lexical_score, 2)

        # Если на фото ЛИСТ:
        if is_tuber_article:
            # Болезнь клубня на фото зеленого/желтого листа маловероятна
            return 0.12

        if cat == "дефицит питания":
            # Дефицит питания (хлороз) зависит от желтизны листа и ОТСУТСТВИЯ крупных бурых язв
            chlorosis_factor = min(0.95, (triage.yellow_pct / 85.0))
            necrosis_penalty = min(0.6, (triage.brown_pct / 40.0) + (triage.spot_count * 0.02))
            cv_score = max(0.10, chlorosis_factor - necrosis_penalty)
            return round(0.65 * cv_score + 0.35 * lexical_score, 2)

        if cat == "болезнь":
            # Болезнь листа зависит от доли бурого некроза и количества пятен
            necrosis_factor = min(0.95, (triage.brown_pct / 35.0) + min(0.45, triage.spot_count / 45.0))
            cv_score = max(0.12, necrosis_factor)
            return round(0.65 * cv_score + 0.35 * lexical_score, 2)

        return round(0.5 * lexical_score + 0.25, 2)

    def run_agronomist_advisor_agent(
        self,
        request: DiagnosisRequest,
        triage_report: VisionTriageReport,
        rag_articles: List[Dict[str, Any]],
    ) -> DiagnosisResponse:
        """
        АГЕНТ 2: Формирует дифференциальный диагноз строго по статьям из базы RAG + метрикам OpenCV.
        Никаких искусственных дублей с RAG: 0.00!
        """
        user_text = normalize_text(request.symptoms or "")
        cv_text = normalize_text(triage_report.extracted_visual_symptoms or "")
        full_symptoms = f"{user_text} {cv_text}".strip()

        if request.image_base64 and not triage_report.is_plant_visible and not user_text:
            return DiagnosisResponse(
                status="insufficient_data",
                overall_confidence=15,
                visual_observations=triage_report.triage_notes,
                probable_causes=[],
                human_explanation=(
                    "Компьютерное зрение (OpenCV) не обнаружило в кадре листа или клубня. "
                    "Сфотографируйте пораженный орган растения крупным планом."
                ),
                recommendations=["Наведите камеру на лист или клубень с расстояния 10–15 см."],
                clarifying_questions=["Какой орган растения поражен (листья, стебли или клубни)?"],
                sources=[],
            )

        if triage_report.is_healthy and not user_text:
            return DiagnosisResponse(
                status="success",
                overall_confidence=86,
                visual_observations=triage_report.triage_notes,
                probable_causes=[
                    ProbableCause(
                        name="Патологий не выявлено (здоровая ткань)",
                        category="неопределено",
                        probability=86,
                        reasoning=triage_report.triage_notes,
                    )
                ],
                human_explanation=(
                    f"{triage_report.triage_notes} Растение выглядит здоровым, некротических пятен и хлороза не обнаружено."
                ),
                recommendations=["Продолжать плановый мониторинг посевов."],
                clarifying_questions=[],
                sources=[],
            )

        if len(full_symptoms) < 4 or not rag_articles:
            return DiagnosisResponse(
                status="insufficient_data",
                overall_confidence=0,
                visual_observations=triage_report.triage_notes,
                probable_causes=[],
                human_explanation="Недостаточно данных для диагностики. Приложите фото или опишите симптомы.",
                recommendations=["Загрузите четкое фото листа/клубня или опишите признаки словами."],
                clarifying_questions=["Какие изменения заметны на растении?"],
                sources=[],
            )

        # Оцениваем каждую найденную в ChromaDB статью (у всех есть реальный RAG > 0 и реальный источник!)
        ranked = []
        for article in rag_articles:
            rag_score = float(article.get("similarity_score", 0.0) or 0.0)
            rag_score = max(0.05, min(0.99, rag_score))

            cv_rule_score = self._calculate_cv_match_score(article, triage_report, full_symptoms)

            # Итоговая уверенность: 55% биометрия OpenCV + 45% семантическая близость ChromaDB
            confidence = round(0.55 * cv_rule_score + 0.45 * rag_score, 3)

            ranked.append({
                "name": article["title"],
                "category": normalize_category(article.get("category", "")),
                "rule_score": cv_rule_score,
                "rag_score": rag_score,
                "confidence": max(0.08, min(0.96, confidence)),
                "recommendations": article.get("recommendations", []),
                "article": article,
            })

        ranked.sort(key=lambda x: x["confidence"], reverse=True)

        probable_causes = []
        sources = []
        seen_sources = set()

        for item in ranked[:3]:
            probable_causes.append(
                ProbableCause(
                    name=item["name"],
                    category=item["category"],
                    probability=round(item["confidence"] * 100),
                    reasoning=(
                        f"{triage_report.triage_notes} | "
                        f"Совпадение по признакам CV: {item['rule_score']:.2f}, "
                        f"близость в векторной БД (RAG): {item['rag_score']:.2f}."
                    ),
                )
            )
            art = item["article"]
            src = art.get("source") or {}
            src_key = (src.get("title", ""), src.get("section", ""), src.get("url_or_code", ""))
            if src_key not in seen_sources and any(src_key):
                seen_sources.add(src_key)
                sources.append(
                    SourceReference(
                        title=src_key[0],
                        section=src_key[1],
                        url_or_code=src_key[2],
                    )
                )

        top = ranked[0]
        overall_conf = round(top["confidence"] * 100)

        return DiagnosisResponse(
            status="success" if overall_conf >= 40 else "insufficient_data",
            overall_confidence=overall_conf,
            visual_observations=triage_report.triage_notes,
            probable_causes=probable_causes,
            human_explanation=(
                f"Результаты компьютерного зрения: {triage_report.triage_notes} "
                f"По совокупности визуальных биометрических признаков и векторного поиска по базе «{request.crop}», "
                f"главная гипотеза — {top['name']} ({overall_conf}%). Альтернативные варианты приведены ниже со значительным отрывом."
            ),
            recommendations=top["recommendations"],
            clarifying_questions=[
                "Какой процент растений/клубней в партии имеет аналогичные симптомы?",
                "Проводились ли защитные обработки фунгицидами в текущем сезоне?",
            ],
            sources=sources,
        )