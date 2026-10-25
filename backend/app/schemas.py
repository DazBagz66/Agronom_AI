"""Формат данных, которыми обмениваются браузер и сервер."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# Входные данные от пользователя
class DiagnosisRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=36)
    crop: Literal["Кукуруза", "Подсолнечник", "Картофель"] = Field(default="Кукуруза", description="Кукуруза и подсолнечник; картофель сохранён для совместимости")
    growth_stage: str = Field(..., description="Фаза развития")
    symptoms: Optional[str] = Field(
        default=None,
        description="Описание симптомов растения"
    )
    image_base64: Optional[str] = Field(None, description="Фото растения в формате Base64")
    plant_part: Literal["auto", "leaf", "ear", "flower", "stem", "tuber", "whole"] = "leaf"


class DiagnosisFeedbackRequest(BaseModel):
    proposed_labels: List[str] = Field(min_length=1, max_length=4)
    plant_part: Literal['leaf', 'ear', 'stem', 'whole', 'unknown'] = 'unknown'
    notes: str = Field(default='', max_length=2000)
    specimen_group: str = Field(default='', max_length=100)

# Отдельная гипотеза (причина)
class ProbableCause(BaseModel):
    name: str = Field(..., description="Название болезни, вредителя или дефицита")
    category: Literal["болезнь", "вредитель", "дефицит питания", "неопределено"]
    probability: int = Field(..., ge=0, le=100, description="Вероятность конкретной причины в %")
    reasoning: str = Field(..., description="Почему ИИ так решил (связь фото и симптомов)")

# Происхождение статьи; в пользовательском ответе источники скрыты.
class SourceReference(BaseModel):
    title: str = Field(..., description="Название документа / справочника")
    section: str = Field(..., description="Раздел или регламент")
    url_or_code: str = Field(..., description="Ссылка или номер в Госкаталоге")

# Итог анализа, который получает интерфейс.
class KnowledgeMatch(BaseModel):
    article_id: str
    title: str
    symptoms_text: str
    recommendations: List[str]
    source: SourceReference
    additional_sources: List[SourceReference] = Field(default_factory=list)
    basis: Literal['model', 'symptoms', 'organ']


class DiagnosisResponse(BaseModel):
    session_id: Optional[str] = None
    context_observations: List[str] = Field(default_factory=list)
    symptom_assessment: Optional[str] = None
    symptom_hypotheses: List[str] = Field(default_factory=list)
    experimental_observation: Optional[str] = None
    experimental_model_sha256: Optional[str] = None
    previous_photo_assessment: Optional[str] = None
    check_id: Optional[int] = None
    model_label: Optional[str] = None
    detected_plant_part: Optional[Literal['leaf','ear','whole','unknown']] = None
    organ_detection_method: Optional[Literal['automatic']] = None
    organ_confidence: Optional[int] = Field(default=None, ge=0, le=100)
    organ_model_sha256: Optional[str] = None
    knowledge_matches: List[KnowledgeMatch] = Field(default_factory=list)
    model_observation: Optional[Literal['healthy_leaf_like']] = None
    status: Literal["success", "healthy", "insufficient_data"] = Field(
        ..., description="Статус: диагноз поставлен или данных недостаточно"
    )
    overall_confidence: int = Field(..., ge=0, le=100, description="Общий уровень уверенности (0-100%)")
    visual_observations: str = Field(..., description="Что именно ИИ увидел на фото (или почему фото плохое)")
    probable_causes: List[ProbableCause] = Field(default_factory=list)
    human_explanation: str = Field(
        ..., description="Понятное объяснение для агронома/фермера: что происходит и что делать по шагам"
    )
    recommendations: List[str] = Field(default_factory=list, description="Конкретные шаги действий")
    clarifying_questions: List[str] = Field(
        default_factory=list, description="Вопросы пользователю, если данных не хватило"
    )
    sources: List[SourceReference] = Field(default_factory=list, description="Источники из RAG")
    possible_factors: List[str] = Field(default_factory=list)
    assessment_basis: Literal["model", "symptoms", "visual_observation"] = "symptoms"
    created_at: Optional[datetime] = None

