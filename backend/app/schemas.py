"""Формат данных, которыми обмениваются браузер и сервер."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# Входные данные от пользователя
class DiagnosisRequest(BaseModel):
    crop: Literal["Картофель"] = Field(default="Картофель", description="Поддерживается только картофель")
    growth_stage: str = Field(..., description="Фаза развития")
    symptoms: Optional[str] = Field(
        default=None,
        description="Описание симптомов растения"
    )
    image_base64: Optional[str] = Field(None, description="Фото растения в формате Base64")
    plant_part: Literal["auto", "leaf", "stem", "tuber", "whole"] = "auto"

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
class DiagnosisResponse(BaseModel):
    check_id: Optional[int] = None
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

