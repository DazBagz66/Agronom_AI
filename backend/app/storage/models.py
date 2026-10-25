"""Описание таблиц базы данных. Это не модели нейросетей."""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.sql import func
from app.storage.database import Base


class KnowledgeArticle(Base):
    """Карточка болезни: симптомы, действия и происхождение текста."""
    __tablename__ = "knowledge_articles"

    id = Column(String, primary_key=True, index=True)  # Устойчивый ID: повторный импорт обновляет ту же карточку.
    crop = Column(String, nullable=False, index=True)  # Индекс ускоряет фильтрацию.
    category = Column(String, nullable=False, index=True)  # 'болезнь', 'вредитель', 'дефицит питания'
    title = Column(String, nullable=False)
    growth_stages = Column(JSON, nullable=False)  # Например: ["Цветение", "Уборка и хранение"].
    symptoms_text = Column(Text, nullable=False)
    recommendations = Column(JSON, nullable=False)  # Список рекомендуемых действий.
    source_title = Column(String, nullable=False)
    source_section = Column(String, nullable=False)
    source_code = Column(String, nullable=False)

    # Индекс для поиска сразу по культуре и категории.
    __table_args__ = (
        Index('idx_crop_category', 'crop', 'category'),
    )


class CheckHistory(Base):
    """Входные данные проверки и сохранённый ответ."""
    __tablename__ = "check_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    crop = Column(String, nullable=False, index=True)
    growth_stage = Column(String, nullable=False)
    region = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)
    image_base64 = Column(Text, nullable=True)

    # Сохраняем итог, чтобы историю можно было открыть без повторного анализа.
    status = Column(String, nullable=False)  # success, healthy или insufficient_data.
    overall_confidence = Column(Integer, nullable=False)
    result_json = Column(JSON, nullable=False)  # Полный структурированный ответ
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class KnowledgeFetch(Base):
    """Когда и откуда получена последняя версия статьи."""
    __tablename__ = "knowledge_fetches"
    article_id = Column(String, ForeignKey("knowledge_articles.id"), primary_key=True)
    source_url = Column(String, nullable=False)
    fetched_at = Column(String, nullable=False)
    content_sha256 = Column(String, nullable=False)
    source_updated = Column(String, nullable=True)
    license = Column(String, nullable=False)


class DiagnosisFeedback(Base):
    """Append-only user reports awaiting independent review, never training labels."""
    __tablename__ = 'diagnosis_feedback'
    id = Column(Integer, primary_key=True, autoincrement=True)
    check_id = Column(Integer, ForeignKey('check_history.id'), nullable=False, index=True)
    proposed_labels = Column(JSON, nullable=False)
    plant_part = Column(String, nullable=False)
    notes = Column(Text, nullable=False)
    specimen_group = Column(String, nullable=False)
    reviewer_status = Column(String, nullable=False, default='pending')
    model_snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class InvestigationSession(Base):
    """An ongoing examination; observations remain immutable history records."""
    __tablename__ = 'investigation_sessions'
    id = Column(String, primary_key=True)
    crop = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class SessionObservation(Base):
    __tablename__ = 'session_observations'
    check_id = Column(Integer, ForeignKey('check_history.id'), primary_key=True)
    session_id = Column(String, ForeignKey('investigation_sessions.id'), nullable=False, index=True)
