from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON, Index
from sqlalchemy.sql import func
from database import Base


class KnowledgeArticle(Base):
    """Таблица официального справочника болезней, вредителей и дефицитов питания"""
    __tablename__ = "knowledge_articles"

    id = Column(String, primary_key=True, index=True)  # Например: 'wheat_septoria_01'
    crop = Column(String, nullable=False, index=True)  # B-Tree индекс для быстрой фильтрации по культуре
    category = Column(String, nullable=False, index=True)  # 'болезнь', 'вредитель', 'дефицит питания'
    title = Column(String, nullable=False)
    growth_stages = Column(JSON, nullable=False)  # Список фаз: ["Кущение", "Выход в трубку"]
    symptoms_text = Column(Text, nullable=False)
    recommendations = Column(JSON, nullable=False)  # Список шагов лечения
    source_title = Column(String, nullable=False)
    source_section = Column(String, nullable=False)
    source_code = Column(String, nullable=False)

    # Составной индекс для мгновенного поиска по культуре и категории
    __table_args__ = (
        Index('idx_crop_category', 'crop', 'category'),
    )


class CheckHistory(Base):
    """История обращений агронома"""
    __tablename__ = "check_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    crop = Column(String, nullable=False, index=True)
    growth_stage = Column(String, nullable=False)
    region = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)
    image_base64 = Column(Text, nullable=True)

    # Результаты работы ИИ-агентов
    status = Column(String, nullable=False)  # 'success' или 'insufficient_data'
    overall_confidence = Column(Integer, nullable=False)
    result_json = Column(JSON, nullable=False)  # Полный структурированный ответ
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Feedback(Base):
    """Обратная связь от пользователя по конкретной проверке"""
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    check_id = Column(Integer, ForeignKey("check_history.id"), nullable=False, index=True)
    is_helpful = Column(Boolean, nullable=False)
    user_comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())