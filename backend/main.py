from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
from fastapi.responses import FileResponse

from database import engine, Base, get_db, SessionLocal
from models import KnowledgeArticle, CheckHistory, Feedback
from schemas import DiagnosisRequest, DiagnosisResponse, FeedbackRequest
from rag_service import AgroRAGService
from agents import AgroAgentOrchestrator
from seed import seed_initial_knowledge


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Советник агронома API",
    description="Мультиагентный сервис диагностики растений с гибридным RAG (SQLite + ChromaDB) и поддержкой офлайн-режима",
    version="1.0.0"
)

# Разрешаем CORS для фронтенда и мобильного PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = AgroRAGService()
orchestrator = AgroAgentOrchestrator()


@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        seed_initial_knowledge(db, rag_service)
    finally:
        db.close()


@app.post("/api/diagnose", response_model=DiagnosisResponse)
def diagnose_plant(request: DiagnosisRequest, db: Session = Depends(get_db)):
    """
    Полный пайплайн:
    Обращение -> Агент 1 (Vision & Триаж) -> Векторный поиск ChromaDB -> Агент 2 (Диагноз, объяснение и источники) -> Сохранение в БД
    """
    # Шаг 1: Агент компьютерного зрения и первичного триажа
    triage_report = orchestrator.run_vision_triage_agent(request)

    # Шаг 2: Обогащенный запрос в векторную БД (текст пользователя + визуальные признаки с фото от Агента 1)
    enriched_query = f"{request.symptoms}. Визуально на фото: {triage_report.extracted_visual_symptoms}"
    rag_articles = rag_service.retrieve(
        db=db,
        crop=request.crop,
        growth_stage=request.growth_stage,
        query_symptoms=enriched_query,
        top_k=5
    )

    # Шаг 3: Агент-Агроном формирует ответ, скоринг уверенности и объяснение
    diagnosis_result = orchestrator.run_agronomist_advisor_agent(
        request=request,
        triage_report=triage_report,
        rag_articles=rag_articles
    )

    # Шаг 4: Сохраняем обращение и результат в историю проверок (SQLite)
    history_record = CheckHistory(
        crop=request.crop,
        growth_stage=request.growth_stage,
        region=request.region,
        symptoms=request.symptoms,
        image_base64=request.image_base64,
        status=diagnosis_result.status,
        overall_confidence=diagnosis_result.overall_confidence,
        result_json=diagnosis_result.model_dump(mode="json")
    )
    db.add(history_record)
    db.commit()
    db.refresh(history_record)

    diagnosis_result.check_id = history_record.id
    diagnosis_result.created_at = history_record.created_at
    return diagnosis_result


@app.get("/api/history")
def get_check_history(limit: int = 20, db: Session = Depends(get_db)):
    """Возвращает историю проверок вместе с оставленной обратной связью"""
    records = db.query(CheckHistory).order_by(CheckHistory.created_at.desc()).limit(limit).all()
    feedbacks = {f.check_id: f for f in db.query(Feedback).all()}

    output = []
    for r in records:
        fb = feedbacks.get(r.id)
        output.append({
            "id": r.id,
            "crop": r.crop,
            "growth_stage": r.growth_stage,
            "region": r.region,
            "symptoms": r.symptoms,
            "status": r.status,
            "overall_confidence": r.overall_confidence,
            "result": r.result_json,
            "created_at": r.created_at,
            "feedback": {
                "is_helpful": fb.is_helpful,
                "user_comment": fb.user_comment
            } if fb else None
        })
    return output

@app.get("/")
def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path)

@app.post("/api/feedback")
def submit_feedback(fb_req: FeedbackRequest, db: Session = Depends(get_db)):
    """Сохраняет оценку агронома (помогло / не помогло + реальный диагноз)"""
    check = db.query(CheckHistory).filter(CheckHistory.id == fb_req.check_id).first()
    if not check:
        raise HTTPException(status_code=404, detail="Проверка не найдена")

    existing_fb = db.query(Feedback).filter(Feedback.check_id == fb_req.check_id).first()
    if existing_fb:
        existing_fb.is_helpful = fb_req.is_helpful
        existing_fb.user_comment = fb_req.user_comment
    else:
        new_fb = Feedback(
            check_id=fb_req.check_id,
            is_helpful=fb_req.is_helpful,
            user_comment=fb_req.user_comment
        )
        db.add(new_fb)

    db.commit()
    return {"status": "ok", "message": "Обратная связь успешно сохранена"}


@app.get("/api/offline-bundle")
def get_offline_bundle(db: Session = Depends(get_db)):
    """
    Отдает сжатый справочник из БД на мобильное устройство (в IndexedDB/localStorage),
    чтобы приложение могло проводить предварительную диагностику в поле без интернета!
    """
    articles = db.query(KnowledgeArticle).all()
    return {
        "version": "2026.10.1",
        "crops": sorted(list(set(a.crop for a in articles))),
        "articles": [
            {
                "id": a.id,
                "crop": a.crop,
                "category": a.category,
                "title": a.title,
                "growth_stages": a.growth_stages,
                "symptoms_text": a.symptoms_text,
                "recommendations": a.recommendations,
                "source": {
                    "title": a.source_title,
                    "section": a.source_section,
                    "url_or_code": a.source_code
                }
            }
            for a in articles
        ]
    }