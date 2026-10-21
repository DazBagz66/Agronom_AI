"""HTTP-запросы: активные культуры, история и выдача интерфейса."""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Literal
from app.crops import CROPS
from app.diagnostics.crop_advisor import diagnose_crop, apply_crop_display_policy

from app.storage.database import engine, Base, get_db, SessionLocal
from app.storage.models import KnowledgeArticle, CheckHistory
from app.schemas import DiagnosisRequest, DiagnosisResponse
from app.knowledge.search import KnowledgeSearch, active_articles, article_payload
from app.diagnostics.policy import apply_diagnosis_policy
from app.diagnostics.symptoms import disease_key
from app.diagnostics.advisor import PotatoAdvisor
from app.knowledge.seed import seed_initial_knowledge


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Советник агронома API",
    description="Предварительная диагностика кукурузы и подсолнечника. Порог отображения — 65%. Картофель в бэклоге.",
    version="1.0.0"
)

# Разрешаем запросы браузера, в том числе с отдельного сервера разработки.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_service = KnowledgeSearch()
advisor = PotatoAdvisor()
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "react-frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        seed_initial_knowledge(db)
    finally:
        db.close()


@app.post("/api/diagnose", response_model=DiagnosisResponse)
def diagnose_plant(request: DiagnosisRequest, db: Session = Depends(get_db)):
    """
    Фото и описание -> оценка признаков -> поиск карточек -> вывод -> история.
    """
    if request.crop in CROPS:
        return save_diagnosis(request, diagnose_crop(request), db)
    # Совместимость с прежними клиентами; картофель не предлагается в интерфейсе.
    # 1. Рассматриваем фотографию.
    triage_report = advisor.analyze_photo(request)

    # 2. Находим подходящие описания во внутренней базе.
    enriched_query = f"{request.symptoms or ''} {triage_report.extracted_visual_symptoms}"
    rag_articles = rag_service.retrieve(
        db=db,
        crop=request.crop,
        growth_stage=request.growth_stage,
        query_symptoms=enriched_query,
        top_k=50
    )
    # Добавляем карточку модельного диагноза, даже если текстовый поиск её пропустил.
    predicted_keys = {key for key, score in triage_report.class_probabilities.items() if score >= .65}
    included_ids = {article["id"] for article in rag_articles}
    for article in active_articles(db).all():
        if article.id not in included_ids and disease_key(article.title) in predicted_keys:
            rag_articles.append(dict(article_payload(article), similarity_score=0.))

    # 3. Объединяем признаки и применяем порог 65%.
    diagnosis_result = advisor.assess(
        request=request,
        triage_report=triage_report,
        rag_articles=rag_articles
    )
    diagnosis_result = apply_diagnosis_policy(diagnosis_result)

    return save_diagnosis(request, diagnosis_result, db)


def save_diagnosis(request, diagnosis_result, db):
    # Сохраняем результат в историю без смешивания культур.
    history_record = CheckHistory(
        crop=request.crop,
        growth_stage=request.growth_stage,
        region="",  # Старая колонка сохранена для совместимости с историей.
        symptoms=request.symptoms or "",
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
def get_check_history(limit: int = Query(default=20, ge=1, le=100),
                      crop: Literal["Кукуруза", "Подсолнечник", "Картофель"] | None = None,
                      db: Session = Depends(get_db)):
    """Возвращает историю проверок с актуальным фильтром отображения"""
    query = db.query(CheckHistory)
    query = query.filter(CheckHistory.crop == crop) if crop else query.filter(CheckHistory.crop.in_(CROPS))
    records = query.order_by(CheckHistory.created_at.desc(), CheckHistory.id.desc()).limit(limit).all()

    output = []
    for r in records:
        result = DiagnosisResponse.model_validate(r.result_json)
        if r.crop == "Картофель":
            result = apply_diagnosis_policy(result)
        else:
            result = apply_crop_display_policy(result)
        output.append({
            "id": r.id,
            "crop": r.crop,
            "growth_stage": r.growth_stage,
            "symptoms": r.symptoms,
            "status": result.status,
            "overall_confidence": result.overall_confidence,
            "result": result.model_dump(mode="json"),
            "created_at": r.created_at,

        })
    return output

@app.get("/")
def serve_frontend():
    frontend_path = FRONTEND_DIST / "index.html"
    if not frontend_path.is_file():
        raise HTTPException(status_code=503, detail="Соберите интерфейс: cd react-frontend; npm install; npm run build")
    return FileResponse(frontend_path)

@app.get("/api/knowledge")
def get_knowledge(crop: Literal["Кукуруза", "Подсолнечник", "Картофель"] = "Кукуруза", db: Session = Depends(get_db)):
    """Метки активных датасетов или сохранённый справочник картофеля."""
    if crop in CROPS:
        config = CROPS[crop]
        articles = [{"id": f"{config['id']}_{key}", "crop": crop, "title": label,
                     "source": {"title": "Классы датасета", "url_or_code": f"https://www.kaggle.com/datasets/{config['dataset']}"}}
                    for key, label in config["labels"].items()]
        return {"crop": crop, "count": len(articles), "articles": articles,
                "scope": "Метки классификатора; агрономическая база рекомендаций ещё не подготовлена."}
    articles = active_articles(db).order_by(KnowledgeArticle.title).all()
    return {
        "crop": "Картофель",
        "count": len(articles),
        "articles": [article_payload(a) for a in articles],
    }
