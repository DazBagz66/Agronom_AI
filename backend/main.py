"""HTTP-запросы: активные культуры, история и выдача интерфейса."""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, load_only
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Literal
import cv2
from app.crops import CROPS
from app.diagnostics.crop_advisor import diagnose_crop, apply_crop_display_policy

from app.storage.database import engine, Base, get_db, SessionLocal
from app.storage.models import KnowledgeArticle, CheckHistory, DiagnosisFeedback, InvestigationSession, SessionObservation
from app.storage.investigations import backfill_sessions, observations, attach_observation
from app.diagnostics.corn_text import add_text_assessment, inferred_part, symptom_context
from app.schemas import DiagnosisRequest, DiagnosisResponse, DiagnosisFeedbackRequest
from app.corn_taxonomy import CORN_LABELS
from app.diagnostics.vision import decode_image
from app.diagnostics.crop_model import load_model, crop_capabilities
from app.knowledge.search import KnowledgeSearch, active_articles, article_payload
from app.diagnostics.policy import apply_diagnosis_policy
from app.diagnostics.symptoms import disease_key
from app.diagnostics.advisor import PotatoAdvisor
from app.knowledge.seed import seed_initial_knowledge
from app.knowledge.corn_advice import enrich_corn_advice
from app.diagnostics.feedback_policy import apply_feedback_display_policy
from app.diagnostics.organ_routing import route_corn_photo


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
        backfill_sessions(db)
    finally:
        db.close()


@app.post("/api/diagnose", response_model=DiagnosisResponse)
def diagnose_plant(request: DiagnosisRequest, db: Session = Depends(get_db)):
    """
    Фото и описание -> оценка признаков -> поиск карточек -> вывод -> история.
    """
    previous = []
    if request.session_id:
        investigation = db.get(InvestigationSession, request.session_id)
        if not investigation:
            raise HTTPException(status_code=404, detail='Сессия не найдена')
        if investigation.crop != request.crop:
            raise HTTPException(status_code=422, detail='Для другой культуры начните новую сессию')
        previous = observations(db, request.session_id)
    if request.crop in CROPS:
        analysis_request = request
        if not request.image_base64 and request.crop == 'Кукуруза':
            fallback = previous[-1].result_json.get('input_plant_part', 'auto') if previous else request.plant_part
            part = inferred_part(request.symptoms, fallback if request.plant_part == 'auto' else request.plant_part)
            analysis_request = request.model_copy(update={'plant_part': part})
        resolved, organ_metadata, early_result = route_corn_photo(analysis_request)
        result = early_result if early_result is not None else diagnose_crop(resolved)
        prior_text = [r.symptoms for r in previous if r.symptoms]
        result = add_text_assessment(resolved, result, prior_text)
        if not request.image_base64:
            prior_photo = next((r for r in reversed(previous) if r.image_base64), None)
            if prior_photo:
                prior_result = displayed_record(db, prior_photo)['result']
                result.previous_photo_assessment = (
                    f"Фото от {prior_photo.created_at}: {prior_result['human_explanation']} "
                    'Это результат прежнего снимка, а не оценка текущего состояния.'
                )
        knowledge_request = resolved.model_copy(update={'symptoms': symptom_context(prior_text, request.symptoms)})
        result = enrich_corn_advice(db, knowledge_request, result, rag_service)
        result = result.model_copy(update=organ_metadata)
        return save_diagnosis(request, result, db, resolved_part=resolved.plant_part)
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


def save_diagnosis(request, diagnosis_result, db, resolved_part=None):
    # Сохраняем результат в историю без смешивания культур.
    snapshot=diagnosis_result.model_dump(mode='json')
    part = diagnosis_result.detected_plant_part or resolved_part or request.plant_part
    snapshot['input_plant_part']=part
    snapshot['requested_plant_part']=request.plant_part
    if request.crop in CROPS and part in {'leaf','ear','flower'}:
        try:
            manifest = load_model(request.crop, 'ear' if part == 'ear' else 'leaf')[1]
            snapshot['model_manifest_sha256']=manifest['_loaded_manifest_sha256']
        except (OSError,ValueError,KeyError,cv2.error):pass
    history_record = CheckHistory(
        crop=request.crop,
        growth_stage=request.growth_stage,
        region="",  # Старая колонка сохранена для совместимости с историей.
        symptoms=request.symptoms or "",
        image_base64=request.image_base64,
        status=diagnosis_result.status,
        overall_confidence=diagnosis_result.overall_confidence,
        result_json=snapshot
    )
    db.add(history_record)
    db.flush()
    diagnosis_result.session_id = attach_observation(db, request, history_record)
    snapshot['session_id'] = diagnosis_result.session_id
    history_record.result_json = dict(snapshot)
    db.commit()
    db.refresh(history_record)

    diagnosis_result.check_id = history_record.id
    diagnosis_result.created_at = history_record.created_at
    return diagnosis_result


def displayed_record(db, record, include_image=False):
    result = DiagnosisResponse.model_validate(record.result_json)
    result = (apply_diagnosis_policy(result) if record.crop == 'Картофель'
              else apply_crop_display_policy(result, record.crop))
    part = record.result_json.get('input_plant_part', 'leaf')
    if record.crop == 'Кукуруза':
        feedback = db.query(DiagnosisFeedback).filter_by(check_id=record.id).order_by(DiagnosisFeedback.id.desc()).first()
        result, part = apply_feedback_display_policy(result, part, feedback)
    result.check_id, result.created_at = record.id, record.created_at
    payload = {'id': record.id, 'crop': record.crop, 'growth_stage': record.growth_stage,
        'symptoms': record.symptoms, 'plant_part': part, 'created_at': record.created_at,
        'has_image': bool(record.image_base64), 'status': result.status,
        'overall_confidence': result.overall_confidence, 'result': result.model_dump(mode='json')}
    if include_image:
        payload['image_base64'] = record.image_base64
    return payload


@app.get('/api/sessions')
def list_sessions(limit: int = Query(default=100, ge=1, le=500), db: Session = Depends(get_db)):
    from sqlalchemy import func
    aggregate = db.query(SessionObservation.session_id,
        func.max(SessionObservation.check_id).label('latest'),
        func.count(SessionObservation.check_id).label('count')).group_by(SessionObservation.session_id).subquery()
    rows = db.query(InvestigationSession, CheckHistory, aggregate.c.count).options(load_only(
        CheckHistory.id, CheckHistory.growth_stage, CheckHistory.symptoms)).join(
        aggregate, aggregate.c.session_id == InvestigationSession.id).join(
        CheckHistory, CheckHistory.id == aggregate.c.latest).order_by(
        InvestigationSession.updated_at.desc(), CheckHistory.id.desc()).limit(limit).all()
    return [dict(id=session.id, title=session.title, crop=session.crop, growth_stage=record.growth_stage,
        symptoms=record.symptoms, observation_count=count, updated_at=session.updated_at)
        for session, record, count in rows]


@app.get('/api/sessions/{session_id}')
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.get(InvestigationSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail='Сессия не найдена')
    return {'id': session.id, 'crop': session.crop, 'title': session.title,
        'observations': [displayed_record(db, r, include_image=True) for r in observations(db, session_id)]}


@app.get("/api/history")
def get_check_history(limit: int = Query(default=20, ge=1, le=100),
                      crop: Literal["Кукуруза", "Подсолнечник", "Картофель"] | None = None,
                      db: Session = Depends(get_db)):
    """Возвращает историю проверок с актуальным фильтром отображения"""
    query = db.query(CheckHistory)
    query = query.filter(CheckHistory.crop == crop) if crop else query.filter(CheckHistory.crop.in_(CROPS))
    records = query.order_by(CheckHistory.created_at.desc(), CheckHistory.id.desc()).limit(limit).all()
    latest_feedback = {}
    if records:
        for feedback in db.query(DiagnosisFeedback).filter(
                DiagnosisFeedback.check_id.in_([r.id for r in records])).order_by(DiagnosisFeedback.id):
            latest_feedback[feedback.check_id] = feedback

    output = []
    for r in records:
        result = DiagnosisResponse.model_validate(r.result_json)
        if r.crop == "Картофель":
            result = apply_diagnosis_policy(result)
        else:
            result = apply_crop_display_policy(result, r.crop)
        part = r.result_json.get('input_plant_part', 'leaf')
        if r.crop == 'Кукуруза':
            result, part = apply_feedback_display_policy(result, part, latest_feedback.get(r.id))
        output.append({
            "id": r.id,
            "crop": r.crop,
            "growth_stage": r.growth_stage,
            "symptoms": r.symptoms,
            "plant_part": part,
            "original_plant_part": r.result_json.get('input_plant_part', 'leaf'),
            "status": result.status,
            "overall_confidence": result.overall_confidence,
            "result": result.model_dump(mode="json"),
            "created_at": r.created_at,

        })
    return output


@app.get('/api/feedback/labels')
def feedback_labels():
    return {'labels': CORN_LABELS, 'purpose': 'review_only'}


@app.get('/api/capabilities')
def capabilities():
    return {crop: crop_capabilities(crop) for crop in CROPS}


@app.get('/api/history/{check_id}/feedback')
def get_feedback(check_id: int, db: Session = Depends(get_db)):
    record = db.get(CheckHistory, check_id)
    if not record or record.crop != 'Кукуруза':
        raise HTTPException(status_code=404, detail='Проверка кукурузы не найдена')
    latest = db.query(DiagnosisFeedback).filter(DiagnosisFeedback.check_id == check_id).order_by(DiagnosisFeedback.id.desc()).first()
    return {'feedback': ({'id': latest.id, 'proposed_labels': latest.proposed_labels,
                          'plant_part': latest.plant_part, 'status': latest.reviewer_status}
                         if latest else None)}


@app.post('/api/history/{check_id}/feedback', status_code=201)
def record_feedback(check_id: int, request: DiagnosisFeedbackRequest, db: Session = Depends(get_db)):
    record = db.get(CheckHistory, check_id)
    if not record or record.crop != 'Кукуруза':
        raise HTTPException(status_code=404, detail='Проверка кукурузы не найдена')
    if not record.image_base64 or decode_image(record.image_base64) is None:
        raise HTTPException(status_code=422, detail='В проверке нет фотографии для разметки')
    if any(label not in CORN_LABELS for label in request.proposed_labels):
        raise HTTPException(status_code=422, detail='Неизвестная категория')
    labels = sorted(set(request.proposed_labels))
    if len(labels) > 1 and set(labels) & {'healthy', 'healthy_ear', 'unknown_disease'}:
        raise HTTPException(status_code=422, detail='Здоровый образец или неизвестную причину укажите отдельно')
    feedback = DiagnosisFeedback(check_id=check_id, proposed_labels=labels,
        plant_part=request.plant_part, notes=request.notes.strip(),
        specimen_group=request.specimen_group.strip(), reviewer_status='pending',
        model_snapshot=record.result_json)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {'id': feedback.id, 'status': 'pending',
            'message': 'Уточнение сохранено. Спорный вывод в истории будет помечен для перепроверки. Отметка не переобучает модель автоматически; после проверки пример может использоваться при обучении.'}

@app.get("/")
def serve_frontend():
    frontend_path = FRONTEND_DIST / "index.html"
    if not frontend_path.is_file():
        raise HTTPException(status_code=503, detail="Соберите интерфейс: cd react-frontend; npm install; npm run build")
    return FileResponse(frontend_path)

@app.get("/api/knowledge")
def get_knowledge(crop: Literal["Кукуруза", "Подсолнечник", "Картофель"] = "Кукуруза", db: Session = Depends(get_db)):
    """Метки активных датасетов или сохранённый справочник картофеля."""
    if crop == 'Кукуруза':
        articles = active_articles(db, crop).order_by(KnowledgeArticle.title).all()
        return {'crop': crop, 'count': len(articles), 'articles': [article_payload(a) for a in articles],
                'scope': 'Карточки рекомендаций для листьев и початков. Наличие статьи не означает поддержку болезни моделью.',
                'model_labels': crop_capabilities(crop)['labels']}
    if crop in CROPS:
        config = CROPS[crop]
        labels = crop_capabilities(crop)['labels']
        articles = [{"id": f"{config['id']}_{key}", "crop": crop, "title": label,
                     "source": {"title": "Классы установленной модели", "url_or_code": f"model:{config['id']}"}}
                    for key, label in labels.items()]
        return {"crop": crop, "count": len(articles), "articles": articles,
                "scope": "Метки классификатора; агрономическая база рекомендаций ещё не подготовлена."}
    articles = active_articles(db).order_by(KnowledgeArticle.title).all()
    return {
        "crop": "Картофель",
        "count": len(articles),
        "articles": [article_payload(a) for a in articles],
    }
