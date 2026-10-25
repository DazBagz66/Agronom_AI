"""Additive session storage: old checks and photographs are preserved."""
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy.orm import load_only
from app.storage.models import CheckHistory, InvestigationSession, SessionObservation


def backfill_sessions(db):
    records = db.query(CheckHistory).options(load_only(CheckHistory.id, CheckHistory.crop,
        CheckHistory.symptoms, CheckHistory.created_at)).outerjoin(SessionObservation,
        SessionObservation.check_id == CheckHistory.id).filter(SessionObservation.check_id.is_(None)).all()
    for record in records:
        session = InvestigationSession(id=str(uuid4()), crop=record.crop,
            title=(record.symptoms.strip() or f'{record.crop}: обследование')[:100],
            created_at=record.created_at, updated_at=record.created_at)
        db.add(session)
        db.flush()
        db.add(SessionObservation(session_id=session.id, check_id=record.id))
    db.commit()


def observations(db, session_id):
    return db.query(CheckHistory).join(SessionObservation,
        SessionObservation.check_id == CheckHistory.id).filter(
        SessionObservation.session_id == session_id).order_by(CheckHistory.id).all()


def attach_observation(db, request, record):
    session = db.get(InvestigationSession, request.session_id) if request.session_id else None
    if session is None:
        session = InvestigationSession(id=str(uuid4()), crop=request.crop,
            title=((request.symptoms or '').strip() or f'{request.crop}: обследование')[:100])
        db.add(session)
    session.updated_at = datetime.now(timezone.utc)
    db.flush()
    db.add(SessionObservation(session_id=session.id, check_id=record.id))
    return session.id
