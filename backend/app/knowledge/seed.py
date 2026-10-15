"""Добавляем начальные карточки, сохраняя уже существующие записи."""
import json
from pathlib import Path
from sqlalchemy.orm import Session
from app.storage.models import KnowledgeArticle

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge" / "potato.json"
LEGACY_DEMO_IDS = {
    "potato_scab_tuber_01", "potato_tuber_rot_02", "potato_late_blight_leaf_03",
    "potato_alternaria_leaf_04", "potato_chlorosis_def_05", "potato_colorado_beetle_06",
}


def seed_initial_knowledge(db: Session):
    items = json.loads(DATA_PATH.read_text(encoding="utf-8"))["articles"]
    extra = DATA_PATH.with_name("bashinkom.json")
    if extra.exists():
        items += json.loads(extra.read_text(encoding="utf-8"))["articles"]
    for item in items:
        if db.get(KnowledgeArticle, item["id"]) is None:
            db.add(KnowledgeArticle(**item))
    # Старые демонстрационные записи сохраняем, но поиск их не использует.
    db.commit()
