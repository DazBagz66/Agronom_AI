"""Возможные условия развития болезни из местного справочника."""
import json
from functools import lru_cache
from pathlib import Path
from app.diagnostics.symptoms import disease_key


@lru_cache(maxsize=1)
def factor_data():
    path = Path(__file__).resolve().parents[2] / "data/knowledge/disease_factors.json"
    return json.loads(path.read_text(encoding="utf-8"))


def possible_factors(name):
    return list(factor_data()["conditions"].get(disease_key(name), []))
