"""Активные культуры. Картофель остаётся устаревшим модулем вне этого списка.

Поля записи:
  id        — имя каталога моделей и ключ реестра (backend/data/models/registry.json);
  parts     — части растения, которые можно выбрать в интерфейсе;
  organs    — орган модели → части интерфейса, которые она обслуживает
              (у подсолнечника модель листа применяется и к корзинке);
  dataset   — исходный набор Kaggle для обучения из папок-классов;
  class_map — папка набора → ключ класса;
  labels    — ключ класса → русское название.
Для кукурузы ключи классов дополнительно описаны в corn_taxonomy.CORN_LABELS.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CROPS = {
    "Кукуруза": {
        "id": "corn",
        "parts": ["leaf"],
        "organs": {"leaf": ["leaf"], "ear": ["ear"]},
        "dataset": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
        "class_map": {"Blight": "blight", "Common_Rust": "common_rust",
                      "Gray_Leaf_Spot": "gray_leaf_spot", "Healthy": "healthy"},
        "labels": {"blight": "Ожогоподобное поражение листьев",
                   "common_rust": "Обыкновенная ржавчина кукурузы",
                   "gray_leaf_spot": "Серая пятнистость листьев кукурузы",
                   "healthy": "Здоровый лист"},
    },
    "Подсолнечник": {
        "id": "sunflower",
        "parts": ["leaf", "flower"],
        "organs": {"leaf": ["leaf", "flower"]},
        "dataset": "noamaanabdulazeem/sunflower-fruits-and-leaves-dataset",
        "class_map": {"Downy mildew": "downy_mildew", "Fresh Leaf": "healthy",
                      "Gray mold": "gray_mold", "Leaf scars": "leaf_scars"},
        "labels": {"downy_mildew": "Ложная мучнистая роса подсолнечника",
                   "healthy": "Здоровый лист",
                   "gray_mold": "Серая гниль подсолнечника",
                   "leaf_scars": "Повреждения листьев (Leaf scars)"},
    },
}


def by_id(crop_id):
    return next(config for config in CROPS.values() if config["id"] == crop_id)
