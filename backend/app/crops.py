"""Active crops and exact dataset class mappings. Potato remains a legacy module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CROPS = {
    "Кукуруза": {
        "id": "corn", "parts": ["leaf"],
        "dataset": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
        "image_subdir": "data",
        "class_map": {"Blight": "blight", "Common_Rust": "common_rust", "Gray_Leaf_Spot": "gray_leaf_spot", "Healthy": "healthy"},
        "labels": {"blight": "Ожог листьев кукурузы (Blight)", "common_rust": "Обыкновенная ржавчина кукурузы",
                   "gray_leaf_spot": "Серая пятнистость листьев кукурузы", "healthy": "Здоровый лист"},
    },
    "Подсолнечник": {
        "id": "sunflower", "parts": ["leaf", "flower"],
        "dataset": "noamaanabdulazeem/sunflower-fruits-and-leaves-dataset",
        "image_subdir": "Original Image",
        "class_map": {"Downy mildew": "downy_mildew", "Fresh Leaf": "healthy", "Gray mold": "gray_mold", "Leaf scars": "leaf_scars"},
        "labels": {"downy_mildew": "Ложная мучнистая роса подсолнечника", "healthy": "Здоровый лист",
                   "gray_mold": "Серая гниль подсолнечника", "leaf_scars": "Повреждения листьев (Leaf scars)"},
    },
}


def by_id(crop_id):
    return next(config for config in CROPS.values() if config["id"] == crop_id)
