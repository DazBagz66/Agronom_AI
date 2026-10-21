"""Re-evaluate every saved holdout image, without training or changing the user DB.

Writes a reproducible internal audit. This is not an independent field evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.crops import CROPS, ROOT
from app.diagnostics.crop_model import classify_crop


def audit(crop, config):
    directory = ROOT / "backend/data/models" / config["id"]
    report = json.loads((directory / "training_report.json").read_text(encoding="utf-8"))
    source = ROOT if report.get('path_basis') == 'project_root' else ROOT / ".cache/datasets" / config["id"] / "images" / config["image_subdir"]
    saved = {row["file"]: row for row in report["test_predictions"]}
    predictions = []
    mismatches = []
    for index, record in enumerate(report["splits"]["test"], 1):
        path = source / record["file"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Test image changed: {path}")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        scores = classify_crop(crop, image)
        predicted = max(scores, key=scores.get)
        previous = saved[record["file"]]
        if predicted != previous["predicted"] or abs(scores[predicted] - previous["confidence"]) > 1e-5:
            mismatches.append(record["file"])
        predictions.append({"file": record["file"], "actual": record["class"], "predicted": predicted,
                            "confidence": scores[predicted]})
        if index % 100 == 0:
            print(f"{config['id']}: verified {index} holdout images", flush=True)
    actual = [row["actual"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    false_healthy = [row for row in predictions if row["actual"] != "healthy" and row["predicted"] == "healthy"]
    positive_count = sum(label != "healthy" for label in actual)
    return {
        "crop": crop, "evaluation": "Re-run of saved evaluation predictions; see training_report for dataset scope and limitations",
        "method": report["method"], "test_count": len(predictions), "counts": dict(Counter(actual)),
        "accuracy": accuracy_score(actual, predicted), "classes": report["classes"],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=report["classes"]).tolist(),
        "saved_prediction_mismatches": mismatches,
        "diseased_test_count": positive_count, "false_healthy_count": len(false_healthy),
        "false_healthy_rate": len(false_healthy) / positive_count if positive_count else None,
        "confident_false_healthy": [row for row in false_healthy if row["confidence"] >= .65],
        "interpretation": "These results only cover the dataset classes and capture conditions. High softmax scores are not evidence that an unfamiliar plant is healthy.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/diagnosis_audit/holdout_audit.json")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    results = [audit(crop, config) for crop, config in CROPS.items()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))
