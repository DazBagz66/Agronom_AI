"""Обучение головы классификатора на готовых признаках MobileNetV2.

Читаем ZIP или папки классов. Похожие фото остаются в одной выборке;
отложенный test не участвует в обучении и подборе температуры."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.diagnostics.tuber_model import extract_features, softmax

CLASS_MAP = {"Black Scurf": "rhizoctonia", "Blackleg": "blackleg",
             "Common Scab": "common_scab", "Dry Rot": "fusarium",
             "Healthy Potatoes": "healthy", "Miscellaneous": "unknown", "Pink Rot": "pink_rot"}


def records(source, class_map=None):
    """Читаем изображения и метки из ZIP или папок без распаковки архива."""
    class_map = CLASS_MAP if class_map is None else class_map
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                parts = Path(name).parts
                if len(parts) >= 2 and parts[-2] in class_map and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    yield name, class_map[parts[-2]], archive.read(name)
    else:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.parent.name in class_map and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield str(path.relative_to(source)), class_map[path.parent.name], path.read_bytes()


def phash(image):
    """Строим короткий отпечаток изображения для поиска похожих снимков."""
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = cv2.dct(gray)[:8, :8].ravel()[1:]
    return values > np.median(values)


def duplicate_groups(hashes):
    # Объединяем похожие фото, в том числе с разными метками.
    parent = list(range(len(hashes)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(hashes)):
        for j in range(i):
            if np.count_nonzero(hashes[i] != hashes[j]) <= 6:
                parent[root(i)] = root(j)
    return np.array([root(i) for i in range(len(hashes))])


def train(source, backbone, output, seed, class_map=None, dataset=None, supported_organ="tuber"):
    class_map = CLASS_MAP if class_map is None else class_map
    cv2.setNumThreads(2)
    net = cv2.dnn.readNetFromONNX(str(backbone))
    names, labels, features, hashes, digests = [], [], [], [], []
    for name, label, data in records(source, class_map):
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {name}")
        names.append(name); labels.append(label); hashes.append(phash(image))
        digests.append(hashlib.sha256(data).hexdigest())
        features.append(extract_features(net, image))
        if len(names) % 100 == 0:
            print(f"Extracted {len(names)} images", flush=True)
    x, y = np.array(features), np.array(labels)
    if len(x) < 70 or set(y) != set(class_map.values()):
        raise ValueError("All configured classes and at least 70 images are required")
    groups = duplicate_groups(hashes)
    # Исключаем похожие фото с противоречивыми диагнозами.
    ambiguous = {g for g in set(groups) if len(set(y[groups == g])) > 1}
    keep = np.array([g not in ambiguous for g in groups])
    excluded = [names[i] for i in np.flatnonzero(~keep)]
    names = np.array(names)[keep]; digests = np.array(digests)[keep]
    x, y, groups = x[keep], y[keep], groups[keep]
    if min(Counter(y).values()) < 10:
        raise ValueError("Insufficient examples after duplicate audit")
    # Разделяем группы: обучение, настройка и независимая от подбора проверка.
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups))
    test = folds[0][1]; validation = folds[1][1]
    training = np.array(sorted(set(range(len(y))) - set(test) - set(validation)))
    for a, b in [(training, validation), (training, test), (validation, test)]:
        assert not set(groups[a]) & set(groups[b])
    # Средние значения и веса подбираем только на обучающей части.
    scaler = StandardScaler().fit(x[training])
    head = LogisticRegression(C=.1, max_iter=2000, class_weight="balanced", random_state=seed)
    head.fit(scaler.transform(x[training]), y[training])
    classes = head.classes_.tolist()
    # Температура смягчает уверенность; подбираем её без использования test.
    validation_logits = head.decision_function(scaler.transform(x[validation]))
    fit = minimize_scalar(lambda t: log_loss(y[validation], softmax(validation_logits/t), labels=classes),
                          bounds=(.5, 10), method="bounded")
    temperature = float(fit.x)
    test_probabilities = softmax(head.decision_function(scaler.transform(x[test])) / temperature)
    predicted = head.classes_[test_probabilities.argmax(axis=1)]
    accepted = (test_probabilities.max(axis=1) >= .65) & (predicted != "unknown")
    report = {
        "seed": seed, "method": "Frozen ImageNet MobileNetV2 + StandardScaler + multinomial LogisticRegression C=0.1",
        "temperature": temperature, "temperature_fit": "validation only; test untouched",
        "dataset_size": len(y), "counts": dict(Counter(y)), "excluded_conflicting_duplicates": excluded,
        "duplicate_groups": len(set(groups)), "split_counts": {"train": len(training), "validation": len(validation), "test": len(test)},
        "test_accuracy": accuracy_score(y[test], predicted), "test_balanced_accuracy": balanced_accuracy_score(y[test], predicted),
        "test_log_loss": log_loss(y[test], test_probabilities, labels=classes),
        "classes": classes, "confusion_matrix": confusion_matrix(y[test], predicted, labels=classes).tolist(),
        "classification_report": classification_report(y[test], predicted, labels=classes, output_dict=True, zero_division=0),
        "accepted_at_65": int(accepted.sum()), "test_count": len(test),
        "accepted_accuracy": float(np.mean(predicted[accepted] == y[test][accepted])) if accepted.any() else None,
        "limitations": "Dataset-only evaluation; pHash groups reduce but cannot eliminate source/specimen leakage. Not independent field validation. Healthy means no visible supported disease. No reliable out-of-distribution or crop detector.",
        "splits": {key: [{"file": str(names[i]), "sha256": str(digests[i]), "class": str(y[i]), "group": int(groups[i])} for i in ids]
                   for key, ids in [("train", training), ("validation", validation), ("test", test)]},
        "test_predictions": [{"file": str(names[i]), "actual": str(y[i]), "predicted": str(predicted[j]), "confidence": float(test_probabilities[j].max())} for j, i in enumerate(test)]}
    # Сохраняем веса и отчёт, чтобы обучение можно было проверить и повторить.
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "classifier.npz", mean=scaler.mean_, scale=scaler.scale_,
                        coef=head.coef_, intercept=head.intercept_, temperature=np.array(temperature))
    if backbone.resolve() != (output / "mobilenetv2_features.onnx").resolve():
        (output / "mobilenetv2_features.onnx").write_bytes(backbone.read_bytes())
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"classes": classes, "supported_organ": supported_organ, "input": "RGB 224x224 letterbox, ImageNet mean/std; frozen 1280-dimensional features",
                "files": {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ["mobilenetv2_features.onnx", "classifier.npz"]},
                "training": report["method"], "limitations": report["limitations"],
                "dataset": dataset or "https://www.kaggle.com/datasets/mukaffimoin/potato-diseases-datasets",
                "backbone": "https://huggingface.co/onnxmodelzoo/mobilenetv2-12"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["split_counts", "test_accuracy", "test_balanced_accuracy", "accepted_at_65", "accepted_accuracy", "temperature"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, default=Path(__file__).resolve().parents[1]/"data/models/tuber/mobilenetv2_features.onnx")
    parser.add_argument("--output", type=Path, required=True, help="Use a separate directory to review a new model before replacing the bundled model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.backbone, args.output, args.seed)
