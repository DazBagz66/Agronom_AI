"""Дообучение MobileNetV3-Large для одной культуры и экспорт в ONNX.

Один скрипт на все культуры. Источники данных:
  --folder DIR         папки-классы (подсолнечник). Разбиение учитывает группы
                       почти одинаковых снимков; готовое разбиение берётся из
                       --splits-from (training_report.json прежней модели).
  --manifests DIR      манифесты corn_field (train/validation/test.jsonl) с
                       фильтром по --organ.

Рецепт: ImageNet-веса, 2 эпохи учится только голова, затем вся сеть с малым
шагом и косинусным затуханием. Выбор чекпойнта и температуры — только по
validation; test считается один раз в конце. На выходе в --output:
  weights.pt, model.onnx, manifest.json (формат onnx_logits_v1, который читает
  сервер), training_report.json и копии манифестов разбиения.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.corn_taxonomy import CORN_LABELS  # noqa: E402
from app.crops import CROPS  # noqa: E402
from app.diagnostics import preprocess  # noqa: E402

THUMBNAIL_CACHE = ROOT / ".cache/thumbnails/rgb320_v1"
SEED = 42
HEAD_ONLY_EPOCHS = 2
RELEASE_RULE = {"minimum_support": 15, "minimum_precision": .7, "minimum_recall": .7}


# ----------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------
def sha256_of(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def difference_hash(image):
    """64-битный dHash: почти одинаковые снимки получают одинаковый или близкий код."""
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def group_near_duplicates(rows, max_distance=6):
    """Соединяем снимки с близким dHash в одну группу, чтобы они не разошлись по разным split."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [row["dhash"] for row in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if hamming(hashes[i], hashes[j]) <= max_distance:
                parent[find(i)] = find(j)
    for index, row in enumerate(rows):
        row["group"] = f"g{find(index)}"


def split_by_group(rows, fractions=(.6, .2, .2)):
    """Детерминированное разбиение целыми группами в пропорции train/validation/test."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    order = sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    random.Random(SEED).shuffle(order)
    splits = {"train": [], "validation": [], "test": []}
    taken = 0
    for key in order:
        share = taken / len(rows)
        name = "train" if share < fractions[0] else "validation" if share < fractions[0] + fractions[1] else "test"
        splits[name].extend(groups[key])
        taken += len(groups[key])
    return splits


def load_folder_dataset(folder, class_map, source, splits_from=None):
    """Папки-классы → строки {path, sha256, label, source, group}."""
    rows = []
    for raw_name, label in class_map.items():
        directory = folder / raw_name
        if not directory.is_dir():
            raise ValueError(f"Нет папки класса: {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_of(path),
                         "label": label, "source": source, "file": path.relative_to(folder).as_posix()})
    if splits_from:
        previous = json.loads(Path(splits_from).read_text(encoding="utf-8"))["splits"]
        by_sha = {row["sha256"]: row for row in rows}
        splits = {}
        for name, entries in previous.items():
            splits[name] = []
            for entry in entries:
                row = by_sha.get(entry["sha256"])
                if row is None:
                    raise ValueError(f"Снимок из прежнего разбиения не найден: {entry['file']}")
                row["group"] = str(entry["group"])
                splits[name].append(row)
        return splits
    for row in rows:
        row["dhash"] = difference_hash(thumbnail(row))
    group_near_duplicates(rows)
    return split_by_group(rows)


def load_manifest_dataset(directory, organ, min_train):
    """Защищённые манифесты corn_field: только один орган, одна метка, достаточные классы."""
    from scripts.train_corn_field import verify_splits
    raw = {name: (directory / f"{name}.jsonl").read_bytes() for name in ["train", "validation", "test"]}
    splits = {}
    for name, data in raw.items():
        splits[name] = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        splits[name] = [row for row in splits[name] if row["organ"] == organ and len(row["labels"]) == 1]
    verify_splits(splits)
    counts = {name: Counter(row["labels"][0] for row in rows) for name, rows in splits.items()}
    classes = sorted(label for label, n in counts["train"].items()
                     if n >= min_train and counts["validation"][label] >= 5 and label in CORN_LABELS
                     and label not in {"other_damage", "unknown_disease"})
    if len(classes) < 2:
        raise ValueError("Нужно хотя бы два класса с достаточным числом примеров")
    result = {}
    for name, rows in splits.items():
        result[name] = [{"path": row["path"], "sha256": row["sha256"], "label": row["labels"][0],
                         "source": row["source"], "group": row["partition_group"]}
                        for row in rows if row["labels"][0] in classes]
    manifest_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    return result, classes, manifest_hashes


def check_isolation(splits):
    """Ни один файл и ни одна группа не встречаются в двух разных split."""
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            for key in [("sha256", row["sha256"]), ("group", row["group"])]:
                if seen.setdefault(key, name) != name:
                    raise ValueError(f"Утечка между разбиениями: {key}")


def thumbnail(row):
    """Уменьшенная копия с кэшем по sha256: оригиналы по 12 Мп читаются один раз."""
    cache = THUMBNAIL_CACHE / f"{row['sha256']}.png"
    if cache.exists():
        with Image.open(cache) as image:
            return image.convert("RGB")
    image = preprocess.load_rgb(ROOT / row["path"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(cache)
    return image


class ImageDataset(Dataset):
    def __init__(self, rows, classes, train=False):
        self.rows, self.classes, self.train = rows, classes, train
        # Без случайной обрезки: она может вырезать поражение, оставив его метку.
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=30, translate=(.06, .06), scale=(.7, 1.0), fill=preprocess.FILL),
            transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.15, hue=.02),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(.1, .8))], p=.15),
        ])
        self.normalize = transforms.Normalize(preprocess.MEAN.tolist(), preprocess.STD.tolist())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = preprocess.letterbox(thumbnail(row))
        if self.train:
            image = self.augment(image)
        tensor = self.normalize(transforms.functional.to_tensor(image))
        return tensor, self.classes.index(row["label"])


def balanced_sampler(rows):
    """Каждый класс и каждый источник внутри класса получают равный вес."""
    class_source = Counter((row["label"], row["source"]) for row in rows)
    sources_per_class = Counter(label for label, _ in class_source)
    weights = [1 / (class_source[(row["label"], row["source"])] * sources_per_class[row["label"]]) for row in rows]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ----------------------------------------------------------------------------
# Модель и метрики
# ----------------------------------------------------------------------------
def make_model(classes, pretrained=True):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights, progress=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    return model


def set_trainable(model, backbone):
    for parameter in model.features.parameters():
        parameter.requires_grad = backbone
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def train_mode(model):
    """Режим обучения, но статистики BatchNorm остаются из ImageNet: данных мало."""
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def predict_logits(model, rows, classes, batch):
    loader = DataLoader(ImageDataset(rows, classes), batch_size=batch, num_workers=0)
    model.eval()
    logits = []
    with torch.inference_mode():
        for images, _ in loader:
            logits.append(model(images).numpy())
    return np.concatenate(logits) if logits else np.zeros((0, len(classes)))


def softmax(logits, temperature=1.):
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(actual, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    return {"count": int(len(actual)), "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "classification_report": classification_report(actual, predicted, labels=labels, target_names=classes,
                                                           output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}


def fit_temperature(logits, actual, classes):
    labels = list(range(len(classes)))

    def objective(temperature):
        return log_loss(actual, softmax(logits, temperature), labels=labels)

    return float(minimize_scalar(objective, bounds=(.5, 10), method="bounded").x)


def release_evidence(report, classes):
    """Какие классы доказали себя на отложенном тесте по простому и прозрачному правилу."""
    evidence, released = {}, []
    for label in classes:
        scores = report["classification_report"][label]
        evidence[label] = {"support": int(scores["support"]), "precision": round(scores["precision"], 4),
                           "recall": round(scores["recall"], 4)}
        if (scores["support"] >= RELEASE_RULE["minimum_support"]
                and scores["precision"] >= RELEASE_RULE["minimum_precision"]
                and scores["recall"] >= RELEASE_RULE["minimum_recall"]):
            released.append(label)
    return evidence, released


# ----------------------------------------------------------------------------
# Обучение
# ----------------------------------------------------------------------------
def train(model, splits, classes, args, log):
    loader = DataLoader(ImageDataset(splits["train"], classes, train=True), batch_size=args.batch,
                        sampler=balanced_sampler(splits["train"]), num_workers=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    validation_actual = np.array([classes.index(row["label"]) for row in splits["validation"]])
    history, best = [], {"macro_f1": -1.}
    optimizer = scheduler = None
    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= HEAD_ONLY_EPOCHS
        if epoch == 1 or epoch == HEAD_ONLY_EPOCHS + 1:
            set_trainable(model, backbone=not head_only)
            if head_only:
                optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-3)
                scheduler = None
            else:
                optimizer = torch.optim.AdamW([{"params": model.features.parameters(), "lr": args.backbone_lr},
                                               {"params": model.classifier.parameters(), "lr": args.backbone_lr * 3}],
                                              weight_decay=1e-3)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - HEAD_ONLY_EPOCHS)
        train_mode(model)
        started, loss_sum, seen = time.monotonic(), 0., 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
        if scheduler:
            scheduler.step()
        validation_probabilities = softmax(predict_logits(model, splits["validation"], classes, args.batch))
        validation = metrics(validation_actual, validation_probabilities, classes)
        row = {"epoch": epoch, "stage": "head" if head_only else "finetune", "loss": loss_sum / seen,
               "validation_macro_f1": validation["macro_f1"], "validation_accuracy": validation["accuracy"],
               "seconds": round(time.monotonic() - started)}
        history.append(row)
        log(row)
        if validation["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": validation["macro_f1"], "epoch": epoch}
            torch.save(model.state_dict(), args.output / "weights.pt")
    model.load_state_dict(torch.load(args.output / "weights.pt", weights_only=True))
    model.eval()
    return history, best["epoch"]


def export_onnx(model, output):
    """Сохраняем ONNX и проверяем, что OpenCV даёт те же логиты, что PyTorch."""
    import cv2
    sample = torch.zeros(1, 3, preprocess.INPUT_SIDE, preprocess.INPUT_SIDE)
    torch.onnx.export(model, sample, str(output / "model.onnx"), opset_version=17,
                      input_names=["images"], output_names=["logits"], dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(output / "model.onnx"))
    generator = torch.Generator().manual_seed(SEED)
    probe = torch.rand(1, 3, preprocess.INPUT_SIDE, preprocess.INPUT_SIDE, generator=generator)
    net.setInput(probe.numpy())
    with torch.inference_mode():
        reference = model(probe).numpy()
    error = float(np.max(np.abs(net.forward() - reference)))
    if error > 1e-3:
        raise ValueError(f"ONNX и PyTorch расходятся: {error}")
    return error


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if args.output.exists() and any(args.output.iterdir()) and not (args.overwrite or args.evaluate_only):
        raise ValueError("Каталог вывода не пуст; укажите новый или --overwrite")

    config = next(c for c in CROPS.values() if c["id"] == args.crop)
    if args.manifests:
        splits, classes, manifest_hashes = load_manifest_dataset(args.manifests, args.organ, args.min_train)
        labels = {label: CORN_LABELS[label] for label in classes}
        dataset_name = args.manifests.relative_to(ROOT).as_posix()
    else:
        splits = load_folder_dataset(args.folder, config["class_map"], f"kaggle_{args.crop}", args.splits_from)
        classes = sorted(config["class_map"].values())
        labels = {label: config["labels"][label] for label in classes}
        manifest_hashes = None
        dataset_name = args.folder.relative_to(ROOT).as_posix()
    if args.limit:
        for name in splits:
            splits[name] = splits[name][:args.limit]
    check_isolation(splits)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        lines = [json.dumps({k: v for k, v in row.items() if k != "dhash"}, ensure_ascii=False) for row in rows]
        (args.output / f"{name}_manifest.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    counts = {name: dict(sorted(Counter(row["label"] for row in rows).items())) for name, rows in splits.items()}
    progress = args.output / "progress.json"
    previous = []
    if args.evaluate_only and progress.exists():
        previous = [row for row in json.loads(progress.read_text(encoding="utf-8")) if "epoch" in row]
    write_json(progress, [])

    def log(row):
        print(json.dumps(row, ensure_ascii=False), flush=True)
        write_json(progress, json.loads(progress.read_text(encoding="utf-8")) + [row])

    log({"classes": classes, "counts": counts, "threads": args.threads})
    model = make_model(classes)
    if args.evaluate_only:
        # Прерванный запуск: берём лучший сохранённый чекпойнт и его историю.
        model.load_state_dict(torch.load(args.output / "weights.pt", weights_only=True))
        model.eval()
        history = previous
        selected_epoch = max(previous, key=lambda row: row["validation_macro_f1"])["epoch"] if previous else 0
        log({"evaluate_only": True, "selected_epoch": selected_epoch,
             "note": "weights.pt from an earlier run; history restored from progress.json"})
    else:
        history, selected_epoch = train(model, splits, classes, args, log)

    validation_logits = predict_logits(model, splits["validation"], classes, args.batch)
    validation_actual = np.array([classes.index(row["label"]) for row in splits["validation"]])
    temperature = fit_temperature(validation_logits, validation_actual, classes)
    log({"selected_epoch": selected_epoch, "temperature": temperature, "note": "evaluating held-out test once"})

    test_rows = splits["test"]
    test_probabilities = softmax(predict_logits(model, test_rows, classes, args.batch), temperature)
    test_actual = np.array([classes.index(row["label"]) for row in test_rows])
    test = metrics(test_actual, test_probabilities, classes)
    by_source = {}
    for source in sorted({row["source"] for row in test_rows}):
        index = [i for i, row in enumerate(test_rows) if row["source"] == source]
        by_source[source] = metrics(test_actual[index], test_probabilities[index], classes)
    evidence, released = release_evidence(test, classes)
    onnx_error = export_onnx(model, args.output)

    manifest = {
        "format": "onnx_logits_v1", "architecture": "mobilenet_v3_large",
        "classes": classes, "labels": labels, "supported_organ": args.organ,
        "release_classes": released, "temperature": temperature, "input": preprocess.DESCRIPTION,
        "files": {"model.onnx": sha256_of(args.output / "model.onnx")},
        "onnx_max_abs_error": onnx_error, "release_status": "candidate",
        "release_rule": RELEASE_RULE, "release_evidence": evidence,
        "dataset": dataset_name, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "limitations": ["Метки взяты у авторов наборов; это не лабораторное подтверждение.",
                        "Тест отложен по группам похожих снимков, но не является независимой полевой проверкой.",
                        "Здоровый класс означает отсутствие видимых поддерживаемых болезней, а не здоровье растения."],
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "training_report.json", {
        "seed": SEED, "architecture": "mobilenet_v3_large", "classes": classes, "organ": args.organ,
        "recipe": f"{HEAD_ONLY_EPOCHS} epochs head only (lr 1e-3), then full fine-tuning "
                  f"(backbone lr {args.backbone_lr}, cosine), AdamW, label smoothing 0.05, class×source balanced sampling",
        "preprocessing": preprocess.DESCRIPTION, "counts": counts, "manifest_hashes": manifest_hashes,
        "history": history, "selected_epoch": selected_epoch, "temperature": temperature,
        "selection": "validation macro-F1 only; temperature fitted to validation NLL; test evaluated once",
        "validation": metrics(validation_actual, softmax(validation_logits, temperature), classes),
        "test": test, "test_by_source": by_source, "release_evidence": evidence, "release_classes": released,
        "predictions": [{"path": row["path"], "source": row["source"], "actual": row["label"],
                         "predicted": classes[int(p.argmax())], "confidence": float(p.max()),
                         "probabilities": dict(zip(classes, map(float, p)))}
                        for row, p in zip(test_rows, test_probabilities)],
    })
    log({"done": True, "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
         "release_classes": released, "by_source": {s: round(m["accuracy"], 3) for s, m in by_source.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crop", choices=[c["id"] for c in CROPS.values()], required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", type=Path, help="папки-классы")
    source.add_argument("--manifests", type=Path, help="каталог с train/validation/test.jsonl")
    parser.add_argument("--splits-from", type=Path, help="training_report.json с прежним разбиением")
    parser.add_argument("--organ", default="leaf")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--min-train", type=int, default=20, help="минимум обучающих примеров класса (манифесты)")
    parser.add_argument("--limit", type=int, default=0, help="урезать каждый split: только для проверки скрипта")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="не обучать: оценить weights.pt из --output и экспортировать его (после прерванного запуска)")
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    for attribute in ["folder", "manifests", "splits_from"]:
        if getattr(arguments, attribute):
            setattr(arguments, attribute, getattr(arguments, attribute).resolve())
    if arguments.epochs <= HEAD_ONLY_EPOCHS:
        parser.error(f"Нужно больше {HEAD_ONLY_EPOCHS} эпох")
    main(arguments)
