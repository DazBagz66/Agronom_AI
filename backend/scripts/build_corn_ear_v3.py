"""Сборка манифестов початков corn_ear_v3 для train_classifier.py.

Три класса: пузырчатая головня (common_smut), гниль початка без уточнения
вида (ear_rot) и початок без видимых признаков болезни (healthy_ear).

Источники (все уже скачаны в .cache/datasets):
  * iNaturalist (inat_smut, inat_smut_recent, inat_maize) — прежние защищённые
    разбиения corn_ear_v2 сохраняются как есть;
  * inat_ears (карантин corn_field) — 8 здоровых початков других наблюдателей,
    уходят в test;
  * hf_ricemaize_ear_rot (крупные планы) и idadp (100 полевых фото) — гниль;
    каждый источник делится по группам похожих снимков 60/20/20.
Сканы OpenEar (конвейерные снимки) и материалы с некоммерческой лицензией
Bugwood не используются. Метки взяты у публикаторов и сообщества, это не
экспертное подтверждение; builder пишет это в card.json.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "backend/data/datasets/corn_ear_v2/inventory.jsonl"
FIELD_QUARANTINE = ROOT / ".cache/datasets/corn_field/manifests/quarantine.jsonl"
OUTPUT = ROOT / "backend/data/datasets/corn_ear_v3"
WIKIMEDIA_METADATA = ROOT / ".cache/datasets/wikimedia_ears/metadata.jsonl"
WIKIMEDIA_REVIEW = OUTPUT / "wikimedia_review.json"
ROLE_TO_SPLIT = {"research_training": "train", "protected_validation": "validation", "protected_test": "test"}
LABEL_MAP = {"common_smut": "common_smut", "healthy_ear": "healthy_ear", "ear_rot_unverified": "ear_rot"}


def clean_row(row, split, label):
    return {
        "path": row["path"], "sha256": row["sha256"], "labels": [label], "organ": "ear",
        "source": row["source"], "duplicate_group": row.get("duplicate_group") or row["sha256"][:24],
        "partition_group": row.get("partition_group") or row.get("duplicate_group") or row["sha256"][:24],
        "license": row.get("license", ""), "attribution": row.get("attribution", ""),
        "label_basis": row.get("label_basis", "publisher_or_community"), "split": split,
        "exclusion_reasons": [], "review_required": False, "protected": split != "train",
    }


def main():
    random.seed(42)
    rows = [json.loads(line) for line in INVENTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = defaultdict(list)
    rot_candidates = []
    for row in rows:
        label = LABEL_MAP.get(row["labels"][0]) if len(row["labels"]) == 1 else None
        if label is None:
            continue
        if row["role"] in ROLE_TO_SPLIT:
            splits[ROLE_TO_SPLIT[row["role"]]].append(clean_row(row, ROLE_TO_SPLIT[row["role"]], label))
        elif row["source"] == "idadp" and label == "ear_rot":
            rot_candidates.append(row)
        elif row["source"] == "hf_ricemaize_ear_rot" and row["role"] == "license_and_expert_review":
            rot_candidates.append(row)
    # Гниль: внутри каждого источника 60 % групп в train, 20 % validation, 20 % test.
    # Первый запуск держал IDADP целиком в тесте: модель, видевшая гниль только на
    # крупных планах hf, называла полевые початки IDADP головнёй (72 из 100), поэтому
    # оба источника участвуют в обучении, а тест по гнили не является независимым.
    by_source = defaultdict(lambda: defaultdict(list))
    for row in rot_candidates:
        by_source[row["source"]][row.get("partition_group") or row["sha256"][:24]].append(row)
    for source, groups in sorted(by_source.items()):
        order = sorted(groups)
        random.shuffle(order)
        for index, key in enumerate(order):
            share = index / len(order)
            split = "train" if share < .6 else "validation" if share < .8 else "test"
            splits[split].extend(clean_row(row, split, "ear_rot") for row in groups[key])
    for line in FIELD_QUARANTINE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["source"] == "inat_ears" and row["organ"] == "ear" and row["labels"] == ["healthy_ear"]:
            splits["test"].append(clean_row(row, "test", "healthy_ear"))
    # Здоровые початки с Wikimedia Commons после визуального отбора (review_wikimedia_ears.py):
    # 60/20/20 по файлам, чтобы здоровый класс перестал быть самым малочисленным.
    if WIKIMEDIA_REVIEW.exists() and WIKIMEDIA_METADATA.exists():
        accepted = set(json.loads(WIKIMEDIA_REVIEW.read_text(encoding="utf-8"))["accepted"])
        wikimedia = [json.loads(line) for line in WIKIMEDIA_METADATA.read_text(encoding="utf-8").splitlines() if line.strip()]
        wikimedia = sorted((row for row in wikimedia if row["sha256"] in accepted), key=lambda row: row["sha256"])
        random.shuffle(wikimedia)
        for index, row in enumerate(wikimedia):
            share = index / len(wikimedia)
            split = "train" if share < .6 else "validation" if share < .8 else "test"
            row = {**row, "duplicate_group": row["sha256"][:24], "partition_group": row["sha256"][:24],
                   "label_basis": "wikimedia_category_plus_visual_review"}
            splits[split].append(clean_row(row, split, "healthy_ear"))
    # Один и тот же снимок встречается в iNaturalist-подборках дважды: оставляем
    # первое (защищённое) назначение, повтор из другого split отбрасываем.
    seen, dropped = {}, 0
    for split in ["train", "validation", "test"]:
        kept = []
        for row in splits[split]:
            keys = [("sha256", row["sha256"]), ("group", row["partition_group"])]
            if any(seen.get(key, split) != split for key in keys):
                dropped += 1
                continue
            for key in keys:
                seen[key] = split
            kept.append(row)
        splits[split] = kept
    print("dropped duplicates across splits:", dropped)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in ["train", "validation", "test"]:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in splits[split])
        (OUTPUT / f"{split}.jsonl").write_text(text, encoding="utf-8")
        hashes[split] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    card = {
        "version": "corn_ear_v3", "classes": sorted(LABEL_MAP.values()),
        "counts": {split: dict(sorted(Counter(r["labels"][0] for r in splits[split]).items())) for split in splits},
        "sources": {split: dict(sorted(Counter(r["source"] for r in splits[split]).items())) for split in splits},
        "manifest_hashes": hashes,
        "label_basis": "Метки публикаторов и сообщества iNaturalist; экспертного подтверждения нет.",
        "excluded": ["OpenEar: конвейерные сканы, не похожи на фотографии с телефона",
                     "Bugwood: некоммерческая лицензия", "hf_ricemaize со статусом excluded в corn_ear_v2"],
        "test_design": "Гниль: оба источника (hf, IDADP) в train/validation/test по группам, тест не независимый. "
                       "Головня и здоровые початки — прежние защищённые разбиения iNaturalist.",
    }
    (OUTPUT / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(card["counts"], ensure_ascii=False))
    print(json.dumps(card["sources"], ensure_ascii=False))


if __name__ == "__main__":
    main()
