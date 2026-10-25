"""Отбор загруженных с Wikimedia початков: определитель органа + контактные листы.

1. Каждый снимок прогоняется через определитель органа кукурузы; остаются те,
   где класс «початок» набрал не меньше --min-score (по умолчанию 0,5).
2. Для оставшихся собираются контактные листы (20 снимков на лист с номерами),
   чтобы вручную исключить больные, варёные, декоративные и прочие неподходящие.
3. Итог визуального отбора записывается в backend/data/datasets/corn_ear_v3/
   wikimedia_review.json: {"accepted": [sha256, ...], "rejected": {sha256: причина}}.
   Этот файл читает build_corn_ear_v3.py.

    python backend/scripts/review_wikimedia_ears.py            # шаги 1–2
    python backend/scripts/review_wikimedia_ears.py --accept 0 1 2 --reject 3:cooked 7:diseased
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.diagnostics.corn_organ import detect_corn_organ  # noqa: E402

METADATA = ROOT / ".cache/datasets/wikimedia_ears/metadata.jsonl"
CANDIDATES = ROOT / ".cache/datasets/wikimedia_ears/candidates.json"
SHEETS = ROOT / ".cache/datasets/wikimedia_ears/sheets"
REVIEW = ROOT / "backend/data/datasets/corn_ear_v3/wikimedia_review.json"


def route_all(min_score):
    rows = [json.loads(line) for line in METADATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = []
    for row in rows:
        image = cv2.imread(str(ROOT / row["path"]), cv2.IMREAD_COLOR)
        if image is None or min(image.shape[:2]) < 96:
            continue
        observation = detect_corn_organ(image)
        row["organ_probabilities"] = observation["probabilities"]
        if observation["probabilities"]["ear"] >= min_score:
            candidates.append(row)
    for index, row in enumerate(candidates):
        row["index"] = index
    CANDIDATES.write_text(json.dumps(candidates, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} downloaded, {len(candidates)} look like ears (ear >= {min_score})", flush=True)
    return candidates


def contact_sheets(candidates, per=20, tile=300):
    SHEETS.mkdir(parents=True, exist_ok=True)
    for sheet in range((len(candidates) + per - 1) // per):
        canvas = Image.new("RGB", (tile * 5, tile * 4), (40, 40, 40))
        draw = ImageDraw.Draw(canvas)
        for k, row in enumerate(candidates[sheet * per:(sheet + 1) * per]):
            with Image.open(ROOT / row["path"]) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            image.thumbnail((tile - 6, tile - 6))
            x, y = (k % 5) * tile, (k // 5) * tile
            canvas.paste(image, (x + 3, y + 3))
            draw.rectangle([x, y, x + 64, y + 22], fill=(0, 0, 0))
            draw.text((x + 6, y + 4), str(row["index"]), fill=(255, 255, 0))
        canvas.save(SHEETS / f"sheet_{sheet:02d}.jpg", quality=85)
    print(f"sheets written to {SHEETS.relative_to(ROOT).as_posix()}", flush=True)


def record_review(accept, reject):
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    by_index = {row["index"]: row for row in candidates}
    review = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else {"accepted": [], "rejected": {}}
    for index in accept:
        review["accepted"].append(by_index[index]["sha256"])
    for item in reject:
        index, _, reason = item.partition(":")
        review["rejected"][by_index[int(index)]["sha256"]] = reason or "unsuitable"
    review["accepted"] = sorted(set(review["accepted"]) - set(review["rejected"]))
    review["basis"] = "Визуальный отбор по контактным листам: початок без видимых признаков болезни, не варёный, не декоративный."
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"accepted": len(review["accepted"]), "rejected": len(review["rejected"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-score", type=float, default=.5)
    parser.add_argument("--accept", type=int, nargs="*", help="индексы принятых снимков")
    parser.add_argument("--reject", nargs="*", help="индекс:причина для отклонённых")
    arguments = parser.parse_args()
    if arguments.accept or arguments.reject:
        record_review(arguments.accept or [], arguments.reject or [])
    else:
        contact_sheets(route_all(arguments.min_score))
