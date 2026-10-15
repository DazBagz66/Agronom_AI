"""Явная загрузка данных обучения с проверкой контрольных сумм.

Восстановление backbone требует onnx; обычная диагностика этот скрипт не вызывает."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "backend/data/models/tuber"


def download(url, destination, expected):
    """Скачиваем файл только при необходимости и сверяем SHA-256."""
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Downloaded content changed: checksum mismatch; review provenance before updating")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(cache, rebuild):
    provenance = json.loads((MODEL_DIR / "provenance.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    download("https://www.kaggle.com/api/v1/datasets/download/mukaffimoin/potato-diseases-datasets",
             cache / "potato-diseases.zip", provenance["dataset"]["archive_sha256"])
    if rebuild:
        import onnx
        info = provenance["backbone"]
        source = cache / "mobilenetv2.onnx"
        download(f"{info['url']}/resolve/{info['revision']}/mobilenetv2-12.onnx", source, info["source_sha256"])
        model = onnx.load(source)
        # Новый файл сохраняем отдельно для проверки, не заменяя рабочую модель.
        onnx.utils.extract_model(str(source), str(cache/"mobilenetv2_features.onnx"),
                                 [model.graph.input[0].name], ["464"])
    print(f"Dataset ready: {cache / 'potato-diseases.zip'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT/".cache/tuber_training")
    parser.add_argument("--rebuild-backbone", action="store_true")
    args = parser.parse_args()
    prepare(args.cache, args.rebuild_backbone)
