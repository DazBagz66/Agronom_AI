"""Включение обученной модели: проверка артефакта и запись в реестр.

    python backend/scripts/activate_model.py backend/data/models/sunflower_candidates/leaf_v2
    python backend/scripts/activate_model.py backend/data/models/corn_candidates/leaf_v3 --classes common_rust healthy

Что проверяется перед записью в backend/data/models/registry.json:
  * каталог лежит внутри backend/data/models;
  * контрольная сумма model.onnx совпадает с манифестом;
  * ONNX сошёлся с PyTorch при экспорте (onnx_max_abs_error);
  * есть хотя бы один выпущенный класс. По умолчанию берутся классы, прошедшие
    правило из манифеста (release_rule); --classes сужает или расширяет список,
    и это решение фиксируется в манифесте как ручное.
После активации перезапустите сервер: модели кэшируются в памяти.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.crops import CROPS  # noqa: E402
from app.diagnostics import crop_model  # noqa: E402


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def activate(directory, crop_id, classes=None):
    directory = directory.resolve()
    if not directory.is_relative_to(crop_model.MODELS.resolve()):
        raise ValueError("Candidate must live inside backend/data/models")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = next(c for c in CROPS.values() if c["id"] == crop_id)
    organ = manifest["supported_organ"]
    if hashlib.sha256((directory / "model.onnx").read_bytes()).hexdigest() != manifest["files"]["model.onnx"]:
        raise ValueError("Candidate checksum mismatch")
    if manifest.get("onnx_max_abs_error", 1) > 1e-3:
        raise ValueError("ONNX export was not verified against PyTorch")
    if classes:
        unknown = set(classes) - set(manifest["classes"])
        if unknown:
            raise ValueError(f"Unknown classes: {sorted(unknown)}")
        manifest["release_classes"] = sorted(classes)
        manifest["release_decision"] = "manual"
    else:
        manifest["release_decision"] = "release_rule"
    if not manifest.get("release_classes"):
        raise ValueError("No class passed the release rule; pass --classes to release manually")
    manifest["release_status"] = "approved"
    manifest["activated_at"] = datetime.now(timezone.utc).isoformat()
    crop_model.validate_manifest(manifest, config, organ)
    write_json(manifest_path, manifest)
    entries = crop_model.registry()
    entries.setdefault(crop_id, {})[organ] = directory.relative_to(crop_model.ROOT).as_posix()
    write_json(crop_model.REGISTRY, entries)
    return {"crop": crop_id, "organ": organ, "directory": entries[crop_id][organ],
            "release_classes": manifest["release_classes"], "restart_backend": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--crop", choices=[c["id"] for c in CROPS.values()],
                        help="по умолчанию берётся из имени каталога: corn_candidates → corn")
    parser.add_argument("--classes", nargs="+", help="явный список выпускаемых классов")
    arguments = parser.parse_args()
    crop = arguments.crop or arguments.candidate.resolve().parent.name.split("_")[0]
    print(json.dumps(activate(arguments.candidate, crop, arguments.classes), ensure_ascii=False, indent=2))
