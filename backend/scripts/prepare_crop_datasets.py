"""Download the two selected Kaggle datasets, pin metadata and safely extract images."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "corn": "smaranjitghose/corn-or-maize-leaf-disease-dataset",
    "sunflower": "noamaanabdulazeem/sunflower-fruits-and-leaves-dataset",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(crop):
    ref = DATASETS[crop]
    target = ROOT / ".cache/datasets" / crop
    target.mkdir(parents=True, exist_ok=True)
    metadata_path = target / "kaggle_metadata.json"
    if not metadata_path.exists():
        with urllib.request.urlopen(f"https://www.kaggle.com/api/v1/datasets/view/{ref}", timeout=60) as response:
            metadata = json.load(response)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version = metadata["currentVersionNumber"]
    archive = target / "dataset.zip"
    report_dir = ROOT / "backend/data/datasets"
    previous_report = report_dir / f"{crop}.json"
    if archive.exists() and previous_report.exists():
        pinned = json.loads(previous_report.read_text(encoding="utf-8"))
        if pinned["version"] != version or sha256(archive) != pinned["archive_sha256"]:
            raise ValueError("Dataset archive differs from recorded provenance; review it before reinstalling")
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}?datasetVersionNumber={version}"
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            total = 0
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
                total += len(chunk)
                if total % (64 * 1024 * 1024) == 0:
                    print(f"{crop}: downloaded {total // (1024 * 1024)} MiB", flush=True)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Kaggle did not return a ZIP archive")
        temporary.replace(archive)
    image_root = (target / "images").resolve()
    counts = Counter()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            destination = (image_root / entry.filename).resolve()
            if not destination.is_relative_to(image_root):
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if entry.is_dir():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != entry.file_size:
                with source.open(entry) as inp, destination.open("wb") as out:
                    shutil.copyfileobj(inp, out)
            if destination.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                counts[destination.parent.relative_to(image_root).as_posix()] += 1
    report = {
        "crop": crop, "source": f"https://www.kaggle.com/datasets/{ref}",
        "version": version, "license": metadata.get("licenseName"),
        "description": metadata.get("description"),
        "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
        "image_counts": dict(sorted(counts.items())),
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "local_path": str(image_root.relative_to(ROOT)).replace("\\", "/"),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{crop}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    for crop in DATASETS if args.crop == "all" else [args.crop]:
        prepare(crop)
