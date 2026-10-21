"""Train independent corn/sunflower heads; sunflower uses original photos only."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.crops import CROPS, ROOT
from scripts.train_tuber_model import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=["corn", "sunflower", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.crop in {'corn', 'all'}:
        parser.error('Corn now uses audited multi-source splits. Run python backend/scripts/train_corn_expanded.py --output <new candidate directory>. For sunflower use --crop sunflower.')
    backbone_dir = ROOT / "backend/data/models/tuber"
    for config in CROPS.values():
        if args.crop not in {"all", config["id"]}:
            continue
        source = ROOT / ".cache/datasets" / config["id"] / "images" / config["image_subdir"]
        actual = {p.name for p in source.iterdir() if p.is_dir()}
        if actual != set(config["class_map"]):
            raise ValueError(f"Review dataset class mapping before training: {actual}")
        output = ROOT / "backend/data/models" / config["id"]
        train(source, backbone_dir / "mobilenetv2_features.onnx", output, args.seed,
              class_map=config["class_map"], dataset=f"https://www.kaggle.com/datasets/{config['dataset']}",
              supported_organ=",".join(config["parts"]))
        for filename in ["BACKBONE_LICENSE.txt", "BACKBONE_README.md"]:
            shutil.copyfile(backbone_dir / filename, output / filename)
        shutil.copyfile(ROOT / "backend/data/datasets" / f"{config['id']}.json", output / "dataset_provenance.json")
