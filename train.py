"""Train on the official TN5000 development splits and select a checkpoint on validation."""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch

from models import fit, metrics
from scans import SHA256, SIZE, get_release, subsample, tn5000_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/full")
    parser.add_argument("--model", choices=["mobilenet", "resnet18"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Stratified 384 train / 128 validation images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4, help="CPU compute threads.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 2:
        parser.error("Choose at least one epoch and a batch size of at least two.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("Choose a new output directory to preserve the existing experiment.")
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame, duplicates = tn5000_manifest(get_release(args.data))
    frame.drop(columns="image_path").to_csv(out / "split_manifest.csv", index=False)
    duplicates.drop(columns="image_path").to_csv(out / "excluded_views.csv", index=False)
    train = subsample(frame[frame.split == "train"], 384 if args.pilot else None, args.seed)
    val = subsample(frame[frame.split == "val"], 128 if args.pilot else None, args.seed)
    for part, name in [(train, "train_ids.csv"), (val, "validation_ids.csv")]:
        part[["image_id", "sha256"]].to_csv(out / name, index=False)
    config = dict(dataset="TN5000 Figshare v1", sha256=SHA256, device=device, platform=platform.platform(),
                  python=sys.version.split()[0], versions={k: version(k) for k in [
                      "torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "Pillow"]},
                  train_images=len(train), validation_images=len(val), excluded_images=len(duplicates),
                  image_size=SIZE, lr=0.0003, weight_decay=0.0001, **vars(args))
    (out / "run.json").write_text(json.dumps(config, indent=2))
    print(f"{len(train)} training and {len(val)} validation images after duplicate audit; using {device}.", flush=True)
    _, ck = fit(train, val, out, args.model, device, args.epochs, args.seed, args.workers, args.batch)
    p = pd.read_csv(out / "validation_predictions.csv")
    report = metrics(p.label, p.score, ck["threshold"])
    report["selected_epoch"] = ck["epoch"]
    (out / "validation_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
