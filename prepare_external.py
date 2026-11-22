"""Join publisher-verified labels and patient IDs to official ThyroidXL test images and masks."""
import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from scans import external_manifest


def build(root, metadata, out, image_col="image_id", patient_col="patient_id", label_col="label"):
    root, out = Path(root).resolve(), Path(out).resolve()
    table = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    chosen = [image_col, patient_col, label_col]
    if not set(chosen).issubset(table.columns) or len(set(chosen)) != 3:
        raise ValueError(f"Select distinct columns for image, patient and reference label from {list(table.columns)}.")
    mapped = table[chosen].copy()
    mapped.columns = ["image_id", "patient_id", "label"]
    mapped["image_id"] = mapped.image_id.map(lambda x: Path(x.strip().replace("\\", "/")).stem)
    paths = sorted((root / "test" / "images").glob("*.png"))
    if not paths:
        raise FileNotFoundError("Expected the official release folder test/images with PNG views.")
    indexed = pd.DataFrame(dict(image_id=[p.stem for p in paths],
                                image_path=[os.path.relpath(p, out.parent) for p in paths],
                                mask_path=[os.path.relpath(root / "test" / "masks" / p.name, out.parent) for p in paths]))
    mapped = mapped[mapped.image_id.isin(indexed.image_id)]
    joined = indexed.merge(mapped, on="image_id", how="left", validate="one_to_one", indicator=True)
    if (joined._merge != "both").any() or not joined.label.isin(["0", "1"]).all():
        raise ValueError("Each official test image needs one mapped 0=benign / 1=malignant reference label.")
    if joined.patient_id.str.strip().eq("").any():
        raise ValueError("Every image needs a verified patient ID.")
    joined = joined.drop(columns="_merge").assign(split="test")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Validate relative paths and grouping before the completed file becomes visible.
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        # Validate using absolute paths because the provisional CSV lives in a subdirectory.
        provisional = Path(temp) / "preview.csv"
        checked = joined.copy()
        for col in ["image_path", "mask_path"]:
            checked[col] = checked[col].map(lambda p: str(out.parent / p))
        checked.to_csv(provisional, index=False)
        external_manifest(provisional)
        joined.to_csv(provisional, index=False)
        provisional.replace(out)
    print(f"{len(joined)} views, {joined.patient_id.nunique()} patients -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Local ThyroidXL release with test/images and test/masks.")
    p.add_argument("--metadata", required=True, help="One row per view, exported from provider metadata.")
    p.add_argument("--out", required=True)
    p.add_argument("--image-col", default="image_id")
    p.add_argument("--patient-col", default="patient_id")
    p.add_argument("--label-col", default="label")
    a = p.parse_args()
    build(a.root, a.metadata, a.out, a.image_col, a.patient_col, a.label_col)
