"""Evaluate a frozen checkpoint on validation, official TN5000 test or an external cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from explanations import audit_image, localization
from models import load_checkpoint, metrics
from scans import external_manifest, fit_image, fit_roi, get_release

BOOT = 2000


def clean_external(df, run):
    # Identical grayscale data across sources cannot support independent-site evaluation.
    known = pd.read_csv(run / "split_manifest.csv", usecols=["sha256"])
    excluded = pd.read_csv(run / "excluded_views.csv", usecols=["sha256"])
    internal = set(known.sha256) | set(excluded.sha256)
    hashes = []
    for p in df.image_path:
        with Image.open(p) as img:
            g = img.convert("L")
            hashes.append(hashlib.sha256(f"{g.width}x{g.height}:".encode() + g.tobytes()).hexdigest())
    df = df.assign(sha256=hashes)
    repeated = df.sha256.duplicated(keep=False) | df.sha256.isin(internal)
    return df.loc[~repeated].copy(), df.loc[repeated, ["image_id", "patient_id", "sha256"]]


def finite_mean(series):
    values = series.dropna()
    return float(values.mean()) if len(values) else None


def primary_summary(cases, external, seed=1729):
    positives = cases.loc[cases.label == 1]
    valid = positives.loc[positives.gap.notna()]
    summary = dict(reference_malignant=len(positives), evaluable=len(valid),
                   mean_gap=finite_mean(valid.gap), mean_hotspot_drop=finite_mean(valid.hot_drop),
                   mean_control_drop=finite_mean(valid.control_drop),
                   mean_null_positive_rate=finite_mean(valid.null_positive_rate) if "null_positive_rate" in valid else None,
                   fraction_with_positive_gap=finite_mean((valid.gap > 0).astype(float)),
                   center_evaluable=int(positives.center_gap.notna().sum()),
                   mean_center_gap=finite_mean(positives.center_gap),
                   mean_box_mass_minus_area=finite_mean(positives.box_excess),
                   mean_box_mass_minus_center_prior=finite_mean(positives.box_minus_center))
    if len(valid) >= 2:
        rng = np.random.default_rng(seed)
        delta = valid.gap.to_numpy()
        means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(BOOT)]
        key = "mean_gap_patient_bootstrap_95ci" if external else "mean_gap_image_bootstrap_95ci"
        summary[key] = np.quantile(means, [0.025, 0.975]).tolist()
    joint = positives[["gap", "box_excess"]].dropna()
    summary["localization_gap_pairs"] = len(joint)
    summary["spearman_excess_vs_gap"] = None
    if len(joint) >= 3 and joint.nunique().min() > 1:
        summary["spearman_excess_vs_gap"] = float(spearmanr(joint.box_excess, joint.gap).statistic)
    return summary


def stratified_auc_ci(cases):
    rng = np.random.default_rng(1730)
    groups = [part.index.to_numpy() for _, part in cases.groupby("label")]
    y, scores = cases.label.to_numpy(), cases.score.to_numpy()
    values = []
    for _ in range(BOOT):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(roc_auc_score(y[idx], scores[idx]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def evaluate(args):
    out, run = Path(args.out), Path(args.checkpoint).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose a new evaluation folder to preserve existing results.")
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(args.checkpoint, device)
    external = args.phase == "external"
    if external:
        if not args.csv:
            raise ValueError("Supply the external test manifest with --csv.")
        data, duplicates = clean_external(external_manifest(args.csv), run)
        duplicates.to_csv(out / "excluded_external_exact_duplicates.csv", index=False)
    else:
        manifest = pd.read_csv(run / "split_manifest.csv", dtype={"image_id": str})
        split = "val" if args.phase == "validation" else "test"
        data = manifest.loc[manifest.split == split].copy()
        if split == "val":
            ids = pd.read_csv(run / "validation_ids.csv", dtype={"image_id": str})
            data = data.loc[data.image_id.isin(ids.image_id)].copy()
        folder = get_release(args.data) / "JPEGImages"
        data["image_path"] = [str(folder / f"{i}.jpg") for i in data.image_id]
    if set(data.label) != {0, 1}:
        raise ValueError("Evaluation requires both reference classes after duplicate exclusions.")
    rows = []
    for _, row in data.iterrows():
        with Image.open(row.image_path) as img:
            gray, valid = fit_image(img)
            roi = fit_roi(row, img.size)
        r, heat, center = audit_image(model, ck["architecture"], gray, valid, roi, device, args.grid, args.fill, args.cam)
        r.update(image_id=str(row.image_id), label=int(row.label))
        if external:
            r["patient_id"] = str(row.patient_id)
            r["contour_mass"], r["contour_excess"], _ = localization(heat, center, roi, valid)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "view_results.csv", index=False)
    cols = ["score", "gap", "hot_drop", "control_drop", "null_positive_rate", "center_gap", "box_mass", "box_excess",
            "box_minus_center"]
    if external:
        cols += ["contour_mass", "contour_excess"]
        # One index nodule per ThyroidXL patient. Each patient receives equal weight.
        cases = df.groupby(["patient_id", "label"], as_index=False)[cols].mean()
        counts = df.groupby("patient_id").gap.agg(n_views="size", evaluable_views="count")
        cases = cases.join(counts, on="patient_id")
    else:
        cases = df[["image_id", "label", *cols]].copy()
    cases.to_csv(out / "case_results.csv", index=False)
    summary = dict(phase=args.phase, architecture=ck["architecture"], unit="patient" if external else "image",
                   grid=args.grid, fill=args.fill, cam=args.cam, views=len(df),
                   view_reasons={str(k): int(v) for k, v in df.reason.value_counts().items()},
                   classification=metrics(cases.label, cases.score, ck["threshold"]),
                   primary=primary_summary(cases, external))
    if external:
        summary["classification"]["patient_auroc_bootstrap_95ci"] = stratified_auc_ci(cases.reset_index(drop=True))
        summary["empty_external_masks"] = int(df.roi_empty.sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase", choices=["validation", "internal", "external"], default="validation")
    p.add_argument("--data", default="data")
    p.add_argument("--csv", help="External manifest; see README.")
    p.add_argument("--out", required=True)
    p.add_argument("--grid", choices=[4, 8], type=int, default=4)
    p.add_argument("--fill", choices=["blur", "mean"], default="blur")
    p.add_argument("--cam", choices=["gradcam", "gradcampp"], default="gradcam",
                   help="Attribution method. The reported study used gradcam.")
    p.add_argument("--threads", type=int, default=4)
    evaluate(p.parse_args())
