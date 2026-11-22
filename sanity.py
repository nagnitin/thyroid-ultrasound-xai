"""Cascading model-randomization sanity check for the Grad-CAM hotspot audit.

Weights are re-initialized from the top of the network downward, one stage at a time (Adebayo et al.,
NeurIPS 2018). After each stage the script recomputes the malignant-class Grad-CAM map for the same
images and reports (i) the rank correlation with the map from the trained model and (ii) the hotspot
contrast statistics. A map that explains the trained model should decorrelate as the weights are
destroyed; a map that does not has little to do with what was learned.

Example:
    python sanity.py --checkpoint runs/development/best.pt --out runs/mobile_sanity --n 150
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

from explanations import gradcam, centered_prior, patch_scores
from models import build_model, load_checkpoint
from scans import fit_image, get_release


def stages(model, name):
    """Module groups ordered from the output layer toward the input."""
    if name == "mobilenet":
        blocks = list(model.features)
        return [("classifier", [model.classifier]), ("features[10:]", blocks[10:]),
                ("features[7:10]", blocks[7:10]), ("features[4:7]", blocks[4:7]),
                ("features[0:4]", blocks[0:4])]
    if name == "resnet18":
        return [("fc", [model.fc]), ("layer4", [model.layer4]), ("layer3", [model.layer3]),
                ("layer2", [model.layer2]), ("layer1", [model.layer1]), ("conv1+bn1", [model.conv1, model.bn1])]
    raise ValueError(f"Unknown backbone: {name}")


def reinitialize(modules, seed):
    torch.manual_seed(seed)
    for top in modules:
        for m in top.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


def rank_correlation(a, b, content):
    x, y = a[content].ravel(), b[content].ravel()
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return np.nan
    return float(spearmanr(x, y).statistic)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose a new output folder to preserve existing results.")
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(args.checkpoint, device)
    name = ck["architecture"]
    manifest = pd.read_csv(Path(args.checkpoint).resolve().parent / "split_manifest.csv", dtype={"image_id": str})
    test = manifest.loc[(manifest.split == "test") & (manifest.label == 1)]
    test = test.sample(n=min(args.n, len(test)), random_state=args.seed).sort_values("image_id")
    folder = get_release(args.data) / "JPEGImages"
    images = []
    for image_id in test.image_id:
        with Image.open(folder / f"{image_id}.jpg") as img:
            images.append(fit_image(img))

    def explain(net):
        heats, rows = [], []
        for gray, content in images:
            logit, heat = gradcam(net, name, gray, device, args.cam)
            r = patch_scores(net, gray, content, heat, centered_prior(content), logit, device)
            heats.append(heat)
            rows.append(r)
        return heats, pd.DataFrame(rows)

    reference, ref_rows = explain(model)
    records = [dict(stage="trained", spearman_vs_trained=1.0, n_maps=len(images),
                    **summarize(ref_rows))]
    for k in range(1, len(stages(model, name)) + 1):
        broken, _ = load_checkpoint(args.checkpoint, device)
        for label, modules in stages(broken, name)[:k]:
            reinitialize(modules, args.seed + k)
        heats, rows = explain(broken)
        corr = [rank_correlation(h0, h1, c) for h0, h1, (_, c) in zip(reference, heats, images)]
        records.append(dict(stage="randomized through " + stages(broken, name)[k - 1][0],
                            spearman_vs_trained=float(np.nanmean(corr)) if np.isfinite(corr).any() else None,
                            n_maps=int(np.isfinite(corr).sum()), **summarize(rows)))
        print(records[-1], flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "sanity_by_stage.csv", index=False)
    (out / "sanity.json").write_text(json.dumps(dict(architecture=name, cam=args.cam, images=len(images),
                                                     stages=records), indent=2, allow_nan=False))
    print(frame.to_string(index=False))


def summarize(rows):
    valid = rows.loc[rows.gap.notna()]
    return dict(evaluable=len(valid),
                fraction_with_positive_gap=float((valid.gap > 0).mean()) if len(valid) else None,
                mean_gap=float(valid.gap.mean()) if len(valid) else None,
                mean_null_positive_rate=float(valid.null_positive_rate.mean()) if len(valid) else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=150, help="Malignant internal-test images to explain.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cam", choices=["gradcam", "gradcampp"], default="gradcam")
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
