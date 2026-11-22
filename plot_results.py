"""Regenerate the paper figure from a completed primary evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def figure(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    data = pd.read_csv(folder / "case_results.csv")
    m, xai = summary["classification"], summary["primary"]
    selected = data.loc[(data.label == 1) & data.gap.notna()]
    navy, teal, gray = "#203956", "#087e82", "#58687a"
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.3))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.855, bottom=0.09, hspace=0.56, wspace=0.31)
    fig.text(0.10, 0.956, "Thyroid nodule classification and Grad-CAM audit", fontsize=19,
             weight="bold", color=navy)
    fig.text(0.10, 0.918, f"TN5000 held-out test · {len(data):,} images · {len(selected):,} malignant images with an evaluable perturbation",
             fontsize=10, color=gray)
    for ax in axs.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#a0acb6")
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    a, b, c, d = axs.flat
    fpr, tpr, _ = roc_curve(data.label, data.score)
    a.plot(fpr, tpr, color=navy, lw=2, label=f"AUROC {m['auroc']:.3f}")
    a.plot([0, 1], [0, 1], "--", color=gray, lw=1, label="Chance")
    a.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    a.legend(loc="lower right", frameon=False, fontsize=9)
    pr, recall, _ = precision_recall_curve(data.label, data.score)
    b.step(recall[::-1], pr[::-1], where="pre", color=teal, lw=2, label=f"Average precision {m['average_precision']:.3f}")
    b.axhline(data.label.mean(), ls="--", color=gray, lw=1, label=f"Prevalence {data.label.mean():.3f}")
    b.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    b.legend(loc="lower left", frameon=False, fontsize=9)
    c.scatter(selected.control_drop, selected.hot_drop, alpha=.42, s=11, color=teal, edgecolors="none")
    lo = min(selected.control_drop.min(), selected.hot_drop.min())
    hi = max(selected.control_drop.max(), selected.hot_drop.max())
    c.plot([lo, hi], [lo, hi], "--", color=gray, lw=1)
    c.set(xlim=(lo - .2, hi + .2), ylim=(lo - .2, hi + .2),
          xlabel="Mean control-patch logit drop", ylabel="Hotspot logit drop")
    c.text(.04, .94, f"{xai['fraction_with_positive_gap']:.1%} above equality", va="top",
           transform=c.transAxes, color=navy, fontsize=9)
    d.scatter(selected.box_excess, selected.gap, alpha=.42, s=11, color=navy, edgecolors="none")
    d.axhline(0, ls="--", color=gray, lw=1)
    d.axvline(0, ls="--", color=gray, lw=1)
    d.set(xlabel="Box heatmap mass minus box area fraction",
          ylabel="Hotspot minus control logit drop")
    d.text(.04, .94, f"Spearman ρ = {xai['spearman_excess_vs_gap']:.2f}", va="top",
           transform=d.transAxes, color=navy, fontsize=9)
    for ax, title in zip(axs.flat, ["Receiver operating characteristic", "Precision and recall",
                                    "Paired blur effects", "Anatomical overlap and score change"]):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", color=navy, pad=9)
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="runs/internal")
    p.add_argument("--out", default="internal_figure.png")
    a = p.parse_args()
    figure(a.results).savefig(a.out, dpi=200, facecolor="white")
