"""Paired image-level intervals. Run after extracting the complete project."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def intervals(positive, negative, differences):
    rng = np.random.default_rng(73)
    estimates = []
    p, n = len(positive), len(negative)
    for _ in range(2000):
        scores = np.concatenate([positive[rng.integers(p, size=p)], negative[rng.integers(n, size=n)]])
        auc = (rankdata(scores, axis=0)[:p].sum(axis=0) - p*(p+1)/2)/(p*n)
        estimates.append([auc[1]-auc[0], rng.choice(differences, len(differences)).mean()])
    return np.quantile(estimates, [0.025, 0.975], axis=0)

if __name__ == "__main__":
    paired = pd.read_csv(Path(__file__).resolve().parents[1] / "comparison/paired_images.csv")
    keys = ["mobile_score","resnet_score"]
    positive = paired.query("label == 1")
    common = positive.dropna(subset=["mobile_gap","resnet_gap"])
    delta = (common.mobile_gap > 0).astype(int) - (common.resnet_gap > 0).astype(int)
    print(intervals(positive[keys].to_numpy(), paired.query("label == 0")[keys].to_numpy(), delta))
