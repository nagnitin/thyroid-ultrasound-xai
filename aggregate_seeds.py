"""Aggregate repeated-seed evaluations into mean, standard deviation and range per architecture.

Accepts evaluate.py `summary.json` files (internal or external) and paper/external_study.py
`results.json` files. Example:

    python aggregate_seeds.py "runs/mobile_s*_internal/summary.json" "runs/resnet_s*_internal/summary.json"
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rows_from(path):
    data = json.loads(Path(path).read_text())
    if "classification" in data:  # evaluate.py summary
        c, p = data["classification"], data["primary"]
        return [dict(file=str(path), model=data["architecture"], phase=data["phase"], auroc=c["auroc"],
                     sensitivity=c["sensitivity"], specificity=c["specificity"],
                     positive_gap_rate=p["fraction_with_positive_gap"], mean_gap_logit=p["mean_gap"],
                     null_positive_rate=p.get("mean_null_positive_rate"))]
    if "hotspot" in data:  # external_study results.json holds both models
        h = data["hotspot"]
        rows = []
        for key, name in (("mobile", "mobilenet"), ("resnet", "resnet18")):
            s = data[key]["all"]
            rows.append(dict(file=str(path), model=name, phase="external", auroc=s["auroc"],
                             sensitivity=s["sensitivity"], specificity=s["specificity"],
                             positive_gap_rate=h[key] / h["n"],
                             mean_gap_logit=data.get("effect_size", {}).get(key, {}).get("mean_gap"),
                             null_positive_rate=data.get("effect_size", {}).get(key, {}).get("mean_null_positive_rate")))
        return rows
    raise ValueError(f"Unrecognized result file: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="+", help="Glob patterns for summary.json or results.json files.")
    parser.add_argument("--csv", help="Optionally write the per-run table here.")
    args = parser.parse_args()
    paths = sorted({p for pattern in args.patterns for p in glob.glob(pattern)})
    if not paths:
        parser.error("No result files matched.")
    runs = pd.DataFrame([r for p in paths for r in rows_from(p)])
    metrics = ["auroc", "sensitivity", "specificity", "positive_gap_rate", "mean_gap_logit", "null_positive_rate"]
    table = runs.groupby(["phase", "model"])[metrics].agg(["count", "mean", "std", "min", "max"])
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    if runs.groupby(["phase", "model"]).size().min() < 3:
        print("\nFewer than three runs for at least one group: standard deviations are not informative.")
    if args.csv:
        runs.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
