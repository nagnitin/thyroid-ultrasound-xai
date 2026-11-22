"""Recompute the external headline figures from the distributed predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

folder = Path(__file__).resolve().parent / "external"
views = pd.read_csv(folder / "views.csv")
cases = pd.read_csv(folder / "patients.csv").set_index("patient").sort_index()
report = json.loads((folder / "results.json").read_text())
assert len(cases) == report["retained_patients"]
assert views.image_id.is_unique and views.sha256.is_unique
assert len(views) == report["retained_views"]
assert views.groupby("patient").label.nunique().eq(1).all()
index_views = views.sort_values("image_id").drop_duplicates("patient").set_index("patient")
assert (cases.image_id == index_views.loc[cases.index, "image_id"]).all()
malignant = cases[cases.label == 1].dropna(subset=["mobile_gap", "resnet_gap"])
assert len(malignant) == report["hotspot"]["n"]
for name in ["mobile", "resnet"]:
    s = report[name]["all"]
    p = views.groupby("patient")[name].mean().loc[cases.index]
    assert np.allclose(p, cases[name])
    assert abs(roc_auc_score(cases.label, p) - s["auroc"]) < 1e-10
    assert confusion_matrix(cases.label, p >= s["threshold"]).ravel().tolist() == [s[k] for k in ("tn","fp","fn","tp")]
    assert int((malignant[name+"_gap"] > 0).sum()) == report["hotspot"][name]
print("Verified external patient aggregation, ROC areas, confusion matrices, index views and paired hotspot counts.")
