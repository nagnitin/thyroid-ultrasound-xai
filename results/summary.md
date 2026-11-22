# Robustness Analysis Summary

## Status of Each Task

### ✅ Branch & patch (complete)

| Step | Result |
|------|--------|
| Branch `robustness-fixes` created | ✅ |
| `git am` applied patch cleanly | ✅ commit `702ea5e` |
| `python -m unittest discover -s tests` | ✅ **5/5 tests pass** (1.158 s) |
| Branch pushed to `origin/robustness-fixes` | ✅ |
| PR to main | ⚠️ Open manually: https://github.com/nagnitin/thyroid-ultrasound-xai/pull/new/robustness-fixes |

---

### ⚠️ External study (`paper/external_study.py`) — blocked: missing data

**Required:** `batch1.rar` (MD5 `a6fb578883ae70a08fd9e92fa7da7ab5`) and `batch2.rar` (MD5 `08d02473840c0e916d5edf6a462026d8`) from [ThyUS2Path (Figshare 27021604)](https://doi.org/10.6084/m9.figshare.27021604.v1).  
**Required:** `runs/resnet18/best.pt` checkpoint — **not found on this machine**.

The existing `runs/development/best.pt` (MobileNet, seed 42) is at `D:\Downloads\thyroid_xai\runs\development\best.pt`.

**Reported paper values to verify (not yet runnable):**

| Metric | Paper value |
|--------|-------------|
| MobileNet external AUROC | 0.527 |
| ResNet-18 external AUROC | 0.538 |
| MobileNet hotspot count / N | 325 / 423 |
| ResNet-18 hotspot count / N | 288 / 423 |
| Paired AUROC difference | 8.7 pp (ResNet − Mobile) |

**To run once data is available:**
```bash
python paper/external_study.py --project . --archives data/thyus2path --out runs/jiaxing
python paper/verify.py
```

---

### ⚠️ Multi-seed robustness (a) — blocked: missing data + resnet18 checkpoint

**Required:** TN5000 dataset (Figshare 28455641, ~SHA256 `f513017e...`) and `runs/resnet18/best.pt`.

The TN5000 dataset is not present locally. `get_release()` in `scans.py` will auto-download it from Figshare if `data/` is writable and network is available.

**Commands to run once data/checkpoints are available:**
```bash
# Windows PowerShell — adapt paths as needed
$py = "C:\Users\nitin\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$root = "c:\Users\nitin\OneDrive\Desktop\thyroid"
foreach ($s in 42,43,44,45,46) {
    & $py train.py --model mobilenet --seed $s --out runs/mobile_s$s
    & $py train.py --model resnet18  --seed $s --out runs/resnet_s$s
    & $py evaluate.py --checkpoint runs/mobile_s$s/best.pt --phase internal --out runs/mobile_s${s}_internal
    & $py evaluate.py --checkpoint runs/resnet_s$s/best.pt --phase internal --out runs/resnet_s${s}_internal
}
& $py aggregate_seeds.py "runs/*_internal/summary.json" --csv results/seeds.csv
```

---

### ⚠️ GradCAM++ seeds (b) — blocked: same as (a)

```bash
foreach ($s in 42,43,44,45,46) {
    & $py evaluate.py --checkpoint runs/mobile_s$s/best.pt --phase internal --cam gradcampp --out runs/mobile_s${s}_internal_gcpp
    & $py evaluate.py --checkpoint runs/resnet_s$s/best.pt --phase internal --cam gradcampp --out runs/resnet_s${s}_internal_gcpp
    & $py paper/external_study.py --project . --archives data/thyus2path --cam gradcampp --mobile-run runs/mobile_s42 --resnet-run runs/resnet_s42 --out runs/jiaxing_gcpp_s42
}
```

---

### ⚠️ Sanity check (c) — blocked: TN5000 data needed for image loading

`sanity.py` loads malignant test images from `data/TN5000_forReview/JPEGImages/`. Once data is downloaded:

```bash
& $py sanity.py --checkpoint D:\Downloads\thyroid_xai\runs\development\best.pt --data data --out runs/mobile_sanity --n 150
```

The checkpoint exists. Only the TN5000 images are missing.

---

## Available Results (from existing `D:\Downloads\thyroid_xai\runs\`)

### MobileNet (seed 42) — Internal Test (`runs/internal/summary.json`)

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV3-Small |
| Seed | 42 |
| AUROC | **0.8115** |
| Sensitivity | 0.7460 |
| Specificity | 0.7422 |
| Views | 941 |
| Evaluable malignant | 678 / 685 |
| Fraction with positive gap | **0.8879** (88.8%) |
| Mean gap (logit units) | **2.131** |
| Mean hotspot drop | 2.846 |
| Mean control drop | 0.714 |
| Mean center gap | 1.575 |
| Spearman excess vs. gap | 0.354 |

> [!NOTE] `null_positive_rate` is not in this summary because it was computed with the pre-patch code. Re-running `evaluate.py` with the patched code will add it.

---

## What's needed to complete all tasks

| Item | Source |
|------|--------|
| `batch1.rar` + `batch2.rar` | [ThyUS2Path Figshare 27021604](https://doi.org/10.6084/m9.figshare.27021604.v1) |
| `runs/resnet18/best.pt` | Train with `python train.py --model resnet18 --seed 42 --out runs/resnet18` |
| TN5000 dataset | Auto-downloaded by `scans.get_release("data")` on first run |

Once these three items are present, all remaining tasks (a–d + external study + verify) can run end-to-end with the commands above.
