# Thyroid Ultrasound XAI

How well does a classifier trained on one collection of thyroid ultrasound images perform at another site, and which parts of an image influence its score? This project compares **MobileNetV3-Small** and **ResNet-18** on benign versus malignant classification of a **known thyroid nodule**. It pairs each prediction with a Grad-CAM overlay and tests how the prediction changes when an influential region is blurred.

ResNet-18 achieved an AUROC of **0.901** on the internal image-level test and **0.538** on the independent patient-level test. Both models performed close to chance on the external cohort. The viewer and API are research tools for exploring this behavior. Their scores are uncalibrated and unsuitable for clinical decision-making.

## Run your first prediction

The public repository contains the source code. Trained checkpoints, ultrasound scans, and patient-level study outputs are separate. To run the viewer from a fresh clone, start with one model.

**1. Install the project.** Use Python 3.12 and run the commands from the repository root.

```bash
git clone https://github.com/nagnitin/thyroid-ultrasound-xai.git
cd thyroid-ultrasound-xai
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS or Linux, or `.venv\Scripts\activate.bat` in Windows Command Prompt.

```bash
python -m pip install -r requirements.txt
```

**2. Prepare a checkpoint.** Choose the command for the classifier you want. Training downloads TN5000 and the pretrained initialization on first use. If you already have a project checkpoint, place it at the corresponding path and skip that training run.

| Classifier | Training command | Checkpoint expected by the viewer |
| --- | --- | --- |
| MobileNetV3-Small | `python train.py --out runs/development` | `runs/development/best.pt` |
| ResNet-18 | `python train.py --model resnet18 --out runs/resnet18` | `runs/resnet18/best.pt` |

**3. Open the viewer.**

```bash
python -m streamlit run app.py
```

Open the local address printed by Streamlit. Select the classifier whose checkpoint you prepared and upload a de-identified PNG or JPEG, up to 8 MiB and 16 megapixels. The viewer shows the preprocessed image, malignant-class overlay, model score, and validation-selected threshold. **Download research report** exports those results as a Word document. Inference uses CUDA when available and otherwise runs on CPU.

## Study results

These are aggregate results from the original runs. The internal unit was one TN5000 test image. For the independent Jiaxing evaluation, the study averaged the scores of retained views for each patient before computing classification metrics. The decision threshold for each model was selected on development data and kept fixed for testing.

| Evaluation | MobileNetV3-Small | ResNet-18 |
| --- | ---: | ---: |
| Internal AUROC · 941 images | 0.812 | 0.901 |
| Internal sensitivity | 74.6% | 87.0% |
| Internal specificity | 74.2% | 75.0% |
| External AUROC · 737 patients | 0.527 | 0.538 |
| External sensitivity | 21.5% | 11.0% |

ResNet-18 ranked malignant images more accurately in the internal cohort. At its frozen threshold, it identified only 11.0% of malignant cases in the external cohort. The original training used one seed per architecture, and the internal partitions were image-level. The internal estimates therefore do not establish patient-level generalization.

## How the explanation works

1. **Prepare the input.** The pipeline converts the image to grayscale, resizes it proportionally to fit a 160 × 160 canvas, and pads the remaining space. It replicates grayscale into three channels and normalizes the tensor for the backbone.
2. **Generate the overlay.** Grad-CAM uses gradients of the malignant-class logit at the final convolutional stage. The viewer overlays the heatmap on the image content and flags a flat map when it has no spatial variation.
3. **Test the hotspot.** The default evaluation divides the canvas into a 4 × 4 grid. It blurs each eligible tile in turn and compares the logit drop for the hottest tile or tied tiles with the mean drop for the other eligible tiles. A positive gap means blurring the hotspot reduced the malignant-class logit more than blurring the controls. Cases with insufficient eligible tiles or an uninformative map are flagged.
4. **Check localization.** Where annotations are available, the audit measures the heatmap mass inside the nodule box and compares it with the box’s share of image area and a center-weighted baseline. A large perturbation effect measures model dependence on a region. It does not establish that the predicted label is correct.

## Train and evaluate

The full training defaults are four epochs, batch size 32, and seed 42. Training uses AdamW, selects the checkpoint with the best validation AUROC, and chooses a threshold by maximizing validation sensitivity minus the false-positive rate. A smaller pilot uses stratified subsets of 384 training and 128 validation images.

```bash
python train.py --pilot --epochs 2 --out runs/pilot
python evaluate.py --checkpoint runs/pilot/best.pt --phase validation --out runs/pilot_validation
```

For a full internal evaluation, first complete the corresponding full training run. Then use a fresh output folder for each evaluation.

```bash
python evaluate.py --checkpoint runs/development/best.pt --phase internal --out runs/mobile_internal
python evaluate.py --checkpoint runs/resnet18/best.pt --phase internal --out runs/resnet_internal
```

| Output | What to use it for |
| --- | --- |
| `best.pt`, `run.json`, `history.csv` | Load the selected model and inspect the training configuration and epoch history. |
| `split_manifest.csv`, `excluded_views.csv`, `validation_ids.csv` | Reconstruct evaluation inputs and check duplicate exclusions. Keep the training manifests beside the checkpoint for evaluation. |
| `validation_predictions.csv`, `validation_metrics.json` | Inspect validation scores, the selected epoch, and the operating threshold. |
| `summary.json`, `view_results.csv`, `case_results.csv` | Inspect aggregate evaluation metrics, per-view explanations, and case-level results. |

**External evaluation.** To reproduce the reported Jiaxing study, obtain the two original ThyUS2Path archives, name them `batch1.rar` and `batch2.rar`, and place them in the archives directory. The script checks their digests and expects both checkpoints at the viewer paths above, plus the MobileNet training manifests.

```bash
python paper/external_study.py --project . --archives data/thyus2path --out runs/jiaxing
```

The script needs `libarchive-c` (in `requirements.txt`) and the system `libarchive` library to read the `.rar` files. Use `--mobile-run` and `--resnet-run` (paths relative to `--project`) to evaluate other training runs, such as additional seeds, and `--cam gradcampp` for an alternative attribution method. `results.json` keeps the original keys and adds `effect_size`, which reports the contrast in logit units with bootstrap intervals and the random-patch null rate.

For a separate cohort with verified masks, use `evaluate.py --phase external` with a CSV containing `image_id`, `patient_id`, `image_path`, `mask_path`, `label`, and `split`. Use one row per view, binary labels (0 benign, 1 malignant), and `test` for the split. Relative image and mask paths resolve from the CSV directory. Keep generated patient- and view-level outputs local.

```bash
python evaluate.py --checkpoint runs/development/best.pt --phase external --csv data/external_test.csv --out runs/external_test
```

## Robustness analyses

The reported study used one training seed per architecture and Grad-CAM only. The options below extend it without changing any default, so the default commands still reproduce the reported numbers.

**Several seeds.** Train and evaluate each architecture with several seeds, then summarize the spread.

```bash
for s in 42 43 44 45 46; do
  python train.py --model mobilenet --seed $s --out runs/mobile_s$s
  python train.py --model resnet18 --seed $s --out runs/resnet_s$s
  python evaluate.py --checkpoint runs/mobile_s$s/best.pt --phase internal --out runs/mobile_s${s}_internal
  python evaluate.py --checkpoint runs/resnet_s$s/best.pt --phase internal --out runs/resnet_s${s}_internal
done
python aggregate_seeds.py "runs/*_internal/summary.json"
```

**Random-patch baseline and effect size.** `summary.json` and `view_results.csv` include `null_positive_rate`: the share of eligible tiles whose blur effect exceeds the image's mean tile effect. It is the positive-contrast rate an uninformative ranking would give, so compare `fraction_with_positive_gap` with `mean_null_positive_rate`. `mean_gap` is the contrast in logit units, with a bootstrap interval.

**Alternative attribution.** `python evaluate.py ... --cam gradcampp` repeats the audit with Grad-CAM++.

**Model-randomization sanity check.** `python sanity.py --checkpoint runs/development/best.pt --out runs/mobile_sanity` re-initializes the network from the output layer downward and reports how the Grad-CAM map and hotspot statistics change. Maps for the trained model should decorrelate as weights are destroyed.

**Tests.** `python -m unittest discover -s tests` runs offline checks that need no dataset or pretrained weights.

The scripts `paper/verify.py` and `paper/internal_intervals.py` read distributed study outputs (`external/` and `comparison/paired_images.csv`) that are not part of the public source repository.

## Use the prediction API

Start the API after preparing the MobileNet checkpoint. To select another checkpoint, set the `THYROID_CHECKPOINT` environment variable to its path before starting the server.

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open the [interactive API documentation](http://127.0.0.1:8000/docs), or send a multipart request with a local image using the field `image`. Replace the example path with your own.

```bash
curl -F "image=@/path/to/deidentified-scan.png" http://127.0.0.1:8000/predict
```

The JSON response includes `score`, `threshold`, `above_threshold`, `architecture`, a checkpoint hash prefix (`checkpoint`), the Grad-CAM `target`, and `flat_map`. Decode `overlay_png_base64` to recover the overlay as a PNG. The endpoint returns HTTP 422 for unsupported images or files above the upload limits.

## Code and data

| File | Responsibility |
| --- | --- |
| [scans.py](scans.py) | Dataset retrieval, manifests, image geometry, and preprocessing |
| [models.py](models.py) and [train.py](train.py) | Model definitions, optimization, threshold selection, and checkpointing |
| [explanations.py](explanations.py) and [evaluate.py](evaluate.py) | Grad-CAM, perturbation controls, localization, and evaluation summaries |
| [review.py](review.py), [app.py](app.py), and [api.py](api.py) | Shared inference, Word reports, Streamlit viewer, and HTTP endpoint |
| [paper/external_study.py](paper/external_study.py) | Two-model Jiaxing replication and patient-level aggregation |
| [sanity.py](sanity.py), [aggregate_seeds.py](aggregate_seeds.py), [tests/](tests/) | Randomization sanity check, multi-seed summaries, offline tests |

The original development and internal test used [TN5000](https://doi.org/10.6084/m9.figshare.28455641.v1). The independent Jiaxing study used [ThyUS2Path](https://doi.org/10.6084/m9.figshare.27021604.v1). Consult each dataset’s release page for attribution, access terms, and source documentation.
