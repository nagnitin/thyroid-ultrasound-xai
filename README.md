# Thyroid Ultrasound XAI

Compare MobileNetV3-Small and ResNet-18 on benign versus malignant thyroid nodules. The Streamlit viewer displays a model score and a Grad-CAM overlay for a known nodule and exports a Word report. Both trained checkpoints and two attributed example scans are included. A CPU is sufficient for inference.

## Run locally

Extract the project or clone this repository. Open a terminal in the directory containing app.py. Use Python 3.12 and a virtual environment.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate.bat` in Windows Command Prompt or `source .venv/bin/activate` in macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Select a classifier and a bundled example, or upload a de-identified PNG or JPEG. The dashboard displays the model input and the malignant-class heatmap alongside the score and validation-selected threshold.

## API

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Visit [the local API explorer](http://127.0.0.1:8000/docs) or send an example:

```bash
curl -F "image=@examples/000066.jpg" http://127.0.0.1:8000/predict
```

The response includes an uncalibrated score, threshold, checkpoint identifier and base64 PNG overlay. The API defaults to MobileNet. Set THYROID_CHECKPOINT to runs/resnet18/best.pt to use ResNet.

## Research results

| Evaluation | MobileNetV3-Small | ResNet-18 |
| --- | ---: | ---: |
| Internal AUROC, 941 TN5000 images | 0.812 | 0.901 |
| External AUROC, 737 Jiaxing patients | 0.527 | 0.538 |
| External sensitivity at frozen threshold | 21.5% | 11.0% |
| Hotspot blur lowered the score more than control blur, 423 paired external malignant cases | 76.8% | 68.1% |

The external classification results were close to chance. A heatmap can identify pixels influencing an unreliable score. Use this software for research and demonstrations. The internal study used image-level partitions and one training seed per architecture.

![ROC curves for the internal and external evaluations](paper/site_comparison.png)

## Reproduce and explore

- [Executed notebook](Executed_Thyroid_XAI.ipynb) and [editable notebook](Thyroid_XAI.ipynb) for Colab, Jupyter and VS Code.
- [Runbook](RUNBOOK.md) for training, evaluating checkpoints and example attribution.
- [External replication instructions](paper/REPRODUCE.md) for dataset downloads, linkage rules and patient-level analysis.
- [Manuscript](paper/Manuscript.pdf), [editable Word version](paper/Manuscript.docx) and [paired internal results](comparison/COMPARISON.md).

To verify the external scores against the included per-view predictions:

```bash
python paper/verify.py
```

## Data and attribution

Training used [TN5000](https://doi.org/10.6084/m9.figshare.28455641.v1). The two bundled examples are provided under CC BY 4.0 with [source credits](examples/SOURCES.md). External testing used [ThyUS2Path](https://doi.org/10.6084/m9.figshare.27021604.v1). Raw external scans can be obtained from the publisher. The paper includes references for the datasets and Grad-CAM.
