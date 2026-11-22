# Thyroid Ultrasound XAI

Research code for classifying known thyroid nodules from ultrasound images with MobileNetV3-Small and ResNet-18. A Streamlit viewer displays the model score and Grad-CAM overlay and can export a Word report. A FastAPI endpoint exposes the same inference workflow.

## Train and run locally

Use Python 3.12 in a virtual environment. Training requires network access for the TN5000 dataset and pretrained initialization. Run from the repository root.

```bash
python -m pip install -r requirements.txt
python train.py --out runs/development
python train.py --model resnet18 --out runs/resnet18
python -m streamlit run app.py
```

Select a trained classifier and upload a de-identified PNG or JPEG. To launch the API, run `python -m uvicorn api:app --host 127.0.0.1 --port 8000` and open `http://127.0.0.1:8000/docs`. If you already have the project checkpoints, place the two best.pt files in runs/development and runs/resnet18 to skip training.

## Evaluate

```bash
python evaluate.py --checkpoint runs/development/best.pt --phase internal --out runs/internal
```

The original internal image-level AUROC was 0.812 for MobileNet and 0.901 for ResNet. On the independent Jiaxing cohort the patient-level AUROCs were 0.527 and 0.538 respectively. These models are for research use. The public source release excludes individual study records and scans.

Training data: [TN5000](https://doi.org/10.6084/m9.figshare.28455641.v1). External study data: [ThyUS2Path](https://doi.org/10.6084/m9.figshare.27021604.v1).
