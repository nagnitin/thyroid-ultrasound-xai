"""Shared, local research inference and a deterministic Word case report."""
import io
import os
import threading
from pathlib import Path

import torch
from docx import Document
from docx.shared import Inches
from PIL import Image, ImageOps
from scipy.special import expit

from explanations import gradcam, overlay
from models import load_checkpoint
from scans import digest, fit_image

DEFAULT_MODEL = Path(__file__).parent / "runs" / "development" / "best.pt"
MAX_BYTES = 8 * 1024 * 1024


def png_bytes(im):
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def open_upload(data):
    if not 0 < len(data) <= MAX_BYTES:
        raise ValueError("Choose a PNG or JPEG up to 8 MB.")
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in {"JPEG", "PNG"} or im.width * im.height > 16_000_000:
            raise ValueError("Choose a PNG or JPEG with at most 16 megapixels.")
        return ImageOps.exif_transpose(im).convert("L")


class ReviewModel:
    def __init__(self, checkpoint=None):
        checkpoint = Path(checkpoint or os.environ.get("THYROID_CHECKPOINT", DEFAULT_MODEL))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, ck = load_checkpoint(checkpoint, self.device)
        self.name = ck["architecture"]
        self.threshold = ck["threshold"]
        self.model_id = digest(checkpoint)[:12]
        self.lock = threading.Lock()

    def predict(self, img):
        gray, content = fit_image(img)
        with self.lock:
            logit, heat = gradcam(self.net, self.name, gray, self.device)
        score = float(expit(logit))
        flat = bool(heat[content].max() - heat[content].min() <= 1e-12)
        return dict(score=score, threshold=self.threshold, above_threshold=score >= self.threshold,
                    architecture=self.name, checkpoint=self.model_id, flat_map=flat,
                    target="Malignant-class logit", original=png_bytes(Image.fromarray(gray)),
                    overlay=png_bytes(overlay(gray, heat, content)))


def case_report(r):
    doc = Document()
    doc.add_heading("Thyroid ultrasound model review", 0)
    doc.add_paragraph("Research output for an ultrasound image of a known thyroid nodule.")
    table = doc.add_table(rows=0, cols=2)
    table.style = "Light Shading Accent 1"
    rows = [("Model score", f"{r['score']:.4f}"), ("Development threshold", f"{r['threshold']:.4f}"),
            ("Score relative to threshold", "At or above" if r["above_threshold"] else "Below"),
            ("Model", f"{r['architecture']}  |  {r['checkpoint']}"), ("Attribution target", r["target"])]
    for key, val in rows:
        cells = table.add_row().cells
        cells[0].text, cells[1].text = key, val
    doc.add_heading("Input and Grad-CAM overlay", level=2)
    images = doc.add_table(rows=1, cols=2)
    for cell, key in zip(images.rows[0].cells, ["original", "overlay"]):
        cell.paragraphs[0].add_run().add_picture(io.BytesIO(r[key]), width=Inches(2.55))
    caption = ("The malignant-class heatmap is flat for this image." if r["flat_map"] else
               "Brighter overlay colors mark larger positive Grad-CAM values for the malignant class.")
    doc.add_paragraph(caption)
    doc.add_paragraph("The score is uncalibrated. This research prototype has no established clinical use.")
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()
