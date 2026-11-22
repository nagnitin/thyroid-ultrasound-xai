"""Dataset download, split auditing and geometry for the thyroid ultrasound study."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SIZE = 160
URL = "https://ndownloader.figshare.com/files/52597376"
SHA256 = "f513017e2a2c43c1c04e28d1f45e9f89ba6121c18edbedbb2482da7a57f58f88"
RELEASE = "TN5000_forReview"
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_release(cache):
    """Download one pinned public archive. Install the extracted tree atomically."""
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    root = cache / RELEASE
    if root.is_dir():
        return root
    archive = cache / "TN5000.zip"
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        temp = Path(temp)
        if not archive.exists():
            candidate = temp / "download.zip"
            with urllib.request.urlopen(URL, timeout=90) as src, candidate.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if digest(candidate) != SHA256:
                raise ValueError("Downloaded archive failed SHA-256 verification.")
            os.replace(candidate, archive)
        if digest(archive) != SHA256:
            raise ValueError(f"Archive checksum mismatch. Replace {archive} with the Figshare v1 release.")
        with zipfile.ZipFile(archive) as z:
            z.extractall(temp)  # Only the release matching the pinned SHA-256 reaches this call.
        try:
            (temp / RELEASE).rename(root)
        except OSError:
            if not root.is_dir():
                raise
    return root


def tn5000_manifest(root):
    """Read VOC boxes and drop all members of any duplicate decoded-grayscale group."""
    rows = []
    seen = set()
    for split in ("train", "val", "test"):
        ids = (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
        for sample in ids:
            if sample in seen:
                raise ValueError(f"Repeated release identifier: {sample}")
            seen.add(sample)
            xml = ET.parse(root / "Annotations" / f"{sample}.xml").getroot()
            objects = xml.findall("object")
            if len(objects) != 1 or objects[0].findtext("name") not in {"0", "1"}:
                raise ValueError(f"Unexpected nodule annotation for {sample}")
            obj = objects[0]
            xmin, ymin, xmax, ymax = [int(obj.findtext("bndbox/" + key))
                                      for key in ("xmin", "ymin", "xmax", "ymax")]
            path = root / "JPEGImages" / f"{sample}.jpg"
            with Image.open(path) as image:
                gray = image.convert("L")
                w, h = gray.size
                geometry_ok = (w, h) == (int(xml.findtext("size/width")), int(xml.findtext("size/height")))
                content_hash = hashlib.sha256(f"{w}x{h}:".encode() + gray.tobytes()).hexdigest()
            x0, y0, x1, y1 = max(0, xmin - 1), max(0, ymin - 1), min(w, xmax), min(h, ymax)
            geometry_ok = geometry_ok and x0 < x1 and y0 < y1
            rows.append(dict(image_id=sample, image_path=str(path), label=int(obj.findtext("name")),
                             split=split, sha256=content_hash, geometry_ok=geometry_ok, x0=x0, y0=y0, x1=x1, y1=y1))
    frame = pd.DataFrame(rows)
    duplicated = frame.sha256.duplicated(keep=False)
    frame["reason"] = np.select([~frame.geometry_ok, duplicated],
                                ["xml_image_geometry", "duplicate_grayscale_pixels"], default="ok")
    keep = frame.reason == "ok"
    return frame.loc[keep].reset_index(drop=True), frame.loc[~keep].reset_index(drop=True)


def external_manifest(csv_path):
    """One labeled nodule per patient, with one or more views (ThyroidXL official test)."""
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path, dtype={"image_id": str, "patient_id": str, "split": str})
    needed = {"image_id", "patient_id", "image_path", "mask_path", "label", "split"}
    if not needed.issubset(df):
        raise ValueError(f"Missing columns: {sorted(needed - set(df))}")
    if df.empty or df[list(needed)].isna().any().any():
        raise ValueError("Every external view needs its verified patient, image, split, mask and label.")
    for col in ("image_id", "patient_id"):
        df[col] = df[col].str.strip()
        if df[col].eq("").any():
            raise ValueError(f"Empty external {col}.")
    if set(df.split) != {"test"} or not df.label.isin([0, 1]).all():
        raise ValueError("Use only the official test cases with binary reference labels (0 or 1).")
    if df.image_id.duplicated().any() or df.groupby("patient_id").label.nunique().max() != 1:
        raise ValueError("Expected unique view IDs and one consistent index-nodule label per patient.")
    for col in ("image_path", "mask_path"):
        df[col] = df[col].map(lambda p: str((csv_path.parent / p).resolve()))
        missing = [p for p in df[col] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    if df.image_path.duplicated().any():
        raise ValueError("A view path occurs more than once in the external manifest.")
    return df


def subsample(frame, count, seed):
    if count is None or count >= len(frame):
        return frame.reset_index(drop=True)
    indices, _ = train_test_split(np.arange(len(frame)), train_size=count,
                                  stratify=frame.label, random_state=seed)
    return frame.iloc[sorted(indices)].reset_index(drop=True)


def geometry(width, height):
    scale = SIZE / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return w, h, (SIZE - w) // 2, (SIZE - h) // 2


def fit_image(img):
    img = img.convert("L")
    w, h, left, top = geometry(*img.size)
    canvas = Image.new("L", (SIZE, SIZE))
    canvas.paste(img.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    content = np.zeros((SIZE, SIZE), bool)
    content[top:top+h, left:left+w] = True
    return np.asarray(canvas), content


def fit_roi(row, image_size):
    w, h, left, top = geometry(*image_size)
    if "mask_path" in row.index:
        with Image.open(row.mask_path) as mask:
            if mask.size != image_size:
                raise ValueError(f"Image/mask dimensions differ for {row.image_id}.")
            raw = np.asarray(mask)
            if raw.ndim == 3:
                raw = raw[:, :, :3].max(axis=2)
            binary = Image.fromarray((raw > 0).astype("uint8"))
    else:
        raw = np.zeros(image_size[::-1], dtype=np.uint8)
        raw[int(row.y0):int(row.y1), int(row.x0):int(row.x1)] = 1
        binary = Image.fromarray(raw)
    roi = np.zeros((SIZE, SIZE), bool)
    roi[top:top+h, left:left+w] = np.asarray(binary.resize((w, h), Image.Resampling.NEAREST)) > 0
    return roi


def normalize(gray):
    one = torch.from_numpy(np.array(gray, dtype=np.float32, copy=True)).unsqueeze(0) / 255.
    return (one.expand(3, -1, -1) - MEAN) / STD


class ScanSet(Dataset):
    def __init__(self, records, train=False):
        self.records = records.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        with Image.open(row.image_path) as img:
            if self.train and torch.rand(()).item() < 0.5:
                img = ImageOps.mirror(img)
            gray, _ = fit_image(img)
        return normalize(gray), torch.tensor(float(row.label)), str(row.image_id)
