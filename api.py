import base64
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from review import MAX_BYTES, ReviewModel, open_upload

app = FastAPI(title="Thyroid ultrasound research API")


@lru_cache(maxsize=1)
def model():
    return ReviewModel()


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    """Return the uncalibrated score and a PNG Grad-CAM overlay."""
    try:
        scan = open_upload(image.file.read(MAX_BYTES + 1))
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    r = model().predict(scan)
    r.pop("original")
    r["overlay_png_base64"] = base64.b64encode(r.pop("overlay")).decode("ascii")
    return r
