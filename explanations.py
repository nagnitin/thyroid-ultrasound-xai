"""Grad-CAM and a paired, equal-area perturbation audit for a binary classifier."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.special import expit
from torch.nn import functional as F

from models import target_layer
from scans import SIZE, normalize

BLUR_RADIUS = 4
GRID = 4


def gradcam(model, name, gray, device, method="gradcam"):
    """Explain the malignant-class logit for every image, including false negatives.

    method="gradcam" is the original behavior. method="gradcampp" uses the Grad-CAM++ channel
    weighting (Chattopadhay et al., 2018) with the usual closed-form higher-order gradient terms.
    """
    if method not in ("gradcam", "gradcampp"):
        raise ValueError(f"Unknown attribution method: {method}")
    features = []
    def capture(_module, _args, output):
        features.append(output)

    hook = target_layer(model, name).register_forward_hook(capture)
    try:
        with torch.enable_grad():
            x = normalize(gray).unsqueeze(0).to(device).requires_grad_(True)
            logit = model(x).flatten()[0]
            act = features[0]
            grad, = torch.autograd.grad(logit, act)
            if method == "gradcam":
                weights = grad.mean(dim=(2, 3), keepdim=True)
            else:
                g2, g3 = grad.pow(2), grad.pow(3)
                denom = 2 * g2 + act.sum(dim=(2, 3), keepdim=True) * g3
                alpha = torch.where(grad != 0, g2 / torch.where(denom != 0, denom, torch.ones_like(denom)),
                                    torch.zeros_like(g2))
                weights = (alpha * grad.relu()).sum(dim=(2, 3), keepdim=True)
            positive = (weights * act).sum(dim=1, keepdim=True).relu()
            heat = F.interpolate(positive, size=gray.shape, mode="bilinear", align_corners=False)
        return float(logit.detach().cpu()), heat[0, 0].detach().cpu().numpy()
    finally:
        hook.remove()


def centered_prior(content):
    y, x = np.nonzero(content)
    yy, xx = np.indices(content.shape)
    dx = (xx - (x.min() + x.max()) / 2) / (0.20 * (x.max() - x.min() + 1))
    dy = (yy - (y.min() + y.max()) / 2) / (0.20 * (y.max() - y.min() + 1))
    return np.exp(-0.5 * (dx**2 + dy**2)) * content


def tight_box(mask):
    box = np.zeros_like(mask)
    y, x = np.nonzero(mask)
    if len(y):
        box[y.min():y.max()+1, x.min():x.max()+1] = True
    return box


def localization(heat, center, roi, content):
    total = heat[content].sum(dtype=float)
    if not roi.any() or total <= 0 or np.ptp(heat[content]) <= 1e-12:
        return np.nan, np.nan, np.nan
    mass = float(heat[roi].sum(dtype=float) / total)
    area = roi.sum() / content.sum()
    center_mass = center[roi].sum() / center[content].sum()
    return mass, float(mass - area), float(mass - center_mass)


def patch_scores(model, gray, content, heat, center, base, device, grid=GRID, fill="blur"):
    """Average control effect across every other eligible tile. Preserve the original logit units."""
    side = SIZE // grid
    blurred = np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(BLUR_RADIUS)))
    inputs, hotness, centrality = [], [], []
    for y in range(0, SIZE, side):
        for x in range(0, SIZE, side):
            tile = np.s_[y:y+side, x:x+side]
            if not content[tile].all() or (gray[tile] > 12).mean() < 0.60:
                continue
            perturbed = gray.copy()
            perturbed[tile] = blurred[tile] if fill == "blur" else round(float(gray[tile].mean()))
            inputs.append(normalize(perturbed))
            hotness.append(heat[tile].mean())
            centrality.append(center[tile].mean())
    result = dict(eligible_tiles=len(inputs), hot_drop=np.nan, control_drop=np.nan,
                  gap=np.nan, center_gap=np.nan, null_positive_rate=np.nan, reason="ok")
    if len(inputs) < 2:
        result["reason"] = "fewer_than_two_eligible_tiles"
        return result
    if np.ptp(hotness) <= 1e-12 or np.ptp(heat[content]) <= 1e-12:
        result["reason"] = "flat_or_tied_map"
        return result
    with torch.inference_mode():
        batches = (model(torch.stack(inputs[i:i+16]).to(device)).flatten().cpu().numpy()
                   for i in range(0, len(inputs), 16))
        effect = base - np.concatenate(list(batches))
    def top_contrast(values):
        values = np.asarray(values)
        top = np.isclose(values, values.max(), rtol=1e-6, atol=1e-12)
        if top.all():
            return np.nan, np.nan, np.nan
        selected, controls = float(effect[top].mean()), float(effect[~top].mean())
        return selected, controls, selected - controls
    result["hot_drop"], result["control_drop"], result["gap"] = top_contrast(hotness)
    # Random-patch baseline: if the "hotspot" were any single eligible tile chosen without regard to the
    # map, its contrast d_j - mean(d_other) is positive exactly when d_j exceeds the mean over all tiles.
    # The share of such tiles is the positive-contrast rate expected from an uninformative ranking.
    result["null_positive_rate"] = float((effect > effect.mean() + 1e-12).mean())
    result["center_gap"] = top_contrast(centrality)[2]
    if not np.isfinite(result["gap"]):
        result["reason"] = "flat_or_tied_map"
    return result


def audit_image(model, name, gray, content, roi, device, grid=GRID, fill="blur", method="gradcam"):
    base, heat = gradcam(model, name, gray, device, method)
    center = centered_prior(content)
    box = tight_box(roi)
    mass, excess, center_diff = localization(heat, center, box, content)
    result = dict(score=float(expit(base)), roi_empty=not bool(roi.any()), box_mass=mass,
                  box_excess=excess, box_minus_center=center_diff)
    result.update(patch_scores(model, gray, content, heat, center, base, device, grid, fill))
    return result, heat, center


def overlay(gray, heat, content):
    """Blend a sequential warm map onto the scan, leaving the letterbox untouched."""
    from matplotlib import colormaps
    background = np.repeat(gray[..., None].astype(float) / 255, 3, axis=2)
    bounded = heat * content
    if bounded.max() <= 0 or np.ptp(heat[content]) <= 1e-12:
        return Image.fromarray((background * 255).astype("uint8"))
    scaled = bounded / bounded.max()
    colors = colormaps["inferno"](scaled)[..., :3]
    alpha = (0.65 * scaled)[..., None]
    return Image.fromarray(np.rint(255 * (background * (1 - alpha) + colors * alpha)).astype("uint8"))
