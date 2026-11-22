"""Shared model, checkpoint and validation code. Scores are uncalibrated."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import (MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small,
                                resnet18)

from scans import ScanSet


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name, pretrained=False):
    if name == "mobilenet":
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 1)
    elif name == "resnet18":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 1)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return net


def target_layer(model, name):
    return model.features[-1] if name == "mobilenet" else model.layer4[-1]


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = build_model(ck["architecture"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


def loader(records, train, batch_size, workers, seed):
    return DataLoader(ScanSet(records, train), batch_size=batch_size, shuffle=train,
                      num_workers=workers, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict(model, records, device, batch=32, workers=0):
    model.eval()
    results = []
    for imgs, labels, ids in loader(records, False, batch, workers, 0):
        probs = model(imgs.to(device)).flatten().sigmoid().cpu().numpy()
        results.extend(zip(ids, labels.int().tolist(), probs.tolist()))
    return pd.DataFrame(results, columns=["image_id", "label", "score"])


def choose_threshold(y, p):
    fp, tp, thresholds = roc_curve(y, p, drop_intermediate=False)
    ok = np.isfinite(thresholds)
    return float(thresholds[ok][np.argmax((tp - fp)[ok])])


def metrics(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p)
    if len(np.unique(y)) != 2:
        raise ValueError("Classification metrics require both reference classes.")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=int(len(y)), malignant=int(y.sum()), auroc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
                accuracy=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred, zero_division=0)),
                brier=float(brier_score_loss(y, p)), threshold=float(threshold),
                tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def fit(train, val, out, name, device, epochs, seed, workers, batch=32, lr=0.0003):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = build_model(name, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best = -1.
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for imgs, labels, _ in loader(train, True, batch, workers, seed + epoch):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs).flatten(), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        preds = predict(model, val, device, batch, workers)
        auc = roc_auc_score(preds.label, preds.score)
        history.append(dict(epoch=epoch, loss=loss_sum / len(train), val_auroc=float(auc)))
        print(f"Epoch {epoch}/{epochs} | loss {loss_sum / len(train):.4f} | validation AUROC {auc:.4f}", flush=True)
        if auc > best:
            best = auc
            threshold = choose_threshold(preds.label, preds.score)
            checkpoint = dict(state=model.state_dict(), architecture=name, threshold=threshold,
                              epoch=epoch, seed=seed, image_size=160, learning_rate=lr)
            tmp = out / "best.tmp"
            torch.save(checkpoint, tmp)
            tmp.replace(out / "best.pt")
            preds.to_csv(out / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    return load_checkpoint(out / "best.pt", device)
