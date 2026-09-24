"""Inference utilities: checkpoint loading, model run, box rendering.

All functions reusable by CLI, evaluation and the Streamlit app.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models.hybrid_detector import HybridYOLOFasterRCNN
from models.enhancement import ImageEnhancer

CLASS_COLORS = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]

_MODEL_CACHE = {}


def load_checkpoint_model(checkpoint_path: str, device: str = "auto",
                          override_cfg: dict | None = None, cache_key: str | None = None):
    """Load HybridYOLOFasterRCNN from a checkpoint. Cached by (path, device)."""
    ck_path = str(Path(checkpoint_path).resolve())
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else device if device != "auto" else "cpu")
    key = (ck_path, str(dev), str(sorted((override_cfg or {}).items())))
    if cache_key is not None:
        key = (ck_path, str(dev), cache_key)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = override_cfg or ck.get("config") or yaml.safe_load(open("config.yaml"))
    classes = ck.get("classes") or cfg.get("classes")
    model = HybridYOLOFasterRCNN(len(classes), cfg)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    enhancer = ImageEnhancer(cfg["model"].get("enhancement", {})) \
        if cfg["model"].get("enhancement", {}).get("enabled") else None
    out = {
        "model": model, "classes": classes, "cfg": cfg, "device": dev,
        "enhancer": enhancer,
        "epoch": ck.get("epoch", "?"), "smoke_test": ck.get("smoke_test", False),
        "image_size": ck.get("image_size", cfg["dataset"]["image_size"]),
    }
    _MODEL_CACHE[key] = out
    return out


def detect_image(model_bundle: dict, img_bgr: np.ndarray,
                 conf_thr: float | None = None, nms_thr: float | None = None) -> dict:
    """Run the full pipeline on one BGR image. Returns dict with boxes/scores/labels.

    confidence/nms thresholds temporarily override model values when given.
    """
    model = model_bundle["model"]
    size = model_bundle["image_size"]
    old = (model.cls_thr, model.nms_thr)
    if conf_thr is not None:
        model.cls_thr = conf_thr
    if nms_thr is not None:
        model.nms_thr = nms_thr
    try:
        enhanced = model_bundle["enhancer"].process(img_bgr)["enhanced"] \
            if model_bundle["enhancer"] else img_bgr
        img = cv2.resize(enhanced, (size, size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        x = x.to(model_bundle["device"])
        with torch.no_grad():
            det = model(x)[0]
        # rescale boxes back to original image size
        h, w = img_bgr.shape[:2]
        boxes = det["boxes"].numpy().copy()
        if len(boxes):
            boxes[:, [0, 2]] *= w / size
            boxes[:, [1, 3]] *= h / size
        return {
            "boxes": boxes,
            "scores": det["scores"].numpy(),
            "labels": det["labels"].numpy(),
            "enhanced_bgr": enhanced,
        }
    finally:
        model.cls_thr, model.nms_thr = old


def draw_detections(img_bgr: np.ndarray, boxes, scores, labels,
                    class_names, thickness: int = 2) -> np.ndarray:
    """Render boxes with class name + confidence on a copy of the image."""
    out = img_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1 - 6, th + 2)
        cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(out, text, (x1 + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
