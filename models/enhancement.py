"""Classical image enhancement: white balance, CLAHE, multi-scale Retinex (MSR).

All parameters come from config.yaml (model.enhancement.*). Functions operate
on BGR uint8 numpy images (OpenCV convention).
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------- white balance
def white_balance_gray_world(img_bgr: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Gray-world white balance: equalise channel means, blended by `strength`."""
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)                      # (B,G,R)
    gray = means.mean()
    gains = gray / np.maximum(means, 1e-6)
    gains = gains ** strength
    out = img * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance_percentile(img_bgr: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """Percentile-based white balance: stretch each channel by its dark percentile."""
    img = img_bgr.astype(np.float32)
    lo = np.percentile(img, percentile, axis=(0, 1))
    out = img - lo[None, None, :]
    out = np.clip(out / max(255.0 - lo.max(), 1e-6) * 255.0, 0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------- CLAHE
def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0,
                tile_grid: int = 8) -> np.ndarray:
    """CLAHE on the L channel of LAB space (avoids per-channel colour drift)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------- multi-scale Retinex
def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # large-sigma blur is computed at reduced resolution then upscaled (fast MSR)
    h, w = img.shape[:2]
    max_dim = 320
    if max(h, w) > max_dim and sigma > 10:
        f = max_dim / max(h, w)
        small = cv2.resize(img, (max(int(w * f), 8), max(int(h * f), 8)),
                           interpolation=cv2.INTER_AREA)
        s_small = sigma * f
        ksize = int(np.ceil(s_small * 3) * 2 + 1)
        blur = cv2.GaussianBlur(small, (ksize, ksize), s_small)
        return cv2.resize(blur, (w, h), interpolation=cv2.INTER_LINEAR)
    ksize = int(np.ceil(sigma * 3) * 2 + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def multi_scale_retinex(img_bgr: np.ndarray, sigmas=(15, 80, 250),
                        eps: float = 1e-6) -> np.ndarray:
    """MSR on each colour channel: R_c = sum_k w_k * [log(I_c) - log(I_c * G_sigma_k)].

    Equal weights across scales by default.
    """
    img = img_bgr.astype(np.float32) / 255.0
    log_img = np.log(np.maximum(img, eps))
    accum = np.zeros_like(img)
    for sigma in sigmas:
        blur = _gaussian_blur(img, float(sigma))
        accum += log_img - np.log(np.maximum(blur, eps))
    accum /= len(sigmas)
    # normalise contrast: scale by per-image std for stable dynamic range
    std = accum.std()
    out = accum * (0.25 / max(std, 1e-6))
    out = (out + 0.5) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- pipeline
class ImageEnhancer:
    """Full enhancement chain: white balance -> CLAHE -> MSR.

    Returns intermediate stages so the UI can visualise every step.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.white_balance_enabled = bool(cfg.get("white_balance", True))
        clahe_cfg = cfg.get("clahe", {}) or {}
        self.clahe_enabled = bool(clahe_cfg.get("enabled", True))
        self.clahe_clip = float(clahe_cfg.get("clip_limit", 2.0))
        self.clahe_grid = int(clahe_cfg.get("tile_grid", 8))
        rx_cfg = cfg.get("retinex", {}) or {}
        self.retinex_enabled = bool(rx_cfg.get("enabled", True))
        self.sigmas = list(rx_cfg.get("sigmas", [15, 80, 250]))

    def process(self, img_bgr: np.ndarray) -> dict:
        """Returns dict with keys: original, white_balance, clahe, msr, enhanced."""
        stages = {"original": img_bgr.copy()}
        img = img_bgr
        if self.white_balance_enabled:
            img = white_balance_gray_world(img)
            stages["white_balance"] = img.copy()
        if self.clahe_enabled:
            img = apply_clahe(img, self.clahe_clip, self.clahe_grid)
            stages["clahe"] = img.copy()
        if self.retinex_enabled:
            img = multi_scale_retinex(img, self.sigmas)
            stages["msr"] = img.copy()
        stages["enhanced"] = img.copy()
        # fill any missing intermediate stage with the previous image so the UI
        # can always render a full 5-panel comparison
        prev = stages["original"]
        for key in ("white_balance", "clahe", "msr"):
            stages.setdefault(key, prev.copy())
            prev = stages[key]
        return stages
