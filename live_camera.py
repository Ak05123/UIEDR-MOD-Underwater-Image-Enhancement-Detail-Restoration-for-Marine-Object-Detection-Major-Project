"""Utilities for processing live camera frames through the shared enhancement and detector pipeline."""
from __future__ import annotations

import av
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import os
import threading
import time

import cv2
import numpy as np
import torch

from inference import detect_image, draw_detections


_INFERENCE_LOCK = threading.RLock()
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-model-load")
_MODEL_LOAD_FUTURES: dict[str, Future] = {}


def camera_media_stream_constraints(camera_position: str) -> dict:
    """Request the chosen browser camera while permitting single-camera fallback."""
    facing_mode = {
        "Front Camera": "user",
        "Back Camera": "environment",
    }.get(camera_position)
    if facing_mode is None:
        raise ValueError(f"Unsupported camera choice: {camera_position}")
    return {
        "video": {
            "facingMode": {"ideal": facing_mode},
            "width": {"ideal": 1280, "max": 1280},
            "height": {"ideal": 720, "max": 720},
        },
        "audio": False,
    }


def build_rtc_configuration(webrtc_cfg: dict, environ: dict[str, str] | None = None) -> dict | None:
    """Build ICE configuration from public YAML servers and optional server-side TURN env vars."""
    env = os.environ if environ is None else environ
    ice_servers = list(webrtc_cfg.get("ice_servers", []))
    turn_url = env.get("UIEDR_TURN_URL")
    turn_username = env.get("UIEDR_TURN_USERNAME")
    turn_credential = env.get("UIEDR_TURN_CREDENTIAL")
    if turn_url and turn_username and turn_credential:
        ice_servers.append({
            "urls": [url.strip() for url in turn_url.split(",") if url.strip()],
            "username": turn_username,
            "credential": turn_credential,
        })
    return {"iceServers": ice_servers} if ice_servers else None


class LiveCameraStats:
    """Thread-safe measurements published by the WebRTC frame callback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._completed_frames = []
        self._session_started_at = None
        self._values = {
            "fps": 0.0,
            "processing_ms": 0.0,
            "detections": 0,
            "device": "Not started",
            "error": None,
            "status": "stopped",
            "received_frames": 0,
            "processed_frames": 0,
            "latest_rendered": None,
        }

    def update(self, completed_at: float, processing_ms: float,
               detections: int, device: str, rendered: np.ndarray) -> None:
        with self._lock:
            self._completed_frames.append(completed_at)
            self._completed_frames = self._completed_frames[-30:]
            if len(self._completed_frames) > 1:
                elapsed = self._completed_frames[-1] - self._completed_frames[0]
                fps = (len(self._completed_frames) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0
            self._values = {
                "fps": fps,
                "processing_ms": processing_ms,
                "detections": detections,
                "device": device,
                "error": None,
                "status": "streaming",
                "received_frames": self._values["received_frames"],
                "processed_frames": self._values["processed_frames"] + 1,
                "latest_rendered": rendered.copy(),
            }

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._values["status"] = status
            self._values["error"] = error

    def note_frame_received(self) -> None:
        with self._lock:
            self._values["received_frames"] += 1
            if self._values["status"] == "connecting":
                self._values["status"] = "receiving"

    def begin_session(self) -> None:
        with self._lock:
            self._completed_frames.clear()
            self._session_started_at = time.monotonic()
            self._values.update({
                "fps": 0.0,
                "processing_ms": 0.0,
                "detections": 0,
                "device": "Not started",
                "error": None,
                "status": "connecting",
                "received_frames": 0,
                "processed_frames": 0,
                "latest_rendered": None,
            })

    def snapshot(self) -> dict:
        with self._lock:
            values = dict(self._values)
            values["session_age_s"] = (
                time.monotonic() - self._session_started_at
                if self._session_started_at is not None else 0.0
            )
            return values


def prepare_camera_frame(frame: np.ndarray | None) -> np.ndarray:
    """Normalize a camera frame to a BGR uint8 array for downstream processing."""
    if frame is None:
        raise ValueError("Camera frame is empty.")
    array = np.asarray(frame)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[-1] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array.copy()


def resize_camera_frame(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    """Bound frame work while preserving the camera's aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale == 1.0:
        return frame.copy()
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def restore_detail_image(image_bgr: np.ndarray, bundle: dict) -> np.ndarray:
    """Run the checkpoint Residual CNN and return its restored image at source size."""
    residual_cnn = getattr(bundle["model"], "residual_cnn", None)
    if residual_cnn is None:
        raise RuntimeError("The selected model has no Residual CNN detail-restoration stage.")
    size = int(bundle["image_size"])
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 127.5 - 1.0
    tensor = tensor.to(bundle["device"])
    with torch.no_grad():
        restored = residual_cnn(tensor).clamp(-1, 1)
    restored_rgb = ((restored[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(restored_bgr, (width, height), interpolation=cv2.INTER_LINEAR)


def process_camera_frame(frame: np.ndarray | None, bundle: dict | None,
                         conf_thr: float = 0.5, nms_thr: float = 0.4,
                         include_diagnostics: bool = False) -> dict:
    """Run the existing enhancement, residual restoration, and detector on one frame."""
    if bundle is None:
        raise ValueError("No trained checkpoint found.")

    original = prepare_camera_frame(frame)
    if bundle.get("enhancer") is None:
        raise RuntimeError("The selected checkpoint has no configured enhancement pipeline.")
    if getattr(bundle["model"], "residual_cnn", None) is None:
        raise RuntimeError("The selected model has no Residual CNN detail-restoration stage.")

    size = int(bundle["image_size"])
    frame_for_inference = resize_camera_frame(original, max_dimension=size * 2)
    with _INFERENCE_LOCK:
        stages = bundle["enhancer"].process(frame_for_inference)
        enhanced = restore_detail_image(stages["msr"], bundle)
        result = detect_image(
            bundle,
            enhanced,
            conf_thr=conf_thr,
            nms_thr=nms_thr,
            include_diagnostics=include_diagnostics,
            skip_enhancement=True,
        )
        rendered = draw_detections(
            enhanced,
            result["boxes"],
            result["scores"],
            result["labels"],
            bundle["classes"],
        )
    return {
        "original_bgr": original,
        "enhanced_bgr": enhanced,
        "boxes": result["boxes"],
        "scores": result["scores"],
        "labels": result["labels"],
        "rendered": rendered,
        "device": str(bundle["device"]),
    }


def make_video_frame_callback(bundle_or_loader: dict | Callable[[], dict], conf_thr: float, nms_thr: float,
                              stats: LiveCameraStats, load_key: str = "camera-model"):
    """Create a low-latency callback; it never calls Streamlit APIs."""
    bundle_holder = {"bundle": bundle_or_loader if isinstance(bundle_or_loader, dict) else None}
    load_future = None
    load_error = None

    def get_bundle() -> dict:
        nonlocal load_future, load_error
        if bundle_holder["bundle"] is not None:
            return bundle_holder["bundle"]
        if load_error is not None:
            raise load_error
        if load_future is None:
            with _MODEL_LOAD_LOCK:
                load_future = _MODEL_LOAD_FUTURES.get(load_key)
                if load_future is None:
                    if not callable(bundle_or_loader):
                        raise TypeError("A callable checkpoint loader is required for live camera.")
                    load_future = _MODEL_LOAD_EXECUTOR.submit(bundle_or_loader)
                    _MODEL_LOAD_FUTURES[load_key] = load_future
        if not load_future.done():
            return None
        try:
            bundle = load_future.result()
        except Exception as exc:
            load_error = exc
            raise
        bundle["model"].eval()
        bundle_holder["bundle"] = bundle
        return bundle

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        stats.note_frame_received()
        try:
            bundle = get_bundle()
        except Exception as exc:
            stats.set_status("model_error", f"{type(exc).__name__}: {exc}")
            return frame
        if bundle is None:
            stats.set_status("loading_model")
            return frame

        started = time.perf_counter()
        try:
            result = process_camera_frame(frame.to_ndarray(format="bgr24"), bundle, conf_thr, nms_thr)
        except Exception as exc:
            stats.set_status("pipeline_error", f"{type(exc).__name__}: {exc}")
            return frame

        stats.update(
            time.monotonic(),
            (time.perf_counter() - started) * 1000.0,
            len(result["boxes"]),
            result["device"],
            result["rendered"],
        )
        return frame

    return video_frame_callback
