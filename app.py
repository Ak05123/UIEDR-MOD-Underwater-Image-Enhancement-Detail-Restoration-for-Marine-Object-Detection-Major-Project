"""Streamlit interface for the Hybrid YOLOv11 + Faster R-CNN project."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from evaluate import run_evaluation
from inference import detect_image, draw_detections, load_checkpoint_model
from live_camera import LiveCameraStats, build_rtc_configuration, camera_media_stream_constraints, make_video_frame_callback, restore_detail_image
from models.enhancement import ImageEnhancer
from streamlit_webrtc import webrtc_streamer
from training.metrics import compute_quality_metrics
from training.trainer import Trainer, build_loaders, build_model, load_config, make_optimizer
from utils.device import describe_device


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
CONFIG_PATH = ROOT / "config.yaml"

st.set_page_config(page_title="UIEDR MOD", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#edf2f2; --muted:#91a4a5; --line:#294245; --panel:#112225; --panel2:#162c2f; --aqua:#6ee7d2; --amber:#f4bb6b; --rose:#e98c84; }
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
body { color:var(--ink); }
.stApp { background: radial-gradient(circle at 100% 0%, #1b4845 0, #0b171b 35%, #081013 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #091719, #071114); border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:linear-gradient(145deg, rgba(23,50,52,.92), rgba(10,26,29,.94)); border:1px solid var(--line); padding:16px; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--ink); }
.eyebrow { color:var(--aqua); font:500 12px 'DM Mono', monospace; letter-spacing:1.5px; text-transform:uppercase; }
.hero { padding:24px 28px 30px; border:1px solid var(--line); border-radius:18px; margin-bottom:28px; background:linear-gradient(115deg, rgba(17,45,47,.9), rgba(8,20,24,.55)); box-shadow:0 18px 50px rgba(0,0,0,.16); }
.hero h1 { font-size:clamp(30px, 5vw, 58px); line-height:1.02; margin:10px 0; letter-spacing:0; max-width:820px; }
.hero p { color:var(--muted); max-width:700px; font-size:16px; }
.section { border-top:1px solid var(--line); padding-top:22px; margin-top:28px; }
.section h2 { margin:0 0 5px; }
.section p { color:var(--muted); }
.status { padding:13px 16px; border:1px solid #31585a; border-left:3px solid var(--aqua); background:rgba(17,40,42,.8); border-radius:10px; color:var(--ink); }
.warning { border-left-color:var(--amber); background:#2b2316; }
.danger { border-left-color:var(--rose); background:#2b191b; }
.mono { font-family:'DM Mono', monospace; color:var(--aqua); }
.glass { border:1px solid var(--line); border-radius:14px; padding:18px; background:rgba(13,31,34,.72); box-shadow:0 12px 34px rgba(0,0,0,.12); }
.hero-grid { display:grid; grid-template-columns:1.35fr .65fr; gap:24px; align-items:end; }
.hero-tag { color:var(--amber); font:500 12px 'DM Mono', monospace; text-transform:uppercase; letter-spacing:1.2px; }
.stat-line { display:flex; justify-content:space-between; gap:14px; border-bottom:1px solid var(--line); padding:10px 0; color:var(--muted); }
.stat-line strong { color:var(--ink); text-align:right; }
.delta { color:var(--aqua); font:500 12px 'DM Mono', monospace; }
.arch { display:flex; flex-direction:column; align-items:center; gap:7px; margin:12px auto; max-width:700px; }
.arch-step { width:100%; border:1px solid #2e5050; background:linear-gradient(100deg,#132b2d,#102124); padding:12px 16px; border-radius:8px; text-align:center; }
.arrow { color:var(--amber); font-size:20px; }
.caption { color:var(--muted); font-size:12px; }
.st-key-live-camera-card { width:100%; max-width:1100px; margin:0 auto; padding:clamp(12px, 2vw, 24px); overflow:hidden; border:1px solid var(--line); border-radius:12px; background:rgba(9,24,27,.76); }
.st-key-live-camera-card iframe { width:100% !important; max-width:100% !important; border:0; }
.st-key-live-camera-card [data-testid="stImage"] img { width:100%; height:auto; max-height:70vh; object-fit:contain; background:#081013; border-radius:8px; }
@media (max-width:720px) {
    .st-key-live-camera-card { padding:12px; border-radius:10px; }
    .st-key-live-camera-card [data-testid="stHorizontalBlock"] { flex-direction:column !important; gap:8px; }
    .st-key-live-camera-card [data-testid="column"] { width:100% !important; min-width:100% !important; flex:1 1 100% !important; }
    .st-key-live-camera-card [data-testid="stMetric"] { min-height:72px; padding:10px; }
    .st-key-live-camera-card [data-testid="stSelectbox"] [role="combobox"] { min-height:46px; }
    .st-key-live-camera-card [data-testid="stSegmentedControl"] button { min-height:48px; font-size:16px; }
    .st-key-live-camera-card iframe { min-height:240px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def read_config():
    return load_config(str(CONFIG_PATH))


@st.cache_data
def read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def checkpoint_paths():
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "checkpoints").glob("*.pt"))


@st.cache_data
def checkpoint_meta(path: str):
    try:
        checkpoint = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        return {"epoch": checkpoint.get("epoch", "?"), "smoke_test": bool(checkpoint.get("smoke_test", False)), "classes": checkpoint.get("classes", []), "training_metadata": checkpoint.get("training_metadata", {})}
    except Exception as exc:
        return {"error": str(exc), "epoch": "?", "smoke_test": False, "classes": []}


@st.cache_resource(show_spinner=False)
def cached_bundle(path: str, device: str):
    return load_checkpoint_model(path, device=device)


def image_from_upload(uploaded) -> np.ndarray:
    data = uploaded.getvalue()
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    return image


def show_bgr(image: np.ndarray, **kwargs):
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), **kwargs)


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the result image.")
    return encoded.tobytes()


def image_stats(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {"brightness": float(gray.mean()), "contrast": float(gray.std()), "min": int(image.min()), "max": int(image.max())}


def enhancement_delta(original: np.ndarray, enhanced: np.ndarray) -> dict:
    delta = np.abs(enhanced.astype(np.int16) - original.astype(np.int16))
    return {"mean_abs_change": float(delta.mean()), "changed_pixels_pct": float(np.count_nonzero(delta) / delta.size * 100.0)}


def enhancement_quality_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    if original.shape[:2] != enhanced.shape[:2]:
        raise ValueError(f"Original image size {original.shape[:2]} does not match enhanced output {enhanced.shape[:2]}")
    if original.shape != enhanced.shape:
        raise ValueError(f"Original image shape {original.shape} does not match enhanced output {enhanced.shape}")
    return compute_quality_metrics(original, enhanced)


def validate_stage(stage: np.ndarray, reference: np.ndarray) -> None:
    if stage.shape != reference.shape:
        raise ValueError(f"Stage shape changed from {reference.shape} to {stage.shape}")
    if stage.dtype != np.uint8 or int(stage.min()) < 0 or int(stage.max()) > 255:
        raise ValueError(f"Stage output must be uint8 in [0, 255], got {stage.dtype}")


def checkpoint_label(path: str) -> str:
    meta = checkpoint_meta(path)
    marker = "  ·  SMOKE TEST" if meta.get("smoke_test") else ""
    return f"{path}  ·  epoch {meta.get('epoch', '?')}{marker}"


def metric_value(value):
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def hero(title: str, subtitle: str, kicker: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def checkpoint_picker(label: str = "Checkpoint"):
    paths = checkpoint_paths()
    if not paths:
        st.warning("No checkpoint files are available in checkpoints/.")
        return None
    return st.selectbox(label, paths, format_func=checkpoint_label)


def live_camera_panel(cfg):
    checkpoints = checkpoint_paths()
    with st.container(key="live-camera-card", border=True):
        st.markdown('<div class="hero-tag">LIVE CAMERA / WEBRTC</div>', unsafe_allow_html=True)
        st.caption("Live preview and enhanced detection output. Back Camera is the default; use More camera devices when you need a specific connected device.")
        if not checkpoints:
            st.warning("Camera unavailable: add a model checkpoint before starting live detection.")
            return

        checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=lambda path: Path(path).name, key="live_checkpoint")
        conf, nms = st.columns(2)
        conf_thr = conf.slider("Confidence threshold", 0.05, 0.95, float(cfg["inference"]["confidence_threshold"]), 0.05, key="live_conf")
        nms_thr = nms.slider("NMS threshold", 0.05, 0.95, float(cfg["inference"]["nms_threshold"]), 0.05, key="live_nms")

        stats_key = f"live_camera_stats:{checkpoint}"
        if stats_key not in st.session_state:
            st.session_state[stats_key] = LiveCameraStats()
        stats = st.session_state[stats_key]
        checkpoint_path = str(ROOT / checkpoint)
        device = cfg["training"].get("device", "auto")
        camera_position_key = "live_camera_position"
        camera_position = st.session_state.get(camera_position_key, "Back Camera")
        facing_key = "front" if camera_position == "Front Camera" else "back"
        callback = make_video_frame_callback(
            lambda: load_checkpoint_model(checkpoint_path, device=device),
            conf_thr,
            nms_thr,
            stats,
            load_key=f"{checkpoint_path}:{device}",
        )

        try:
            context = webrtc_streamer(
                key=f"enhancement_live_camera_{facing_key}",
                video_frame_callback=callback,
                on_video_ended=lambda: stats.set_status("stopped"),
                media_stream_constraints=camera_media_stream_constraints(camera_position),
                async_processing=True,
                video_receiver_size=1,
                media_toggle_controls=False,
                rtc_configuration=build_rtc_configuration(cfg.get("webrtc", {})),
                video_html_attrs={
                    "autoPlay": True,
                    "playsInline": True,
                    "controls": False,
                    "style": {
                        "width": "100%",
                        "height": "auto",
                        "maxHeight": "70vh",
                        "objectFit": "contain",
                        "backgroundColor": "#081013",
                    },
                },
                translations={
                    "start": "Start Camera",
                    "stop": "Stop Camera",
                    "select_device": "More camera devices",
                    "select_camera": "Select camera device",
                    "media_api_not_available": "Camera permission requires HTTPS. On this computer, open localhost; on a phone, use the HTTPS URL.",
                    "device_ask_permission": "Camera permission required. Allow camera access in your browser prompt.",
                    "device_not_available": "Camera unavailable. Connect or enable a camera device.",
                    "device_access_denied": "Camera permission denied. Allow camera access in browser site settings.",
                },
            )
        except Exception as exc:
            st.error("Camera unavailable. Check device access, browser permission, and HTTPS requirements.")
            st.code(f"{type(exc).__name__}: {exc}")
            return

        camera_active = context.state.playing or context.state.signalling
        st.segmented_control(
            "Camera",
            ["Front Camera", "Back Camera"],
            default=camera_position,
            key=camera_position_key,
            disabled=camera_active,
            width="stretch",
            help="Stop the stream before changing camera. The selected facing mode is applied when you start again; a single-camera device falls back to its available camera.",
        )
        st.caption(f"Selected for next start: {camera_position}. A browser exposing only one camera reuses its available device.")
        if camera_active:
            st.caption("Stop Camera before selecting Front Camera or Back Camera; the next start opens the selected device mode.")

        active_key = f"live_camera_active:{checkpoint}"
        ever_started_key = f"live_camera_started:{checkpoint}"
        is_active = context.state.playing or context.state.signalling
        if is_active and not st.session_state.get(active_key):
            stats.begin_session()
            st.session_state[active_key] = True
        elif not is_active:
            st.session_state[active_key] = False
        if is_active:
            st.session_state[ever_started_key] = True

        @st.fragment(run_every=0.5)
        def live_camera_status():
            values = stats.snapshot()
            if context.state.signalling and not context.state.playing:
                st.info("Connecting to camera…")
            elif context.state.playing:
                if values["status"] == "loading_model":
                    st.info("Camera connected. Loading the selected checkpoint in the background…")
                elif values["error"]:
                    st.error(f"Live processing unavailable: {values['error']}")
                elif values["processed_frames"]:
                    st.success("Camera connected · live frames are being processed.")
                elif values["received_frames"]:
                    st.info("Camera frames are reaching the app. Waiting for the first enhanced detection result…")
                elif values["session_age_s"] > 8:
                    st.error("Camera unavailable: no frames reached the app. Check browser camera permission and WebRTC/ICE connectivity.")
                else:
                    st.info("Connecting to the camera and waiting for the first frame…")
            elif values["error"]:
                st.error(f"Camera unavailable: {values['error']}")
            elif st.session_state.get(ever_started_key):
                st.info("Camera stopped. Press Start Camera to resume.")
            else:
                st.info("Camera permission required. Press Start Camera and allow browser access; connect a camera if none is detected.")

            frames = values["processed_frames"]
            metrics = st.columns(4)
            metrics[0].metric("Actual FPS", f"{values['fps']:.2f}" if frames > 1 else "—")
            metrics[1].metric("Processing time", f"{values['processing_ms']:.1f} ms" if frames else "—")
            metrics[2].metric("Detection count", values["detections"] if frames else "—")
            metrics[3].metric("Device", values["device"] if frames else "—")
            st.caption(f"Frames received: {values['received_frames']} · frames processed: {frames}")
            if values["latest_rendered"] is not None:
                st.subheader("Enhanced + Detection output")
                show_bgr(values["latest_rendered"], width="stretch")
            else:
                st.caption("Enhanced + Detection output will appear after the first processed camera frame.")

        live_camera_status()


def dashboard(cfg):
    runtime = describe_device()
    st.markdown(f'<div class="hero"><div class="hero-grid"><div><div class="eyebrow">MAJOR PROJECT · COMPUTER VISION · UNDERWATER OBJECT DETECTION</div><h1>UIEDR MOD</h1><p>Underwater image enhancement and restoration for marine object detection.</p><div class="hero-tag">Hybrid detector</div></div><div class="glass"><div class="hero-tag">Live project state</div><div class="stat-line"><span>Dataset</span><strong>DUO / RAW-890</strong></div><div class="stat-line"><span>Classes</span><strong>{len(cfg.get("classes", []))}</strong></div><div class="stat-line"><span>Input</span><strong>{cfg["dataset"]["image_size"]} × {cfg["dataset"]["image_size"]}</strong></div><div class="stat-line"><span>Runtime</span><strong>{runtime}</strong></div></div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="status">Operational surface · all inference, evaluation, and training actions are user-triggered.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section"><h2>System snapshot</h2><p>Real artifacts from the current workspace, with unavailable performance called out instead of inferred.</p></div>', unsafe_allow_html=True)
    checkpoints = checkpoint_paths()
    smoke = []
    for path in checkpoints:
        try:
            smoke.append(bool(torch.load(ROOT / path, map_location="cpu", weights_only=False).get("smoke_test", False)))
        except Exception:
            pass
    report = read_json(str(ROOT / "reports" / "dataset_report.json")) if (ROOT / "reports" / "dataset_report.json").exists() else {}
    train = report.get("train", {})
    eval_path = ROOT / "reports" / "eval_best_real_10.json"
    evaluation = read_json(str(eval_path)) if eval_path.exists() else {}
    best = next((p for p in checkpoints if Path(p).name == "best.pt"), None)
    cols = st.columns(5)
    cols[0].metric("Architecture", "Hybrid detector")
    cols[1].metric("Classes", len(cfg.get("classes", [])))
    cols[2].metric("Train images", train.get("num_images_declared", "Not validated"))
    cols[3].metric("Best checkpoint", "Ready" if best else "Not trained")
    cols[4].metric("Evaluation", "Available" if evaluation else "Not evaluated")
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Current checkpoint")
        if checkpoints:
            st.write(checkpoint_label(best or checkpoints[0]))
            if any(smoke):
                st.markdown('<div class="status warning">The available smoke checkpoint is an integration artifact, not a trained final model.</div>', unsafe_allow_html=True)
        else:
            st.info("Add a .pt checkpoint to checkpoints/ to enable inference and evaluation.")
    with right:
        st.subheader("Runtime")
        st.write(f"Device: **{describe_device()}**")
        st.write(f"Input: **{cfg['dataset']['image_size']} × {cfg['dataset']['image_size']}**")
        st.write(f"Dataset: **{cfg['dataset']['raw_dir']}**")
    st.markdown('<div class="section"><h2>Capability board</h2><p>A concise readout of what is present in the workspace today.</p></div>', unsafe_allow_html=True)
    board = st.columns(3)
    with board[0]:
        st.markdown('<div class="glass"><div class="hero-tag">Data foundation</div><h3>DUO dataset</h3><p>6,671 train images · 1,111 test images · four underwater classes.</p><span class="delta">VALIDATED</span></div>', unsafe_allow_html=True)
    with board[1]:
        status = "REAL METRICS" if evaluation else "NOT EVALUATED"
        st.markdown(f'<div class="glass"><div class="hero-tag">Model status</div><h3>{status}</h3><p>{best or "No checkpoint available"}</p><span class="delta">{"BEST CHECKPOINT" if best else "NOT TRAINED"}</span></div>', unsafe_allow_html=True)
    with board[2]:
        st.markdown('<div class="glass"><div class="hero-tag">Architecture</div><h3>Enhance → detect</h3><p>White Balance · CLAHE · MSR · Residual CNN · FPN/PAN · RPN · ROI Align.</p><span class="delta">PIPELINE READY</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section"><h2>Quick actions</h2><p>Use the navigation to move from dataset inspection to enhancement, detection, and evaluation.</p></div>', unsafe_allow_html=True)
    st.info("Choose Detection, Batch Inference, or Evaluation in the navigation. The dashboard never runs a model pass automatically.")


def dataset_page(cfg):
    hero("Dataset observatory", "The official DUO train/test split, read directly from RAW-890 annotations and validation reports.", "DATASET / DUO")
    report_path = ROOT / "reports" / "dataset_report.json"
    if not report_path.exists():
        st.warning("No dataset validation report is available yet.")
        return
    report = read_json(str(report_path))
    raw_candidates = [ROOT / "dataset" / "DUO" / "RAW-890", ROOT / "dataset" / "raw-890", ROOT / "raw-890"]
    raw_path = next((path for path in raw_candidates if path.exists()), None)
    raw_label = str(raw_path.relative_to(ROOT)).replace("\\", "/") if raw_path else "not found"
    st.info(f"Training dataset: DUO at `{cfg['dataset']['raw_dir']}`. Additional image collection: `{raw_label}` (unlabeled; not used for supervised training). The configured validation split is the official DUO test split.")
    tabs = st.tabs(["Overview", "Class distribution", "Real samples"])
    for tab, split in zip(tabs[:2], ["train", "val"]):
        with tab:
            item = report.get(split, {})
            cols = st.columns(4)
            cols[0].metric(f"{split.title()} images", item.get("num_images_declared", "n/a"))
            cols[1].metric("Annotations", item.get("num_annotations", "n/a"))
            cols[2].metric("Empty images", item.get("empty_images", "n/a"))
            cols[3].metric("Missing images", len(item.get("missing_images", [])))
            st.write("Classes:", ", ".join(item.get("classes", cfg.get("classes", []))))
            st.json({"annotation_file": item.get("annotation_file"), "image_dir": item.get("image_dir"), "image_sizes": item.get("image_sizes", {})})
    with tabs[1]:
        st.bar_chart({split: report.get(split, {}).get("class_distribution", {}) for split in ("train", "val")})
    with tabs[2]:
        split = st.selectbox("Sample split", ["train", "val"], format_func=lambda v: "Train" if v == "train" else "Test / validation")
        item = report[split]
        annotation = read_json(str(ROOT / item["annotation_file"]))
        image_dir = ROOT / item["image_dir"]
        count = st.slider("Samples", 1, 8, 4)
        shown = 0
        cols = st.columns(4)
        for record in annotation.get("images", []):
            path = image_dir / record["file_name"]
            if not path.exists():
                continue
            with cols[shown % 4]:
                image = cv2.imread(str(path))
                show_bgr(image, caption=record["file_name"], use_container_width=True)
            shown += 1
            if shown >= count:
                break
        st.caption(f"Validation report: {report_path.relative_to(ROOT)}")


def enhancement_page(cfg):
    hero("Enhancement", "Trace every available restoration stage from the source pixels through the learned residual detail path.", "UIEDR MOD / PREPROCESSING")

    tabs = st.tabs(["Upload Image", "Live Camera"])

    with tabs[1]:
        live_camera_panel(cfg)

    with tabs[0]:
        uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png", "bmp"], key="enhance_upload")
        if uploaded is None:
            st.info("Upload an image to inspect the real enhancement stages.")
            return
        try:
            original = image_from_upload(uploaded)
            stages = ImageEnhancer(cfg["model"].get("enhancement", {})).process(original)
            for key in ("original", "white_balance", "clahe", "msr", "enhanced"):
                validate_stage(stages[key], original)
            checkpoints = checkpoint_paths()
            detail_checkpoint = next((p for p in checkpoints if Path(p).name == "best.pt"), None)
            if detail_checkpoint:
                detail_bundle = cached_bundle(str(ROOT / detail_checkpoint), cfg["training"].get("device", "auto"))
                stages["detail_restoration"] = restore_detail_image(stages["msr"], detail_bundle)
            else:
                stages["detail_restoration"] = stages["msr"].copy()
            validate_stage(stages["detail_restoration"], original)
            stages["final"] = stages["detail_restoration"]

            quality_metrics = enhancement_quality_metrics(original, stages["final"])

            st.session_state["enhancement_original"] = original.copy()
            st.session_state["enhanced_image"] = stages["final"].copy()
            st.session_state["enhancement_metrics"] = quality_metrics

            debug_dir = ROOT / "outputs" / "visualizations" / "enhancement_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for key, filename in {
                "original": "original.png",
                "white_balance": "white_balance.png",
                "clahe": "clahe.png",
                "msr": "msr.png",
                "detail_restoration": "residual.png",
                "final": "final_enhanced.png",
            }.items():
                save_path = debug_dir / filename
                ok, encoded = cv2.imencode(".png", stages[key])
                if not ok:
                    raise ValueError(f"Could not encode enhancement stage {key}.")
                save_path.write_bytes(encoded.tobytes())
        except Exception as exc:
            st.error(f"Could not enhance this image: {exc}")
            return
        original_stats = image_stats(stages["original"])
        final_stats = image_stats(stages["final"])
        delta = enhancement_delta(stages["original"], stages["final"])
        st.markdown('<div class="status">Enhancement verified · every stage was executed and validated as an H × W × 3 uint8 image in the source pixel range.</div>', unsafe_allow_html=True)
        stat_cols = st.columns(4)
        stat_cols[0].metric("Brightness", f"{original_stats['brightness']:.1f} → {final_stats['brightness']:.1f}")
        stat_cols[1].metric("Contrast", f"{original_stats['contrast']:.1f} → {final_stats['contrast']:.1f}")
        stat_cols[2].metric("Mean pixel change", f"{delta['mean_abs_change']:.1f}")
        stat_cols[3].metric("Changed pixels", f"{delta['changed_pixels_pct']:.1f}%")

        st.subheader("Enhancement Metrics")
        st.caption("Metrics compare the enhanced image with the original input image.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("PSNR", f"{quality_metrics['psnr']:.2f} dB")
        metric_cols[1].metric("SSIM", f"{quality_metrics['ssim']:.4f}")
        metric_cols[2].metric("MSE", f"{quality_metrics['mse']:.4f}")

        before, after = st.columns(2)
        with before:
            show_bgr(stages["original"], caption="Original Image", use_container_width=True)
        with after:
            show_bgr(stages["final"], caption="Enhanced / Restored Image", use_container_width=True)
        if st.button("Continue to Detection", type="primary"):
            st.session_state["next_page"] = "Detection"
            st.rerun()
        with st.expander("Inspect every real processing stage", expanded=True):
            labels = [("Original", "original"), ("White balance", "white_balance"), ("CLAHE", "clahe"), ("MSR / Retinex", "msr"), ("Detail restoration · Residual CNN", "detail_restoration"), ("Final enhanced", "final")]
            cols = st.columns(3)
            for col, (label, key) in zip(cols, labels):
                with col:
                    show_bgr(stages[key], caption=label, use_container_width=True)
                    stats = image_stats(stages[key])
                    previous = stages[labels[max(0, labels.index((label, key)) - 1)][1]]
                    diff = enhancement_delta(previous, stages[key])
                    st.caption(f"{stages[key].shape[0]} × {stages[key].shape[1]} × {stages[key].shape[2]} · {stages[key].dtype} · mean {stats['brightness']:.1f} · std {stats['contrast']:.1f} · Δ {diff['mean_abs_change']:.2f}")

def safe_dataframe(rows, **kwargs):
    try:
        import pandas as pd  # lazy optional dependency used only for table rendering
        if rows:
            frame = pd.DataFrame(rows)
            st.dataframe(frame, **kwargs)
        else:
            st.write(rows)
    except Exception as exc:
        st.info("Evaluation dependency unavailable on this Windows installation.")
        st.code(f"Dependency import error: {type(exc).__name__}: {exc}")
        st.write(rows)


def detection_page(cfg):
    hero("Detection", "Run object detection on the enhanced output produced by the Enhancement stage.", "UIEDR MOD / DETECT → LABEL")
    checkpoint = checkpoint_picker()
    enhanced_image = st.session_state.get("enhanced_image")
    if enhanced_image is not None:
        original_image = st.session_state.get("enhancement_original")
        st.success("Using the enhanced image from the Enhancement stage. The detector is receiving the restored output directly without re-running enhancement.")
        st.caption("Detection input: enhanced/restored image produced in memory from the previous stage.")
    else:
        st.warning("Please run Enhancement first.")
        return
    conf, nms = st.columns(2)
    conf_thr = conf.slider("Confidence threshold", 0.05, 0.95, float(cfg["inference"]["confidence_threshold"]), 0.05)
    nms_thr = nms.slider("NMS threshold", 0.05, 0.95, float(cfg["inference"]["nms_threshold"]), 0.05)
    show_diag = st.checkbox("Show Detection Diagnostics", value=False)
    if checkpoint:
        meta = checkpoint_meta(checkpoint)
        st.caption(f"Selected checkpoint: `{checkpoint}` · epoch {meta.get('epoch', '?')} · {'smoke test' if meta.get('smoke_test') else 'real checkpoint'}")
    run = st.button("Run detection", type="primary", disabled=not checkpoint)
    if not run:
        return
    try:
        started = time.perf_counter()
        bundle = cached_bundle(str(ROOT / checkpoint), cfg["training"].get("device", "auto"))
        result = detect_image(bundle, enhanced_image, conf_thr, nms_thr, include_diagnostics=show_diag, skip_enhancement=True)
        rendered = draw_detections(enhanced_image, result["boxes"], result["scores"], result["labels"], bundle["classes"])
        elapsed = time.perf_counter() - started
        output_path = ROOT / cfg["outputs"]["predictions_dir"] / "streamlit_enhancement_detection.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f"Could not save detection result to {output_path}")
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
        return

    st.success(f"Enhanced input → Detector → NMS → Detection results · completed on {bundle['device']}")
    training_metadata = meta.get("training_metadata") or {}
    is_full_cuda_checkpoint = (
        training_metadata.get("device") == "cuda"
        and training_metadata.get("num_training_samples") == 6671
        and bool(training_metadata.get("source_hybrid_detector_sha256"))
    )
    model_status = "Full DUO CUDA checkpoint" if is_full_cuda_checkpoint else "Legacy/diagnostic checkpoint; full CUDA training not verified"
    stat_cols = st.columns(4)
    stat_cols[0].metric("Model checkpoint", Path(checkpoint).name)
    stat_cols[1].metric("Model status", "CUDA-trained" if is_full_cuda_checkpoint else "Not full-DUO verified")
    stat_cols[2].metric("Device", str(bundle["device"]))
    stat_cols[3].metric("Inference time", f"{elapsed:.2f}s")
    st.caption(model_status)
    st.markdown('<div class="glass"><div class="hero-tag">Actual pipeline</div><p class="mono">ORIGINAL → WHITE BALANCE → CLAHE → MSR → DETAIL RESTORATION → FEATURE EXTRACTION → FPN/PAN → RPN → ROI ALIGN → COORDINATE ATTENTION → CLASSIFICATION + BOX REGRESSION → NMS → DETECTION RESULTS</p></div>', unsafe_allow_html=True)
    if len(result["boxes"]):
        st.success(f"{len(result['boxes'])} detections passed the selected thresholds.")
    else:
        st.markdown('<div class="status danger"><strong>NO CONFIDENT OBJECTS DETECTED</strong><br>The current checkpoint did not produce detections above the selected thresholds. A trained DUO checkpoint is required for real underwater detections.</div>', unsafe_allow_html=True)

    if show_diag:
        model_diag = result["diagnostics"]
        st.json({
            "Checkpoint": checkpoint,
            "Checkpoint epoch": meta.get("epoch", "?"),
            "Device": str(bundle["device"]),
            "Input size (model)": model_diag["input_size"],
            "Input size (original)": model_diag["original_size"],
            "RPN proposals": model_diag["rpn_proposals"],
            "Maximum RPN objectness": round(model_diag["max_rpn_objectness"], 6),
            "ROI proposals": model_diag["roi_proposals"],
            "ROI foreground candidates before NMS": model_diag["roi_foreground_candidates"],
            "Maximum foreground confidence": round(model_diag["max_foreground_confidence"], 6),
            "Final detections": model_diag["final_detections"],
            "Predicted classes": [bundle["classes"][idx] for idx in model_diag["predicted_classes"]],
            "Predicted boxes (x1, y1, x2, y2)": model_diag["predicted_boxes_xyxy"],
            "Class distribution": {bundle["classes"][int(lbl)]: int((result["labels"] == lbl).sum()) for lbl in sorted(set(int(x) for x in result["labels"]))} if len(result["boxes"]) else {},
        })
    left, right = st.columns(2)
    with left:
        show_bgr(original_image if original_image is not None else enhanced_image, caption="Original input", use_container_width=True)
    with right:
        show_bgr(rendered, caption="Detection Results", use_container_width=True)
    with st.expander("View enhanced input used by detection", expanded=False):
        show_bgr(result["enhanced_bgr"], caption="Enhanced input used for detection", use_container_width=True)
    rows = []
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        class_id = int(label)
        class_name = bundle["classes"][class_id] if 0 <= class_id < len(bundle["classes"]) else f"unknown_{class_id}"
        rows.append({"Class": class_name, "Confidence": round(float(score), 4), "X1": round(float(box[0]), 1), "Y1": round(float(box[1]), 1), "X2": round(float(box[2]), 1), "Y2": round(float(box[3]), 1)})
    st.subheader(f"Detected Objects: {len(rows)}")
    if rows:
        safe_dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.write("No confident objects detected.")
    st.caption(f"Saved result: `{output_path.relative_to(ROOT)}`")
    st.download_button("Download annotated PNG", png_bytes(rendered), file_name="detection.png", mime="image/png")


def batch_page(cfg):
    hero("Batch", "Process a small uploaded set with the same real checkpoint and renderable outputs.", "INFERENCE / GALLERY")
    checkpoint = checkpoint_picker()
    uploads = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="batch")
    run = st.button("Run batch inference", type="primary", disabled=not (checkpoint and uploads))
    if not run:
        return
    try:
        bundle = cached_bundle(str(ROOT / checkpoint), cfg["training"].get("device", "auto"))
    except Exception as exc:
        st.error(f"Checkpoint could not be loaded: {exc}")
        return
    st.caption(f"Selected checkpoint: `{checkpoint}` · outputs are generated only from uploaded images.")
    for index, upload in enumerate(uploads):
        try:
            image = image_from_upload(upload)
            result = detect_image(bundle, image)
            rendered = draw_detections(image, result["boxes"], result["scores"], result["labels"], bundle["classes"])
            with st.container(border=True):
                left, right = st.columns([1, 2])
                with left:
                    show_bgr(rendered, caption=f"{upload.name} · {len(result['boxes'])} detections", use_container_width=True)
                with right:
                    if len(result["boxes"]):
                        st.success(f"{len(result['boxes'])} real detections")
                    else:
                        st.info("No detections passed the current checkpoint thresholds.")
                    st.write({"file": upload.name, "detections": len(result["boxes"]), "classes": [bundle["classes"][int(x)] for x in result["labels"]]})
                    st.download_button("Download result", png_bytes(rendered), file_name=f"annotated_{index + 1}.png", mime="image/png", key=f"dl_{index}")
        except Exception as exc:
            st.error(f"{upload.name}: Failed\nReason: {type(exc).__name__}: {exc}")


def evaluation_page(cfg):
    hero("Evaluation room", "Compute real detection metrics on the configured DUO test split using the selected checkpoint.", "EVALUATION / METRICS")
    checkpoint = checkpoint_picker()
    max_images = st.number_input("Number of test images", min_value=1, max_value=1111, value=10, step=1)
    conf, nms = st.columns(2)
    conf_thr = conf.slider("Confidence", 0.05, 0.95, 0.5, 0.05, key="eval_conf")
    nms_thr = nms.slider("NMS", 0.05, 0.95, 0.4, 0.05, key="eval_nms")
    run = st.button("Run evaluation", type="primary", disabled=not checkpoint)
    if not run:
        return
    try:
        with st.spinner(f"Evaluating {max_images} real DUO images..."):
            bundle, metrics, _, _, elapsed = run_evaluation(cfg, str(ROOT / checkpoint), int(max_images), conf_thr, nms_thr, cfg["training"].get("batch_size", 4))
    except ImportError as exc:
        st.error("Evaluation dependency unavailable on this Windows installation.")
        st.code(f"Dependency import error: {type(exc).__name__}: {exc}")
        return
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return
    if bundle.get("smoke_test"):
        st.warning("SMOKE-TEST CHECKPOINT: these metrics describe an integration checkpoint, not a trained final model.")
    if bundle.get("epoch", 0) != "?" and int(bundle.get("epoch", 0)) <= 2:
        st.info("Current checkpoint has not been fully trained on the complete DUO dataset.")
    st.caption(f"Evaluated {max_images} images in {elapsed:.1f}s on {bundle['device']}.")
    cols = st.columns(5)
    for col, key in zip(cols, ["precision", "recall", "f1", "mAP@0.50", "mAP@0.50:0.95"]):
        col.metric(key, metric_value(metrics.get(key)))
    st.subheader("Per-class AP")
    rows = [{"class": name, "AP @ 0.50": metric_value(values.get("ap@0.50")), "AP @ 0.50:0.95": metric_value(values.get("ap@0.50:0.95"))} for name, values in metrics.get("per_class", {}).items()]
    safe_dataframe(rows, use_container_width=True, hide_index=True)


def run_one_batch_sanity(cfg: dict) -> dict:
    """Exercise one real batch through the existing Trainer model and optimizer."""
    started = time.perf_counter()
    trainer = Trainer(cfg, run_name="streamlit_sanity")
    batch = next(iter(trainer.train_dl))
    images = batch["images"].to(trainer.device)
    boxes = [value.to(trainer.device) for value in batch["boxes"]]
    labels = [value.to(trainer.device) for value in batch["labels"]]
    trainer.model.train()
    losses = trainer.model(images, gt_boxes=boxes, gt_labels=labels)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError(f"non-finite training loss: {total.item()}")
    trainer.opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)
    trainer.opt.step()
    return {"loss": float(total.detach()), "batch_size": int(images.shape[0]), "seconds": time.perf_counter() - started}


def training_page(cfg):
    hero("Training control", "Configure a run explicitly. Nothing starts when the app opens.", "UIEDR MOD / EXPERIMENT CONTROL")
    if "training_status" not in st.session_state:
        st.session_state.training_status = "READY"
    state = st.session_state.training_status
    state_class = "status warning" if state == "TRAINING" else "status danger" if state == "FAILED" else "status"
    st.markdown(f'<div class="{state_class}"><strong>{state}</strong><br>{"Local CPU training is available for debugging only. Full DUO training should be performed on an NVIDIA CUDA GPU." if str(describe_device()).startswith("CPU") else "CUDA training available"}</div>', unsafe_allow_html=True)
    if state == "FAILED" and st.session_state.get("training_error"):
        with st.expander("Training error details", expanded=True):
            st.exception(st.session_state.training_error)
    if st.session_state.get("training_result"):
        result = st.session_state.training_result
        if "loss" in result:
            st.success(f"Sanity run completed · loss {result['loss']:.4f} · {result['seconds']:.1f}s")
        else:
            st.success(f"Training completed · {result.get('epochs', '?')} epochs · best validation loss {result.get('best_loss', float('nan')):.4f}")
    st.warning("Diagnostic overfit test: available locally. Full DUO training: recommended on Kaggle/NVIDIA GPU.")
    sanity = st.button("Run one-batch real-data sanity check", type="secondary", disabled=state == "TRAINING")
    if sanity:
        st.session_state.training_status = "TRAINING"
        st.session_state.training_error = None
        try:
            result = run_one_batch_sanity(copy.deepcopy(cfg))
            st.session_state.training_result = result
            st.session_state.training_status = "COMPLETED"
        except Exception as exc:
            st.session_state.training_error = exc
            st.session_state.training_status = "FAILED"
        st.rerun()
    tc = cfg["training"]
    epochs = st.number_input("Epochs", 1, 1000, int(tc["epochs"]))
    batch_size = st.number_input("Batch size", 1, 64, int(tc["batch_size"]))
    learning_rate = st.number_input("Learning rate", 0.000001, 1.0, float(tc["learning_rate"]), format="%.6f")
    options = ["None"] + checkpoint_paths()
    resume = st.selectbox("Resume checkpoint", options, format_func=lambda p: "Start fresh" if p == "None" else checkpoint_label(p))
    run_name = st.text_input("Run name", "streamlit_run")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        st.error("Configured dataset training is disabled on CPU. Use the one-batch diagnostic here and run full DUO training on Kaggle/NVIDIA CUDA.")
    launch = st.button("Launch configured training", type="primary", disabled=state == "TRAINING" or not cuda_available)
    if not launch:
        return
    if not torch.cuda.is_available():
        st.error("Full DUO training requires an NVIDIA CUDA GPU; local CPU training is debugging-only.")
        return
    st.session_state.training_status = "TRAINING"
    st.session_state.training_error = None
    run_cfg = copy.deepcopy(cfg)
    run_cfg["training"].update({"epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate)})
    handler = None
    status = None
    try:
        trainer = Trainer(run_cfg, run_name=run_name.strip() or "streamlit_run")
        if resume != "None":
            trainer.resume(str(ROOT / resume))
        log_box = st.empty()
        with st.status("Training in progress", expanded=True) as status:
            class StreamlitHandler(logging.Handler):
                def emit(self, record):
                    log_box.code(self.format(record))
            handler = StreamlitHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            trainer.logger.addHandler(handler)
            best = trainer.fit(max_epochs=int(epochs))
            st.session_state.training_status = "COMPLETED"
            st.session_state.training_result = {"best_loss": float(best), "epochs": int(epochs)}
            status.update(label=f"Training complete · best validation loss {best:.4f}", state="complete")
    except Exception as exc:
        st.session_state.training_status = "FAILED"
        st.session_state.training_error = exc
        if status is not None:
            status.update(label="Training failed", state="error")
        st.error("Training failed. Expand the error details above after rerun.")
    finally:
        if handler is not None:
            trainer.logger.removeHandler(handler)


def architecture_page(cfg):
    hero("Architecture map", "The implemented path from underwater pixels to class-aware boxes.", "UIEDR MOD / SYSTEM DESIGN")
    steps = ["Input", "White Balance", "CLAHE", "MSR / Retinex", "Residual CNN · detail restoration", "Feature extraction backbone", "FPN/PAN · P3 / P4 / P5", "RPN", "ROI Align", "Coordinate Attention", "Classification + Bounding Box Regression", "NMS", "Final Detection"]
    st.markdown('<div class="arch">' + "".join(f'<div class="arch-step">{step}</div><div class="arrow">↓</div>' for step in steps)[:-32] + '</div>', unsafe_allow_html=True)
    st.subheader("Configured tensor flow")
    st.write(f"Input is resized to `{cfg['dataset']['image_size']} × {cfg['dataset']['image_size']}`. Neck outputs are configured as P3/P4/P5 channels `{cfg['model']['neck']['out_channels']}`.")
    checkpoint = checkpoint_picker("Checkpoint for shape inspection")
    if checkpoint and st.button("Inspect actual tensor shapes"):
        try:
            bundle = cached_bundle(str(ROOT / checkpoint), cfg["training"].get("device", "auto"))
            size = bundle["image_size"]
            with st.spinner("Running one shape-only model pass..."):
                shapes = bundle["model"].shape_summary(torch.zeros(1, 3, size, size, device=bundle["device"]))
            st.table([{"stage": name, "shape": " × ".join(map(str, shape))} for name, shape in shapes.items()])
        except Exception as exc:
            st.error(f"Shape inspection failed: {exc}")


def about_page(cfg):
    hero("UIEDR MOD", "Underwater image enhancement and restoration for marine object detection.", "MAJOR PROJECT / CONTEXT")
    st.markdown("""
    ### Methodology
    UIEDR MOD combines underwater image enhancement, detail restoration, hybrid feature extraction, FPN/PAN, region proposals, ROI Align, Coordinate Attention, classification, and bounding-box regression.

    ### Technologies
    PyTorch, TorchVision, OpenCV, RAW-890 annotations, YAML configuration, and Streamlit. The interface calls the project's existing inference, evaluation, enhancement, dataset, and training modules.

    ### Dataset
    DUO provides four underwater object classes: holothurian, echinus, scallop, and starfish. The configured official test split is used for validation and evaluation.

    ### Limitations
    Performance depends on the selected checkpoint and the CPU/GPU available to the environment. Uploaded images are processed locally by the running app. A smoke-test checkpoint validates integration but is not a trained final model.
    """)
    st.caption(f"Configuration source: {CONFIG_PATH.relative_to(ROOT)}")


def main():
    cfg = read_config()
    with st.sidebar:
        st.markdown("<div class='eyebrow'>UIEDR MOD</div>", unsafe_allow_html=True)
        st.markdown("### Underwater detection")
        page = st.radio("Navigate", ["Dashboard", "Enhancement", "Detection", "Batch", "Evaluation", "Training", "Architecture", "About"], label_visibility="collapsed", key="nav_select")
        st.session_state["radio_page"] = page
        st.divider()
        st.caption("DUO · 4 classes")
        st.caption(f"Device · {describe_device()}")
    pages = {"Dashboard": dashboard, "Enhancement": enhancement_page, "Detection": detection_page, "Batch": batch_page, "Evaluation": evaluation_page, "Training": training_page, "Architecture": architecture_page, "About": about_page}
    try:
        if st.session_state.get("next_page"):
            page = st.session_state.pop("next_page")
            st.session_state["radio_page"] = page
        elif "radio_page" in st.session_state:
            page = st.session_state["radio_page"]
        pages[page](cfg)
    except Exception as exc:
        st.error(f"The selected page could not be rendered: {exc}")


if __name__ == "__main__":
    main()