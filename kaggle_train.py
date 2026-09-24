"""Kaggle CUDA orchestrator for full UIEDR MOD training.

This script is intentionally not a local-training shortcut. It exits before
any training command when CUDA is unavailable, validates the complete DUO
dataset, delegates training/evaluation/inference to the existing project
entry points, and writes a factual run summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify_runtime() -> tuple[object, dict]:
    import cv2
    import numpy
    import torch
    import torchvision
    import yaml

    print("Python:", sys.version.split()[0])
    print("NumPy:", numpy.__version__)
    print("OpenCV:", cv2.__version__)
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Full DUO training requires a CUDA GPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        return torch, yaml.safe_load(handle)


def verify_dataset(cfg: dict) -> dict:
    from training.trainer import validate_dataset

    dataset = cfg["dataset"]
    required = [
        dataset["annotation_train"], dataset["annotation_val"],
        dataset["image_dir_train"], dataset["image_dir_val"],
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing DUO paths: {missing}")
    reports, report_path = validate_dataset(cfg)
    train = reports["train"]
    test = reports["val"]
    if train["num_images_declared"] != 6671 or test["num_images_declared"] != 1111:
        raise RuntimeError(f"Unexpected DUO image counts: train={train['num_images_declared']} test={test['num_images_declared']}")
    if train["num_annotations"] != 63998 or test["num_annotations"] != 10517:
        raise RuntimeError(f"Unexpected DUO annotation counts: train={train['num_annotations']} test={test['num_annotations']}")
    for name, report in reports.items():
        if report["missing_images"] or report["invalid_boxes"] or report["unknown_class_ids"]:
            raise RuntimeError(f"DUO {name} validation failed: {report}")
    print("DUO validation report:", report_path)
    print("DUO train images/annotations:", train["num_images_declared"], train["num_annotations"])
    print("DUO test images/annotations:", test["num_images_declared"], test["num_annotations"])
    print("DUO classes:", cfg["classes"])
    return reports


def write_summary(torch, cfg: dict, started: float, eval_report: dict) -> None:
    checkpoint_path = ROOT / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    log_path = ROOT / "outputs" / "metrics" / "train_log_kaggle.jsonl"
    history = []
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "epochs_completed": len(history),
        "training_images": 6671,
        "test_images": 1111,
        "training_time_s": round(time.time() - started, 1),
        "best_epoch": checkpoint.get("epoch"),
        "best_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "precision": eval_report.get("precision"),
        "recall": eval_report.get("recall"),
        "f1": eval_report.get("f1"),
        "mAP@0.50": eval_report.get("mAP@0.50"),
        "mAP@0.50:0.95": eval_report.get("mAP@0.50:0.95"),
        "per_class": eval_report.get("per_class", {}),
    }
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "kaggle_training_summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["UIEDR MOD KAGGLE TRAINING SUMMARY", "=" * 40]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    (reports_dir / "kaggle_training_summary.txt").write_text("\n".join(lines))
    print("Summary:", reports_dir / "kaggle_training_summary.json")


def main() -> None:
    started = time.time()
    torch, cfg = verify_runtime()
    verify_dataset(cfg)
    for directory in ["checkpoints", "outputs/metrics", "outputs/predictions"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    run([sys.executable, "train.py", "--epochs", "50", "--batch-size", "4",
         "--device", "cuda", "--num-workers", "4", "--run-name", "kaggle"])
    run([sys.executable, "evaluate.py", "--checkpoint", "checkpoints/best.pt",
         "--out", "kaggle_best"])

    test_dir = ROOT / cfg["dataset"]["image_dir_val"]
    images = sorted(test_dir.glob("*.jpg"))[:5]
    if len(images) < 5:
        images = sorted(test_dir.glob("*"))[:5]
    for image in images:
        output = ROOT / "outputs" / "predictions" / f"{image.stem}_kaggle_best.jpg"
        run([sys.executable, "infer.py", "--image", str(image),
             "--checkpoint", "checkpoints/best.pt", "--out", str(output)])

    eval_report = json.loads((ROOT / "reports" / "eval_kaggle_best.json").read_text())
    write_summary(torch, cfg, started, eval_report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Kaggle training stopped: {exc}", file=sys.stderr)
        raise