"""Run complete UIEDR MOD training and evaluation on a CUDA GPU.

Start from the extracted training package root with:
    python kaggle_train.py

The runner rejects existing checkpoints so an old local model cannot be
mistaken for this run. It retries CUDA out-of-memory failures by lowering only
the batch size and resuming from the latest completed epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_CLASSES = ["holothurian", "echinus", "scallop", "starfish"]


def verify_runtime():
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
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Do not run full training on CPU.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("classes") != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected class order: {cfg.get('classes')}")
    return torch, cfg


def verify_dataset(cfg):
    from training.trainer import validate_dataset

    dataset = cfg["dataset"]
    train_ann_path = ROOT / dataset["annotation_train"]
    test_ann_path = ROOT / dataset["annotation_val"]
    train_dir = ROOT / dataset["image_dir_train"]
    test_dir = ROOT / dataset["image_dir_val"]
    for path in (train_ann_path, test_ann_path, train_dir, test_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing required DUO path: {path}")

    train_json = json.loads(train_ann_path.read_text(encoding="utf-8"))
    test_json = json.loads(test_ann_path.read_text(encoding="utf-8"))
    train_files = [path for path in train_dir.iterdir() if path.is_file()]
    test_files = [path for path in test_dir.iterdir() if path.is_file()]
    counts = {
        "training_images": len(train_json["images"]),
        "training_files": len(train_files),
        "training_annotations": len(train_json["annotations"]),
        "test_images": len(test_json["images"]),
        "test_files": len(test_files),
        "test_annotations": len(test_json["annotations"]),
    }
    print("DUO counts:", json.dumps(counts, sort_keys=True))
    if counts != {
        "training_images": 6671, "training_files": 6671,
        "training_annotations": 63998, "test_images": 1111,
        "test_files": 1111, "test_annotations": 10517,
    }:
        raise RuntimeError(f"The complete official DUO split is required: {counts}")

    reports, report_path = validate_dataset(cfg)
    for split, report in reports.items():
        if report["missing_images"] or report["invalid_boxes"] or report["unknown_class_ids"]:
            raise RuntimeError(f"DUO {split} validation failed: {report}")
    print("Dataset validation report:", report_path)
    return counts


def train_with_oom_recovery(torch, args, started_at):
    batch_size = args.batch_size
    resume_path = None
    while True:
        command = [
            sys.executable, "train.py", "--epochs", str(args.epochs),
            "--batch-size", str(batch_size), "--device", "cuda",
            "--num-workers", str(args.num_workers),
            "--early-stopping-patience", "0", "--run-name", "kaggle",
        ]
        if resume_path:
            command.extend(["--checkpoint", str(resume_path)])
        print("Starting CUDA training:", " ".join(command), flush=True)
        output_lines = []
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                output_lines.append(line)
            return_code = process.wait()
        if return_code == 0:
            return batch_size

        combined_output = "".join(output_lines).lower()
        is_oom = "out of memory" in combined_output or "cuda error: memory" in combined_output
        if not is_oom or batch_size <= 1:
            raise RuntimeError(f"Training failed with exit code {return_code}; see log above.")
        batch_size = max(1, batch_size // 2)
        latest = ROOT / "checkpoints" / "latest.pt"
        resume_path = latest if latest.exists() and latest.stat().st_mtime >= started_at else None
        print(f"CUDA OOM detected; retrying with batch_size={batch_size}; dataset unchanged.", flush=True)
        torch.cuda.empty_cache()


def run_final_evaluation():
    command = [
        sys.executable, "evaluate.py", "--checkpoint", "checkpoints/best.pt",
        "--batch-size", "1", "--out", "kaggle_final",
    ]
    print("Evaluating every DUO test image:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    source_json = ROOT / "reports" / "eval_kaggle_final.json"
    source_txt = ROOT / "reports" / "eval_kaggle_final.txt"
    report = json.loads(source_json.read_text(encoding="utf-8"))
    if report["dataset"]["images_evaluated"] != 1111:
        raise RuntimeError(f"Evaluation did not cover all 1111 test images: {report['dataset']}")
    final_json = ROOT / "reports" / "final_metrics.json"
    final_txt = ROOT / "reports" / "final_metrics.txt"
    shutil.copyfile(source_json, final_json)
    shutil.copyfile(source_txt, final_txt)
    print("Full-test metrics:", json.dumps({
        key: report[key] for key in
        ("precision", "recall", "f1", "mAP@0.50", "mAP@0.50:0.95")
    }, sort_keys=True))
    print("Metrics saved:", final_json, final_txt)
    return report


def run_real_visual_tests(cfg):
    import cv2

    from inference import detect_image, draw_detections, load_checkpoint_model, load_image

    test_dir = ROOT / cfg["dataset"]["image_dir_val"]
    named = [test_dir / "994.jpg", test_dir / "995.jpg"]
    missing = [str(path) for path in named if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required visual test images missing: {missing}")
    others = [path for path in sorted(test_dir.iterdir())
              if path.is_file() and path.name not in {"994.jpg", "995.jpg"}]
    selected = named + others[:8]
    output_dir = ROOT / "outputs" / "visualizations" / "final_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_checkpoint_model(str(ROOT / "checkpoints" / "best.pt"), "cuda")
    threshold = float(cfg["inference"]["confidence_threshold"])
    summaries = []
    for path in selected:
        image = load_image(str(path))
        candidates = detect_image(bundle, image, conf_thr=0.0)
        scores = candidates["scores"]
        keep = scores >= threshold
        boxes = candidates["boxes"][keep]
        kept_scores = scores[keep]
        labels = candidates["labels"][keep]
        annotated = draw_detections(image, boxes, kept_scores, labels, bundle["classes"])
        output_path = output_dir / f"{path.stem}_predictions.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not write visualization: {output_path}")
        summary = {
            "image": path.name,
            "num_detections": len(boxes),
            "max_foreground_confidence": float(scores.max()) if len(scores) else 0.0,
            "predicted_classes": sorted({bundle["classes"][int(label)] for label in labels}),
            "detections": [
                {"class": bundle["classes"][int(label)], "confidence": float(score),
                 "box_xyxy": [round(float(value), 1) for value in box]}
                for box, score, label in zip(boxes, kept_scores, labels)
            ],
            "visualization": str(output_path.relative_to(ROOT)),
        }
        summaries.append(summary)
        print("REAL_TEST_INFERENCE:", json.dumps(summary, sort_keys=True))
    if len(summaries) < 10:
        raise RuntimeError(f"Expected at least 10 visual tests, got {len(summaries)}")
    index_path = output_dir / "inference_results.json"
    index_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("Prediction visualizations:", output_dir)
    print("Per-image prediction report:", index_path)
    detected = [item for item in summaries if item["num_detections"]]
    print("VISUAL_TEST_SUMMARY:", json.dumps({
        "images_tested": len(summaries),
        "images_with_detections": len(detected),
        "example_predicted_classes": sorted({
            name for item in summaries for name in item["predicted_classes"]
        }),
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Full CUDA DUO training and verification")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="initial CUDA batch size; OOM retries lower it without dropping data")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.epochs != 50:
        raise ValueError("This final training package requires the full 50-epoch target.")
    if args.batch_size < 1 or args.num_workers < 2:
        raise ValueError("Use a positive batch size and multiple workers (at least 2).")

    started_at = time.time()
    torch, cfg = verify_runtime()
    counts = verify_dataset(cfg)
    checkpoint_dir = ROOT / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        raise RuntimeError(
            "Existing checkpoint files found. Start from a fresh extracted package "
            "so no old checkpoint can be mistaken for this CUDA run.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for directory in (ROOT / "reports", ROOT / "outputs" / "metrics",
                      ROOT / "outputs" / "visualizations"):
        directory.mkdir(parents=True, exist_ok=True)

    selected_batch_size = train_with_oom_recovery(torch, args, started_at)
    log_path = ROOT / "outputs" / "metrics" / "train_log_kaggle.jsonl"
    history = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not history or history[-1]["epoch"] != 50:
        raise RuntimeError(f"Training did not complete 50 epochs: last record={history[-1:]}")
    final_record = history[-1]
    loss_keys = (
        "epoch", "train_total", "train_roi_cls", "train_roi_box",
        "train_rpn_objectness", "train_rpn_box", "val_total",
        "val_roi_cls", "val_roi_box", "val_rpn_objectness", "val_rpn_box",
    )
    print("FINAL_EPOCH_LOSSES:", json.dumps({
        key: final_record[key] for key in loss_keys if key in final_record
    }, sort_keys=True))

    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_mtime < started_at:
        raise RuntimeError("best.pt was not newly created by this CUDA training run.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("training_metadata", {})
    expected_hash = hashlib.sha256((ROOT / "models" / "hybrid_detector.py").read_bytes()).hexdigest()
    if checkpoint.get("epoch", 0) < 1 or metadata.get("device") != "cuda" \
            or metadata.get("num_training_samples") != counts["training_images"] \
            or metadata.get("source_hybrid_detector_sha256") != expected_hash:
        raise RuntimeError(f"Checkpoint provenance validation failed: {metadata}")
    print("NEW CUDA CHECKPOINT:")
    print("  path:", checkpoint_path)
    print("  size_bytes:", checkpoint_path.stat().st_size)
    print("  epoch:", checkpoint["epoch"])
    print("  metrics:", json.dumps(checkpoint.get("metrics", {}), sort_keys=True))
    print("  training_metadata:", json.dumps(metadata, sort_keys=True))
    print("  selected_batch_size:", selected_batch_size)
    print("  number_of_training_samples:", counts["training_images"])

    run_final_evaluation()
    run_real_visual_tests(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CUDA training/verification stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise