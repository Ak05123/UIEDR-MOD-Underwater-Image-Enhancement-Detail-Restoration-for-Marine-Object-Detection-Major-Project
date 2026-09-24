"""Evaluate HybridYOLOFasterRCNN on the validation/test split.

Usage:
    .\\.venv\\Scripts\\python.exe evaluate.py --checkpoint checkpoints/smoke_test.pt
        [--max-images 50] [--conf 0.5] [--nms 0.4] [--batch-size 4]

Saves a real-metrics JSON + TXT report to reports/ and detection plots/metrics
to outputs/metrics/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.loader import CocoDetDataset, collate_det
from inference import load_checkpoint_model
from training.metrics import evaluate_predictions
from training.trainer import load_config


@torch.no_grad()
def run_evaluation(cfg, checkpoint, max_images=None, conf_thr=0.5, nms_thr=0.4,
                   batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_checkpoint_model(checkpoint, str(device))
    model = bundle["model"]
    classes = bundle["classes"]

    d = cfg["dataset"]
    ds = CocoDetDataset(d["annotation_val"], d["image_dir_val"], d["image_size"],
                        augment={"enabled": False},
                        enhance=cfg["model"].get("enhancement", {}), classes=classes)
    if max_images:
        ds.images = ds.images[:max_images]
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_det)

    old_thr = (model.cls_thr, model.nms_thr)
    model.cls_thr, model.nms_thr = conf_thr, nms_thr

    detections, ground_truths = [], []
    t0 = time.time()
    for batch in dl:
        for i in range(batch["images"].shape[0]):    # model inference is per-image
            det = model(batch["images"][i: i + 1].to(device))[0]
            detections.append({k: v.numpy() for k, v in det.items()})
            ground_truths.append({"boxes": batch["boxes"][i].numpy(),
                                  "labels": batch["labels"][i].numpy()})
    model.cls_thr, model.nms_thr = old_thr
    elapsed = time.time() - t0

    metrics = evaluate_predictions(detections, ground_truths, classes,
                                   iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7,
                                                   0.75, 0.8, 0.85, 0.9, 0.95])
    return bundle, metrics, detections, ground_truths, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", default=None, help="output report name (json/txt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bundle, metrics, dets, gts, elapsed = run_evaluation(
        cfg, args.checkpoint, args.max_images, args.conf, args.nms, args.batch_size)

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_is_smoke_test": bundle["smoke_test"],
        "checkpoint_epoch": bundle["epoch"],
        "dataset": {
            "val_annotations": cfg["dataset"]["annotation_val"],
            "val_images": cfg["dataset"]["image_dir_val"],
            "images_evaluated": len(dets),
            "total_gt_boxes": metrics["total_gt"],
        },
        "classes": bundle["classes"],
        "thresholds": {"confidence": args.conf, "nms": args.nms,
                       "iou_range": "0.50:0.95 step 0.05"},
        "eval_time_s": round(elapsed, 1),
        "device": str(bundle["device"]),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mAP@0.50": metrics["mAP@0.50"],
        "mAP@0.50:0.95": metrics["mAP@0.50:0.95"],
        "per_class": metrics["per_class"],
    }

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    name = args.out or Path(args.checkpoint).stem
    (out_dir / f"eval_{name}.json").write_text(json.dumps(report, indent=2))
    lines = ["=" * 60, "EVALUATION REPORT", "=" * 60,
             f"checkpoint:  {args.checkpoint}" +
             ("   [SMOKE-TEST CHECKPOINT - not a trained model]" if bundle["smoke_test"] else ""),
             f"epoch:       {bundle['epoch']}",
             f"images:      {len(dets)}  GT boxes: {metrics['total_gt']}",
             f"conf={args.conf}  nms={args.nms}",
             f"Precision:   {metrics['precision']:.4f}",
             f"Recall:      {metrics['recall']:.4f}",
             f"F1:          {metrics['f1']:.4f}",
             f"mAP@0.50:    {metrics['mAP@0.50']:.4f}",
             f"mAP@0.5:0.95:{metrics['mAP@0.50:0.95']:.4f}",
             "per-class AP@0.50:"]
    for c, v in metrics["per_class"].items():
        lines.append(f"  {c:<14} {v['ap@0.50']:.4f}")
    (out_dir / f"eval_{name}.txt").write_text("\n".join(lines))

    print(f"\nEvaluated {len(dets)} images in {elapsed:.1f}s")
    print(f"Precision {metrics['precision']:.4f} | Recall {metrics['recall']:.4f} | "
          f"F1 {metrics['f1']:.4f}")
    print(f"mAP@0.50 {metrics['mAP@0.50']:.4f} | mAP@0.50:0.95 {metrics['mAP@0.50:0.95']:.4f}")
    print(f"Reports: reports/eval_{name}.json, reports/eval_{name}.txt")


if __name__ == "__main__":
    main()
