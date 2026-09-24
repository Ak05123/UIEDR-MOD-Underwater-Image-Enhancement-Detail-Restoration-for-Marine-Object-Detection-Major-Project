"""Entry point for training the Hybrid YOLOv11 + Faster R-CNN detector.

Usage:
    .\\.venv\\Scripts\\python.exe train.py [--epochs 30] [--batch-size 4]
        [--resume checkpoints/latest.pt] [--max-epochs N] [--validate-data]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Train HybridYOLOFasterRCNN")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None,
                    help="training device override")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="DataLoader worker count override")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path to resume from")
    ap.add_argument("--resume", default=None, help="deprecated alias for --checkpoint")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap on epochs this run")
    ap.add_argument("--validate-data", action="store_true", help="run dataset validation first")
    ap.add_argument("--run-name", default="default")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    from training.trainer import Trainer, load_config, validate_dataset

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    cfg = load_config(str(config_path))
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.device:
        cfg["training"]["device"] = args.device
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers

    if args.validate_data:
        reports, path = validate_dataset(cfg)
        print(f"Dataset report written to {path}")
        for name, r in reports.items():
            print(f"  {name}: {r['num_images_declared']} images, "
                  f"{r['num_annotations']} anns, missing={len(r['missing_images'])}, "
                  f"invalid_boxes={r['invalid_boxes']}")

    trainer = Trainer(cfg, run_name=args.run_name)
    checkpoint = args.checkpoint or args.resume
    trainer.fit(resume_path=checkpoint, max_epochs=args.max_epochs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit(130)
