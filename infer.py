"""Single-image inference CLI.

Usage:
    .\\.venv\\Scripts\\python.exe infer.py --image path/to/img.jpg
        [--checkpoint checkpoints/best.pt] [--conf 0.5] [--out out.jpg]
"""
import argparse
import json
from pathlib import Path

from inference import load_checkpoint_model, detect_image, draw_detections, load_image
from training.trainer import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    bundle = load_checkpoint_model(args.checkpoint)
    img = load_image(args.image)
    det = detect_image(bundle, img, conf_thr=args.conf, nms_thr=args.nms)
    annotated = draw_detections(img, det["boxes"], det["scores"], det["labels"],
                                bundle["classes"])

    out = Path(args.out) if args.out else Path("outputs/predictions") / \
        (Path(args.image).stem + "_det.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(out), annotated)

    summary = {
        "image": str(args.image), "checkpoint": args.checkpoint,
        "confidence_threshold": args.conf, "nms_threshold": args.nms,
        "num_detections": len(det["boxes"]),
        "detections": [
            {"class": bundle["classes"][int(l)], "confidence": float(s),
             "box_xyxy": [round(float(v), 1) for v in b]}
            for b, s, l in zip(det["boxes"], det["scores"], det["labels"])],
    }
    print(json.dumps(summary, indent=2))
    print(f"Annotated image: {out}")


if __name__ == "__main__":
    main()
