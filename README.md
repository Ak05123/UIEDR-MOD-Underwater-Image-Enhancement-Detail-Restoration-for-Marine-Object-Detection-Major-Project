# UIEDR MOD — Underwater Image Enhancement Detail Restoration for Marine Object Detection

Major Project · Computer Vision · Underwater Object Detection.

A PyTorch object-detection system for the four-class DUO underwater dataset. The implemented pipeline combines classical underwater image enhancement, a residual CNN, a YOLOv11-style backbone, an FPN/PAN neck, an RPN, ROI Align, coordinate attention, and classification plus bounding-box regression.

## Current implementation

- **Enhancement:** gray-world white balance, LAB CLAHE, and multi-scale Retinex (MSR).
- **Detector:** residual CNN -> YOLOv11-style C3k2/C2PSA/SPPF backbone -> FPN/PAN P3/P4/P5 -> RPN -> multi-level ROI Align -> coordinate attention -> detection heads.
- **Dataset:** RAW-890-format DUO data at `dataset/DUO` with `holothurian`, `echinus`, `scallop`, and `starfish` classes.
- **Training:** configurable AdamW or SGD training, schedulers, checkpointing, resume support, early stopping, and JSONL logs.
- **Evaluation:** precision, recall, F1, mAP@0.50, mAP@0.50:0.95, and per-class AP.
- **Inference:** single-image CLI and reusable Python utilities with annotated output rendering.
- **Frontend:** Streamlit dashboard with dataset inspection, enhancement previews, single and batch inference, evaluation, training controls, architecture inspection, and project information.

The available `checkpoints/smoke_test.pt` checkpoint is an integration smoke-test checkpoint. It is not a trained final model; metrics and detections should be interpreted accordingly.

## Latest verification status

On the available CPU-only environment, a bounded real-data run completed 2 epochs using 8 DUO training images and 4 DUO test/validation images, resuming from `checkpoints/sanity_real.pt`. The run produced `checkpoints/best.pt`, `checkpoints/latest.pt`, epoch checkpoints, and `outputs/metrics/train_log_cpu_practical.jsonl`.

The best checkpoint was evaluated on 10 real DUO test images. At confidence 0.50, precision, recall, F1, mAP@0.50, and mAP@0.50:0.95 were all 0.0000. A separate 0.05-confidence visualization pass produced 10 valid boxes, all predicted as `holothurian`, with mean best IoU per ground-truth box of 0.0. These results indicate that the bounded run is a training/integration verification artifact, not a production-quality trained model. Full-dataset training remains to be completed on a suitable GPU or longer-running CPU environment.

## Environment setup

Use Python 3.10+ and a virtual environment. Install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

For GPU training or inference, install the PyTorch and TorchVision builds appropriate for the target CUDA version from the official PyTorch instructions.

## Run the Streamlit application

From the project root:

```powershell
streamlit run app.py
```

The app starts on the local Streamlit URL. Model loading, inference, evaluation, and training are user-triggered; opening the app does not start a long-running operation.

## Command-line usage

Show command options:

```powershell
python train.py --help
python evaluate.py --help
python infer.py --help
```

Cloud training supports explicit device, worker, and resume controls:

```powershell
python train.py --epochs 50 --batch-size 4 --device cuda --num-workers 4
python train.py --checkpoint checkpoints/latest.pt --device cuda
```

The existing `--resume` option remains available as an alias for `--checkpoint`. Each epoch writes an epoch checkpoint, `latest.pt`, and updates `best.pt` when validation loss improves. Evaluation and inference can be run independently afterward with `evaluate.py` and `infer.py` using `checkpoints/best.pt`.

Train with the configured dataset:

```powershell
python train.py --epochs 30 --batch-size 4
```

Validate the dataset before training:

```powershell
python train.py --validate-data --max-epochs 1
```

Evaluate a checkpoint on a limited number of images:

```powershell
python evaluate.py --checkpoint checkpoints/smoke_test.pt --max-images 10
```

Run inference on one image:

```powershell
python infer.py --image path\to\image.jpg --checkpoint checkpoints\best.pt
```

## Project paths

- `dataset/DUO/annotations/`: RAW-890 train and official test annotations.
- `dataset/DUO/images/train/` and `dataset/DUO/images/test/`: dataset images.
- `checkpoints/`: model checkpoints.
- `outputs/predictions/`: annotated inference results.
- `reports/`: dataset validation and evaluation reports.
- `config.yaml`: dataset, model, training, and inference configuration.

The configured official DUO test split is used as the validation/evaluation split. No separate validation directory is created by the current configuration.

## Limitations

This is a compact research implementation intended to run reliably on CPU as well as CUDA devices. Training and full evaluation can be expensive on CPU. Results depend on checkpoint quality, thresholds, and the selected device. The smoke-test checkpoint verifies integration and is not a production model.
