# UIEDR MOD — Underwater Image enhancementDetail Restoration for Marine Object Detection

# UIEDR MOD

## Underwater Image Enhancement Detail Restoration for Marine Object Detection

**Major Project**

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
- **enhancement:** gray-world white balance, LAB CLAHE, and multi-scale Retinex (MSR).
- **Detector:** residual CNN -> YOLOv11-style C3k2/C2PSA/SPPF backbone -> FPN/PAN P3/P4/P5 -> RPN -> multi-level ROI Align -> coordinate attention -> detection heads.
- **Dataset:** RAW-890-format DUO data at `dataset/DUO` with `holothurian`, `echinus`, `scallop`, and `starfish` classes.
- **Training:** configurable AdamW or SGD training, schedulers, checkpointing, resume support, early stopping, and JSONL logs.
- **Evaluation:** precision, recall, F1, mAP@0.50, mAP@0.50:0.95, and per-class AP.
- **Inference:** single-image CLI and reusable Python utilities with annotated output rendering.
- **Frontend:** Streamlit dashboard with dataset inspection, upload enhancement, live WebRTC camera inference, single and batch inference, evaluation, training controls, architecture inspection, and project information.
=======
UIEDR MOD is an underwater computer vision system that combines **underwater image enhancement, detail restoration, multi-scale feature extraction, region proposal generation, attention mechanisms, and object detection** to identify marine objects in challenging underwater images.
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
UIEDR MOD is an underwater computer vision system that combines **underwater image enhancement, detail restoration, multi-scale feature extraction, region proposal generation, attention mechanisms, and object detection** to identify marine objects in challenging underwater images.
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
UIEDR MOD is an underwater computer vision system that combines **underwater image enhancement, detail restoration, multi-scale feature extraction, region proposal generation, attention mechanisms, and object detection** to identify marine objects in challenging underwater images.
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86

The project uses the **DUO (Detection of Underwater Objects)** dataset and provides a complete **Streamlit-based interface** for dataset inspection, enhancement visualization, object detection, batch inference, evaluation, training, and architecture exploration.

---

## 🚀 Project Overview

Underwater images often suffer from:

* Poor visibility
* Color distortion
* Low contrast
* Uneven illumination
* Blur and loss of fine details
* Reduced object detection accuracy

UIEDR MOD addresses these challenges through a multi-stage pipeline:

```text
Input Image
     ↓
White Balance
     ↓
CLAHE
     ↓
Multi-Scale Retinex (MSR)
     ↓
Residual CNN Detail Restoration
     ↓
YOLO-style Backbone
     ↓
FPN / PAN Feature Fusion
     ↓
Region Proposal Network (RPN)
     ↓
ROI Align
     ↓
Coordinate Attention
     ↓
Faster R-CNN-style Detection Head
     ↓
Final Marine Object Detection
```

---

## ✨ Key Features

* Underwater image enhancement
* White Balance
* CLAHE contrast enhancement
* Multi-Scale Retinex (MSR)
* Residual CNN detail restoration
* YOLO-style backbone
* FPN/PAN multi-scale feature fusion
* Region Proposal Network
* ROI Align
* Coordinate Attention
* Faster R-CNN-style detection head
* DUO COCO-format dataset support
* Four marine-object detection classes
* Single-image inference
* Batch inference
* Evaluation pipeline
* Training pipeline
* Model checkpointing and resume support
* Streamlit web interface
* Enhancement-stage visualization
* Detection visualization with bounding boxes
* Export of annotated prediction images

---

## 🧠 Model Architecture

The system combines image enhancement and object detection into a unified workflow.

### 1. Image Enhancement

The input underwater image passes through multiple enhancement stages:

```text
Original
   ↓
White Balance
   ↓
CLAHE
   ↓
MSR
   ↓
Residual CNN
   ↓
Final Enhanced Image
```

### 2. Feature Extraction

The enhanced image is processed using a YOLO-style backbone to extract hierarchical features at multiple scales.

### 3. Feature Fusion

FPN/PAN combines multi-scale features to improve detection of objects with different sizes.

### 4. Region Proposal

The Region Proposal Network (RPN) generates candidate object regions.

### 5. ROI Processing

ROI Align extracts region-specific features from the proposed regions.

### 6. Attention

Coordinate Attention is applied to improve spatial and channel-aware feature representation.

### 7. Detection

A Faster R-CNN-style detection head performs:

* Object classification
* Bounding-box regression

---

## 📊 Dataset

The project uses the **DUO (Detection of Underwater Objects)** dataset.

### Dataset Statistics

| Split | Images | Annotations |
| ----- | -----: | ----------: |
| Train |  6,671 |      63,998 |
| Test  |  1,111 |      10,517 |

### Detection Classes

| ID | Class       |
| -: | ----------- |
|  0 | Holothurian |
|  1 | Echinus     |
|  2 | Scallop     |
|  3 | Starfish    |

### Dataset Structure

```text
dataset/
└── DUO/
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_test.json
    │
    └── images/
        ├── train/
        └── test/
```

The DUO dataset is stored in **COCO annotation format**.

> **Note:** `raw-890` is not used as the supervised training dataset because it does not provide the required COCO annotations, class labels, and train/test structure.

---

## 🛠️ Tech Stack

### Programming

* Python

### Deep Learning

* PyTorch
* TorchVision

### Computer Vision

* OpenCV
* NumPy
* Pillow

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib

### Configuration

* PyYAML

### Web Application

* Streamlit

### GPU Training

* NVIDIA CUDA environment when available

---

## 📁 Project Structure

```text
UIEDR-MOD/
│
├── app.py
├── train.py
├── evaluate.py
├── infer.py
├── inference.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── models/
│   ├── enhancement.py
│   ├── residual_cnn.py
│   ├── yolo_backbone.py
│   ├── fpn_pan.py
│   ├── coordinate_attention.py
│   ├── rpn.py
│   ├── roi_head.py
│   ├── hybrid_detector.py
│   └── losses.py
│
├── dataset/
│   ├── loader.py
│   ├── validation.py
│   └── DUO/
│
├── training/
│   ├── trainer.py
│   └── metrics.py
│
├── utils/
│   ├── seed.py
│   └── device.py
│
├── checkpoints/
│
├── outputs/
│   ├── predictions/
│   ├── metrics/
│   └── visualizations/
│
└── reports/
```

---

## 💻 Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## 🌊 Streamlit Application

Run the web application:

```powershell
streamlit run app.py
```

The Streamlit interface contains:
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
### Live camera and phone testing

On the development computer, run the app and open `http://localhost:8501`. In **Enhancement → Live Camera**, select **Start Camera** and allow browser camera access. The desktop preview stays live while a separate enhanced/detected result and callback-measured FPS, processing time, device, and detection count update as frames finish. Slow CPU processing drops stale frames instead of accumulating a queue.

To test from a phone:

1. Start Streamlit on the development computer with `streamlit run app.py --server.port 8501`.
2. Expose port 8501 through an HTTPS tunnel, for example `ngrok http 8501` after configuring ngrok, or deploy to an HTTPS Streamlit host.
3. Open the tunnel's generated `https://...` URL on the phone. Do not use `http://<PC-LAN-IP>:8501`; mobile browsers block camera access on that insecure origin.
4. Open **Enhancement → Live Camera**, allow camera permission, and press **Start Camera**. The rear-facing camera is preferred when supported; use **Switch camera** to choose another device.

The public STUN server is configured under `webrtc.ice_servers` in `config.yaml`. Some mobile carriers, corporate networks, and hosted deployments also require TURN. Set `UIEDR_TURN_URL`, `UIEDR_TURN_USERNAME`, and `UIEDR_TURN_CREDENTIAL` as server-side environment secrets; never commit TURN credentials. HTTPS is required for remote browser camera access, and an HTTPS page alone does not guarantee WebRTC connectivity if the network blocks STUN/TURN traffic.

## Command-line usage
=======
=======

>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
1. **Dashboard**
2. **Dataset**
3. **Enhancement**
4. **Detection**
5. **Batch Inference**
6. **Evaluation**
7. **Training**
8. **Architecture**
9. **About**
<<<<<<< HEAD
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86

---

=======

1. **Dashboard**
2. **Dataset**
3. **Enhancement**
4. **Detection**
5. **Batch Inference**
6. **Evaluation**
7. **Training**
8. **Architecture**
9. **About**

---

>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
## 🖼️ Enhancement Pipeline

The Enhancement section visualizes the individual processing stages:

```text
Original
   ↓
White Balance
   ↓
CLAHE
   ↓
MSR
   ↓
Residual Detail Restoration
   ↓
Final Enhanced
```

Each stage is generated through its corresponding processing component rather than using repeated copies of the original image.

---

## 🎯 Object Detection

The Detection module accepts an image and performs inference using the trained UIEDR MOD model.

Detected objects are displayed using their class names:

```text
0 → holothurian
1 → echinus
2 → scallop
3 → starfish
```

Bounding boxes and confidence scores are visualized on the output image.

Annotated prediction images are saved under:

```text
outputs/predictions/
```

---

## 🔄 Batch Inference

Batch inference can process multiple underwater images and save the resulting predictions.

Outputs are stored in:

```text
outputs/predictions/
```

---

## 🏋️ Training

Full model training requires a **CUDA-capable NVIDIA GPU**.

The current development machine uses:

```text
Intel Iris Xe Graphics
```

and does not contain an NVIDIA CUDA-capable GPU.

Therefore, full DUO training should be performed in a suitable cloud GPU or other CUDA-enabled environment.

### Training Command

```bash
python train.py --epochs 50 --batch-size 4 --device cuda --num-workers 4
```

### Resume Training

```bash
python train.py --checkpoint checkpoints/latest.pt --device cuda
```

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
## Full DUO CUDA training on Kaggle

Use `UIEDR_MOD_CUDA_FINAL_TRAINING.zip` as the source package. In Kaggle, create a notebook with a GPU accelerator, attach the package as input, extract it under `/kaggle/working`, and run `python kaggle_train.py` from the extracted package root. The runner requires CUDA, validates all 6,671 train and 1,111 official test images and their annotations, targets 50 epochs, uses multiple workers, and starts with batch size 2. If CUDA runs out of memory, it lowers only the batch size and resumes from the latest completed epoch.

The runner evaluates all 1,111 test images and writes `reports/final_metrics.json`, `reports/final_metrics.txt`, and real model-prediction images under `outputs/visualizations/final_test/`. It will not use or overwrite a pre-existing `.pt` checkpoint. After the Kaggle run finishes, save a notebook version and download `checkpoints/best.pt` from the notebook's Output files; place that file at the local project's `checkpoints/best.pt` before running Streamlit. These artifacts and the Streamlit detection page must be checked after the GPU run; a completed CPU subset diagnostic is not final-model validation.

The existing `--resume` option remains available as an alias for `--checkpoint`. Each epoch writes an epoch checkpoint, `latest.pt`, and updates `best.pt` when validation loss improves. Evaluation and inference can be run independently afterward with `evaluate.py` and `infer.py` using `checkpoints/best.pt`.
=======
Training checkpoints include:
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
Training checkpoints include:
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
Training checkpoints include:
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86

```text
checkpoints/
├── best.pt
└── latest.pt
```

---

## 📈 Evaluation

After obtaining a genuinely trained checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/best.pt
```

The evaluation pipeline is designed to calculate:

* Precision
* Recall
* F1-score
* mAP@0.50
* mAP@0.50:0.95
* Per-class performance

Evaluation outputs are stored under:

```text
outputs/metrics/
reports/
```

---

## 🔍 Inference

Run inference on a DUO test image:

```bash
python infer.py --image dataset/DUO/images/test/994.jpg --checkpoint checkpoints/best.pt
```

Example output:

```text
Image: 994.jpg

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
The configured official DUO test split is used for per-epoch validation, checkpoint selection, and final evaluation. Therefore, final metrics are not an untouched independent holdout result. No separate validation directory is created by the current configuration.
=======
=======
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
Detected Objects:
- Class
- Confidence
- Bounding Box
```
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86
=======
>>>>>>> bdea433fb220d7823339e808e64d98e14742cc86

---

## 🔬 Dataset Validation

The DUO dataset has been validated for:

* Missing image files
* Invalid bounding boxes
* Unknown class IDs
* Annotation/image consistency

Current validation:

```text
Training Images: 6,671
Test Images:     1,111
Missing Images:  0
Invalid Boxes:   0
```

---

## 📌 Current Project Status

### Implemented

* [x] DUO dataset integration
* [x] Dataset validation
* [x] Underwater enhancement pipeline
* [x] Residual CNN detail restoration
* [x] YOLO-style backbone
* [x] FPN/PAN feature fusion
* [x] RPN
* [x] ROI Align
* [x] Coordinate Attention
* [x] Detection head
* [x] Training pipeline
* [x] Evaluation pipeline
* [x] Inference pipeline
* [x] Streamlit application
* [x] Batch inference
* [x] Checkpoint saving and resume support
* [x] Cloud-training CLI support

### 🚧 Remaining Work

* [ ] Full DUO training on a CUDA-capable GPU
* [ ] Final trained `best.pt`
* [ ] Complete 1,111-image test evaluation
* [ ] Final Precision / Recall / F1 results
* [ ] Final mAP@0.50 results
* [ ] Final mAP@0.50:0.95 results
* [ ] Final per-class performance analysis

---

## ⚠️ Important Limitation

The currently available checkpoint was produced during an early bounded training run and is **not a final fully trained DUO detector**.

Therefore, final detection performance and evaluation metrics should only be reported after training on the complete DUO training dataset.

No placeholder or fabricated detection metrics are included in this repository.

---

## 🔮 Future Work

* Full-scale DUO training using CUDA
* Hyperparameter optimization
* Improved enhancement quality
* Advanced attention mechanisms
* Model optimization for faster inference
* GPU-based deployment
* Additional underwater datasets
* Real-time underwater object detection
* Further evaluation across different underwater environments

---

## 📚 Project Goal

The primary goal of UIEDR MOD is to investigate how **underwater image enhancement and detail restoration can be integrated with modern object-detection techniques** to improve the visibility and detection of marine objects in challenging underwater environments.

---

## 👨‍💻 Author

**Akshat Gupta**

Computer Science Engineering — Artificial Intelligence

---

## 📄 License

This project is developed as an academic major project.

Dataset licensing and usage should follow the original DUO dataset terms and conditions.
