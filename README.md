# CVF — Computer Vision Flow

**A config-driven, modular, and extensible Computer Vision execution framework.**

CVF transforms the traditional approach of writing model-specific inference code into a **purely configuration-driven workflow**. Users provide only a YAML config file describing their model, task, backend, device, and pipeline stages — CVF Core handles the rest.

```
config.yaml + user intent
        ↓
CVF Core (Registry + Pipeline Engine + Compatibility Validator)
        ↓
Preprocess → Inference → Adapter → Postprocess → Benchmark/Evaluation
```

---

## Quick Start

### Installation

```bash
# From project root
cd cvf
pip install -e .
# or
pip install pydantic pyyaml onnxruntime opencv-python numpy scipy
```

### Run Inference

```bash
# All inputs are declared in YAML (config-driven)

# YOLOv5 Detection
python -m cvf.cli.main run cvf/configs/inference.yaml

# ResNet50 Classification
python -m cvf.cli.main run cvf/configs/classification.yaml

# LeNet5 Classification (MNIST-style)
python -m cvf.cli.main run cvf/configs/lenet5.yaml

# CRNN HTR (Handwritten Text Recognition)
python -m cvf.cli.main run cvf/configs/crnn.yaml

# Optional: override input from config via CLI
python -m cvf.cli.main run cvf/configs/inference.yaml /path/to/image.jpg

# Via wrapper script (conda env auto-setup)
./run.sh cvf/configs/inference.yaml
./run.sh cvf/configs/inference.yaml /path/to/image.jpg
```

### Run Benchmark

```bash
python -m cvf.cli.main benchmark cvf/configs/benchmark.yaml
```

---

## Config-Driven Design

Instead of writing Python code for each model/task combination, users declare **what they want** in YAML:

```yaml
# cvf/configs/inference.yaml
model:
  family: yolo
  variant: yolov5
  path: /path/to/yolov5.onnx
  target_size: [640, 640]
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]

task:
  type: detection
  conf_threshold: 0.3
  iou_threshold: 0.5
  dataset: coco

backend:
  type: onnxruntime

device:
  type: cpu

pipeline:
  stages:
    - preprocess
    - inference
    - adapter
    - postprocess
    - benchmark

input: /path/to/image.jpg
output_dir: outputs
```

CVF Core automatically:
1. **Validates** config against schema (Pydantic)
2. **Checks compatibility** (model format ↔ backend, backend ↔ device, family ↔ task)
3. **Resolves components** from registries (model, preprocessor, adapter, backend, device)
4. **Executes** the dynamic pipeline stages

---

## Architecture

```
                         config.yaml
                              │
                              ▼
                     ┌─────────────────┐
                     │ Config Validator│
                     └────────┬────────┘
                              ▼
                     Compatibility Check
                              │
                              ▼
                     ┌─────────────────┐
                     │     Registry    │
                     └────────┬────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       MODEL               TASK               BACKEND
          │                   │                   │
     Family/Variant      Detection/etc.       ONNX Runtime
          │                   │                   │
          ▼                   │                   ▼
     Preprocess                │             Device
          │                   │              CPU/CUDA
          ▼                   │
     Inference ────────────────┘
          │
          ▼
   Output Adapter
          │
          ▼
 Canonical Output
          │
          ▼
 Task Postprocess
          │
          ├───────────────┐
          ▼               ▼
     Evaluation       Benchmark
          │               │
          └───────┬───────┘
                  ▼
              Artifacts
```

### Key Principles

| Principle | Description |
|-----------|-------------|
| **Config-Driven** | No Python code needed for normal execution |
| **Dynamic Pipeline** | User selects stages: `preprocess`, `inference`, `adapter`, `postprocess`, `benchmark` |
| **Separate Dimensions** | Model, Task, Backend, Device are independent — composed at runtime |
| **Model Adapters** | Raw model output → Canonical format (model-specific) |
| **Task Postprocess** | Operates on canonical output only (model-agnostic) |
| **Registry Pattern** | Zero `if/elif` chains — dynamic component resolution |
| **Compatibility Validation** | Fail fast before inference starts |

---

## Supported Models

| Family | Variant | Task | Input | Output | Notes |
|--------|---------|------|-------|--------|-------|
| YOLO | yolov5 | Detection | 640×640 RGB | `DetectionOutput(boxes, scores, class_ids)` | Supports FP32/FP16/Mixed FP32+INT8; video via frame-by-frame pipeline |
| Classification | resnet50 | Classification (EuroSAT) | 224×224 RGB | `ClassificationOutput(logits, probs, top-k, top_class_names, top1_name)` | EuroSAT normalization `mean [0.3445,0.3803,0.4077] std [0.0915,0.0652,0.0553]` (see `AI_Script/preprocess/classification/resnet.py`); dataset `eurosat` order `Forest,River,Highway,AnnualCrop,SeaLake,HerbaceousVegetation,Industrial,Residential,PermanentCrop,Pasture`; FP16 auto-cast in `cvf/backends/onnxruntime.py:51` |
| Classification | lenet5 | Classification | 32×32 Gray | `ClassificationOutput(logits, probs, top-k, top_class_names)` | MNIST-style 47-class variant; FP16 logits cast to FP32 |
| CRNN | crnn | HTR | 100×32 Gray | `HTROutput(text, confidence)` | |

---

## API Usage (Programmatic)

```python
from cvf.core import load_config, Pipeline
from cvf.plugins.builtin import register_builtin

# Register built-in components
register_builtin()

# Load config
config = load_config("cvf/configs/inference.yaml")

# Create pipeline
pipeline = Pipeline(config)

# Run with image path
context = pipeline("/path/to/image.jpg")

# Run with preprocessed tensor (numpy array)
import numpy as np
tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)
context = pipeline(tensor)

# Access results
print(context.task_result)      # DetectionOutput/ClassificationOutput/HTROutput
# ClassificationOutput now includes human-readable labels when dataset/class_names set:
#   ClassificationOutput(... class_names=[10 EuroSAT], top_class_names=[..], top1_name="Highway")
print(context.task_result.to_dict()["top_class_names"])  # e.g. ["Highway", "Industrial", ...]
print(context.benchmark_metrics) # Latency, throughput
print(context.stage_timings)    # Per-stage timing
```

---

## Config Reference

| Section | Required | Description |
|---------|----------|-------------|
| `model.family` | ✅ | `yolo`, `resnet`, `crnn` |
| `model.variant` | ✅ | `yolov5`, `resnet50`, `lenet5`, `crnn` |
| `model.path` | ✅ | Path to `.onnx` model |
| `model.target_size` | | `[H, W]` for preprocessing |
| `model.mean` / `model.std` | | Normalization values – **ResNet50 EuroSAT**: `[0.3445,0.3803,0.4077]` / `[0.0915,0.0652,0.0553]` (not ImageNet) |
| `task.type` | ✅ | `detection`, `classification`, `htr` |
| `task.dataset` | | Dataset for label mapping: `coco`, `eurosat` (`eurosat` auto-provides 10 names to `class_names`) |
| `task.class_names` | | Explicit label list – forwarded to adapter via `cvf/core/pipeline/stages/adapter.py:13` (`model.class_names` / `task_class_names` merged) |
| `task.conf_threshold` | | Detection confidence threshold |
| `task.iou_threshold` | | Detection NMS IoU threshold (FP16-safe, `cvf/tasks/detection.py:30`) |
| `task.top_k` | | Classification top-K |
| `task.threshold` | | Classification score threshold |
| `input` | | Input image/video path (CLI arg overrides; `.mp4/.avi/.mov` triggers video pipeline `cvf/cli/main.py:202`) |
| `backend.type` | ✅ | `onnxruntime` (FP16 input auto-cast `cvf/backends/onnxruntime.py:51`) |
| `device.type` | ✅ | `cpu`, `cuda` |
| `pipeline.stages` | | List of stages to execute |
| `output_dir` | | Output directory for artifacts (contains `*_result.json` with `top_class_names/top1_name` `cvf/core/contracts/runtime/classification.py:30` + `*_vis.jpg/.mp4`) |
