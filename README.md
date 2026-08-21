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
# YOLOv5 Detection
python -m cvf.cli.main run cvf/configs/inference.yaml /path/to/image.jpg

# ResNet50 Classification
python -m cvf.cli.main run cvf/configs/classification.yaml /path/to/image.jpg

# LeNet5 Classification (MNIST-style)
python -m cvf.cli.main run cvf/configs/lenet5.yaml /path/to/image.png

# CRNN HTR (Handwritten Text Recognition)
python -m cvf.cli.main run cvf/configs/crnn.yaml /path/to/text_image.jpg
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

| Family | Variant | Task | Input | Output |
|--------|---------|------|-------|--------|
| YOLO | yolov5 | Detection | 640×640 RGB | `DetectionOutput(boxes, scores, class_ids)` |
| ResNet | resnet50 | Classification | 224×224 RGB | `ClassificationOutput(logits, probs, top-k)` |
| ResNet | lenet5 | Classification | 32×32 Gray | `ClassificationOutput(logits, probs, top-k)` |
| CRNN | crnn | HTR | 100×32 Gray | `HTROutput(text, confidence)` |

---

## Extending CVF

### Add a New Model

```python
# cvf/models/myfamily/myvariant/model.py
from cvf.core.registry import model_registry

class MyModel:
    def __init__(self, config): ...
    def load(self, providers): ...
    def infer(self, input_tensor): ...

# Register
model_registry.register("myfamily.myvariant", lambda c: MyModel(c))

# cvf/models/myfamily/myvariant/preprocess.py
class MyPreprocessor:
    def __call__(self, image): ...
    
# cvf/models/myfamily/myvariant/output_adapter.py
class MyAdapter:
    def adapt(self, raw_output): ...  # → CanonicalOutput
```

### Add a New Task

```python
# cvf/tasks/mytask.py
from cvf.core.registry import postprocess_registry

class MyTaskPostprocessor:
    def __call__(self, canonical_output, task_config): ...
    
postprocess_registry.register("mytask", lambda c: MyTaskPostprocessor())
```

### Add a New Backend

```python
# cvf/backends/mybackend.py
from cvf.backends.base import InferenceBackend

class MyBackend(InferenceBackend):
    def load(self, model, device): ...
    def infer(self, handle, input_tensor): ...
    def get_providers(self, device): ...
```

---

## Project Structure

```
cvf/
├── core/
│   ├── config/           # loader.py, schema.py (Pydantic)
│   ├── contracts/        # runtime/ (dataclasses) + config/ (Pydantic)
│   ├── compatibility/    # resolver.py (pre-execution validation)
│   ├── pipeline/         # pipeline.py, stage.py, context.py
│   ├── registry/         # registry.py + specialized registries
│   └── artifacts/        # manager.py
├── models/
│   ├── yolo/yolov5/      # Detection
│   ├── resnet/resnet50/  # Classification
│   ├── resnet/lenet5/    # Classification (MNIST)
│   └── crnn/crnn/        # HTR
├── tasks/
│   ├── detection/        # postprocess, metrics
│   ├── classification/   # postprocess, metrics
│   └── htr/              # postprocess
├── backends/
│   ├── base.py           # InferenceBackend ABC
│   └── onnxruntime/      # ONNX Runtime implementation
├── devices/
│   ├── cpu.py            # CPU device
│   └── cuda.py           # CUDA device
├── benchmark/            # Latency, throughput, memory
├── evaluation/           # Accuracy metrics
├── plugins/builtin.py    # Auto-registration
├── cli/main.py           # CLI entry point
└── configs/              # Example YAML configs
```

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
| `model.mean` / `model.std` | | Normalization values |
| `task.type` | ✅ | `detection`, `classification`, `htr` |
| `task.conf_threshold` | | Detection confidence threshold |
| `task.iou_threshold` | | Detection NMS IoU threshold |
| `task.top_k` | | Classification top-K |
| `backend.type` | ✅ | `onnxruntime` |
| `device.type` | ✅ | `cpu`, `cuda` |
| `pipeline.stages` | | List of stages to execute |
| `output_dir` | | Output directory for artifacts |

---

## Roadmap (Phase 3+)

- [ ] **Benchmark Runner** — Full latency/throughput/memory implementation
- [ ] **Evaluation** — Ground truth evaluation with mAP, accuracy
- [ ] **Optimization** — Quantization (PTQ), Pruning, Distillation
- [ ] **Backends** — PyTorch, TensorRT, OpenVINO, TFLite
- [ ] **Devices** — MPS, NPU support
- [ ] **Tasks** — Segmentation, Pose estimation
- [ ] **Model Zoo** — Pre-trained model configs
- [ ] **Python Package** — `pip install cvf`

---

## License

MIT License — see LICENSE file for details.

---

## Acknowledgments

Built on top of:
- **ONNX Runtime** — High-performance inference
- **OpenCV** — Image preprocessing
- **NumPy/SciPy** — Numerical computing
- **Pydantic** — Config validation