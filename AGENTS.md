# Repository Guidelines

**Project:** XAIP — AI Inference Pipeline (ONNX Runtime)
**Root:** `/home/bht/CODE/xaip`
**Python:** 3.12 | **Runtime:** ONNX Runtime (CPU / CUDA) | **Package Manager:** Conda + pip

---

## Project Overview

XAIP is a modular inference framework for running, debugging, and evaluating computer vision models via ONNX Runtime. It supports six model families across three tasks (object detection, classification, handwritten text recognition) with four operating modes (inference, tracking, debug, evaluate) and quantization-aware comparison tooling.

---

## Architecture & Data Flow

### High-Level Structure

```
xaip/
├── main.py                    # CLI entry: python main.py <config.yaml>
├── run.sh                     # Auto-conda + run wrapper
├── config.yaml                # User runtime config
├── config_templates/          # Schema-only templates per model
├── AI_Script/
│   ├── core/                  # Pipelines + factory dispatch
│   ├── configs/               # Internal model defaults + datasets.json
│   ├── models/                # Model implementations (registry/factory)
│   ├── preprocess/            # Preprocessors (registry/factory)
│   ├── postprocess/           # Postprocessors (registry/factory)
│   ├── tracker/               # ByteTrack (detection only)
│   ├── evaluate/              # Evaluation tasks (registry/factory)
│   └── Functions/             # Standalone QA/utility scripts
└── inputs/                    # Sample datasets (images, videos, labels)
```

### Pipeline Dispatch (Factory Precedence)

```python
# AI_Script/core/factory.py:_get_pipeline()
if self.inference_mode:
    return Pipeline_Tracking if tracking_option else Pipeline_Inference
elif self.debug_mode:
    return Pipeline_Debug
elif self.evaluate_mode:
    return Pipeline_Evaluate
else:
    raise ValueError("Invalid pipeline")
```

**Exactly one mode flag must be `true`.**

| Pipeline | Config Flags | Input Type | Output |
|----------|-------------|------------|--------|
| `Pipeline_Inference` | `inference_mode=true` | Image / folder / video / `.npy` | Annotated images (`.jpg`), video (`.mkv`) |
| `Pipeline_Tracking` | `inference_mode=true` + `tracking_option=true` | Video (or single image) | Tracking video (`.mkv`) with IDs |
| `Pipeline_Debug` | `debug_mode=true` | **Single image only** | Annotated image + profiler JSON + `intermediate_outputs/` per-layer JSONs |
| `Pipeline_Evaluate` | `evaluate_mode=true` | **Folder with `images/` + `labels/`** | `evaluate - <model> - <prec> - CPU - <ts>.json` |

### Registry / Factory Pattern (All Domains)

Each component family uses identical structure:

```python
# Registry (decorator-based)
class XRegistry:
    _registry: Dict[str, Type[BaseX]] = {}
    @classmethod def register(cls, name): ...
    @classmethod def get(cls, name): ...

# Factory (stateless)
class XFactory:
    @staticmethod def create(config, **kwargs):
        return XRegistry.get(config["model_name"])(config, **kwargs)

# Implementation
@XRegistry.register("model_name")
class ConcreteX(BaseX):
    def __init__(self, config, ...): ...
    def __call__(self, input): ...
```

- **Models:** `ModelRegistry` / `ModelFactory` / `BaseModel` → `AI_Script/models/`
- **Preprocess:** `PreprocessRegistry` / `PreprocessorFactory` / `BasePreprocess` → `AI_Script/preprocess/`
- **Postprocess:** `PostProcessorRegistry` / `PostprocessorFactory` / `BasePostprocess` → `AI_Script/postprocess/`
- **Evaluate:** `EvaluateRegistry` / `EvaluateFactory` / `BaseEvaluate` → `AI_Script/evaluate/`

Auto-discovery via `AI_Script/models/__init__.py` etc. (imports trigger `@register` decorators).

### Inference Data Flow (Single Item)

```
input_source
    │
    ▼
check_file()  ──► determines input_type
    │
    ├── image/npy/array ──► _process_single_item()
    │       │
    │       ├─ preprocessor(image) → np.ndarray [1, C, H, W]
    │       └─ model(preprocessed) → List[np.ndarray]  (ONNX outputs)
    │
    └── video ──► frame loop: same per-frame + VideoWriter (FFV1)
```

### Tracking Data Flow

```
frame
    │
    ▼
preprocessor → model → raw_outputs
    │
    ▼
Adapter_Detection (YOLOv5/DETR) → {boxes[N,4]xyxy, conf[N], class[N]}
    │
    ▼
conf_threshold → NMS (torchvision.ops.nms)
    │
    ▼
ByteTrack.update(detections[:,:5], img_w, img_h) → List[STrack]
    │
    ▼
unletterbox(boxes, original_shape, target_size) → orig coords
    │
    ▼
_draw_box_id() → cv2.rectangle + cv2.putText(ID)
    │
    ▼
VideoWriter / imshow
```

### Debug Pipeline (ONNX Graph Manipulation)

`Pipeline_Debug._dump_model()`:
1. Loads ONNX model → `shape_inference.infer_shapes()`
2. Extracts `value_info` + initializers for dtype/shape metadata
3. Creates modified graph with **Identity** nodes attached to **every tensor** (inputs, outputs, intermediates, initializers)
4. Runs inference with `onnxruntime` (profiling enabled) → captures all Identity outputs
5. Writes per-layer JSON: `intermediate_outputs/{idx:04d}__{layer_name}.json`
6. Saves profiler trace: `Output profiler - <model> - <timestamp>.json`

### Evaluation Data Flow

```
dataset_root/
  images/
  labels/
    │
    ▼
Pipeline_Evaluate.run(folder)
    │
    ├─ iterate sorted image files
    ├─ preprocessor(image) → model → raw_output
    ├─ task evaluator.benchmark_single(pred, img_path, label_path) → dict
    └─ aggregate via benchmark_toltal() → final JSON
```

---

## Key Directories

| Path | Purpose |
|------|---------|
| `AI_Script/core/` | Pipeline implementations + `factory_pipeline` dispatch |
| `AI_Script/models/` | ONNX model wrappers (`BaseModel` + per-model classes) |
| `AI_Script/preprocess/` | Input transforms (letterbox, normalize, HWC→CHW, resize) |
| `AI_Script/postprocess/` | NMS, CTC decode, box/label drawing, adapter for detection |
| `AI_Script/tracker/ByteTrack/` | Kalman filter + Hungarian matching tracking |
| `AI_Script/evaluate/` | Metrics: accuracy/PRF/MCC (cls), mAP (det), exact-match/CER/WER (HTR) |
| `AI_Script/configs/` | Internal defaults per model + `datasets.json` (label mappings) |
| `AI_Script/Functions/` | One-off QA scripts (data hygiene, visualization, weight dump) |
| `config_templates/` | Schema-only YAML templates per model (7 files; `yolov5.yaml` = COCO, `yolov5_dota.yaml` separate) |
| `test_configs/` | 55 verified run configs (11 variants × 5 precisions: 7 exported weight sets + 4 fine-tuned YOLOv5 custom models; naming `<variant>_<label>.yaml`) |
| `inputs/dataset/` | Sample datasets: `Dataset_Number` (cls), `Dataset_HTR`, `COCO_Val_2017` (det) |

---

## Development Commands

### Environment Setup

```bash
# Recommended (auto-creates conda env 'xaip')
./run.sh config.yaml

# Manual
conda env create -f environment.yml
conda activate xaip
pip install onnx onnxruntime opencv-python numpy pyyaml torch torchvision pillow scikit-learn matplotlib pandas
# GPU:
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Run Inference / Tracking / Debug / Evaluate

```bash
# Inference (edit config.yaml first)
python main.py config.yaml

# Or start from a template
python main.py config_templates/yolov5.yaml

# Tracking (requires video input + detection model)
# Set: inference_mode: true, tracking_option: true in config

# Debug (single image only, dumps intermediate layers)
# Set: debug_mode: true

# Evaluate (folder with images/ + labels/)
# Set: evaluate_mode: true
```

### Compare Benchmarks / Intermediates (Python API)

```python
from AI_Script.core.factory import factory_pipeline

factory = factory_pipeline(
    model_name="yolov5", precision_format="fp32",
    inference_mode=False, evaluate_mode=False, debug_mode=False,
    output_path="/path/to/output"
)

# Compare two evaluation JSONs → charts
factory.benchmark_dashboard((
    "outputs/evaluate - yolov5 - fp32 - CPU - ...json",
    "outputs/evaluate - yolov5 - int8 - CPU - ...json"
))

# Compare intermediate tensors (MSE per layer)
factory.compare_intermediate((
    "outputs/Output intermediate layer - yolov5 - ...json",
    "outputs/Output intermediate layer - yolov5_int8 - ...json"
))
```

---

## Code Conventions & Common Patterns

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `Pipeline_Inference`, `LENET5`, `ByteTrack` |
| Functions/Methods | snake_case | `_process_single_item`, `benchmark_toltal` |
| Config keys | snake_case | `precision_format`, `target_size`, `conf_threshold` |
| Registry keys | lowercase | `"lenet5"`, `"draw_box"`, `"classification"` |
| Files | snake_case | `pipeline_inference.py`, `registry_model.py` |

### Base Classes (Inheritance Contracts)

```python
# AI_Script/models/base_model.py
class BaseModel(ABC):
    def __init__(self, config, debug_mode=False):
        self.config = config
        self.debug_mode = debug_mode
        self.session = self.load_model()  # onnxruntime.InferenceSession
    def load_model(self): ...             # selects weight by precision_format
    def __call__(self, input): ...        # returns List[np.ndarray]

# AI_Script/preprocess/base_preprocess.py
class BasePreprocess(ABC):
    def __call__(self, image_path_or_array): ...  # returns np.ndarray

# AI_Script/postprocess/base_postprocess.py
class BasePostprocess(ABC):
    def __call__(self, model_output, raw_input=None, save=True): ...

# AI_Script/evaluate/base_evaluate.py
class BaseEvaluate(ABC):
    @abstractmethod def benchmark_single(self, pred, img_path, label_path): ...
    @abstractmethod def benchmark_toltal(self, benchmark_dict): ...  # note spelling
```

### Error Handling

- **Config validation** at multiple layers: `main.py`, `factory_pipeline.__init__`, `BaseModel._session_option`, every pipeline `__init__` — all check `output_path` required.
- **Fail-fast** with descriptive `ValueError` / `FileNotFoundError` / `NotADirectoryError`.
- **No silent fallbacks** for missing weights — raises if `weight_path` and internal `weight_path_{prec}` both absent.

### Config Injection Pattern

```python
# factory_pipeline.__init__: single source of truth
self.config = self.load_config(self.model_name)      # loads AI_Script/configs/<model>.yaml
self.config["output_path"] = self.output_path        # inject required paths
self.config["display"] = self.display
# then apply CLI overwrites
if self.weight_path: self.config["weight_path"] = str(self.weight_path)
if extras: self.config.update(extras)                # forward unknown YAML keys (main.py)
```

Extra YAML keys not handled by `main.py` are forwarded via `extras` — this is how
`input_dtype` and `graph_optimization_level` reach model/preprocessor code. All downstream
components receive the **same mutated config dict**.

### Tensor / Image Conventions

| Aspect | Convention |
|--------|------------|
| Color order | **BGR** (OpenCV default) → preprocess converts to RGB if needed |
| Layout | Model input: **NCHW** (`[1, C, H, W]`) |
| Normalization | Per-model `mean`/`std` in internal config (e.g., ImageNet for ResNet/ViT; zeros/ones for YOLO/DETR) |
| Letterboxing | Detection preprocessors use `letterbox()` → preserves aspect ratio, pads to `target_size` |
| Detection output | Postprocess expects `[x1, y1, x2, y2]` (xyxy) after adapter + unletterbox |

---

## Important Files

| File | Role |
|------|------|
| `main.py` | CLI entry, config load, factory init, run |
| `run.sh` | Conda bootstrap + run |
| `AI_Script/core/factory.py` | Pipeline dispatch, config merging, `benchmark_dashboard()`, `compare_intermediate()` |
| `AI_Script/core/pipeline_inference.py` | Image/folder/video inference loop |
| `AI_Script/core/pipeline_tracking.py` | ByteTrack integration |
| `AI_Script/core/pipeline_debug.py` | ONNX graph rewrite + intermediate dump |
| `AI_Script/core/pipeline_evaluate.py` | Batch evaluation orchestrator |
| `AI_Script/core/pipeline_benchmark.py` | Comparison charts from two eval JSONs |
| `AI_Script/core/pipeline_compare_intermediate.py` | MSE diff of intermediate tensors |
| `AI_Script/models/factory_model.py` | Model factory + registry import |
| `AI_Script/preprocess/factory_preprocess.py` | Preprocessor factory |
| `AI_Script/postprocess/factory_postprocess.py` | Postprocessor factory |
| `AI_Script/evaluate/factory_evaluate.py` | Evaluator factory |
| `AI_Script/configs/datasets.json` | Label name → index mappings for all datasets |
| `config.yaml` / `config_templates/*.yaml` | User runtime config + templates |

---

## Runtime / Tooling Preferences

| Category | Choice |
|----------|--------|
| Python | 3.12 (enforced by `environment.yml`) |
| Inference Engine | ONNX Runtime (`onnxruntime` / `onnxruntime-gpu`) |
| Execution Providers | `CPUExecutionProvider` (default), `CUDAExecutionProvider` (if `target.device: gpu`) |
| Computer Vision | OpenCV (`cv2`) |
| Deep Learning (Tracker only) | PyTorch + torchvision (ByteTrack) |
| Numerics | NumPy |
| Metrics | scikit-learn (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `matthews_corrcoef`) |
| Plotting | Matplotlib + Pandas (benchmark dashboard) |
| Config Format | YAML (PyYAML) |
| No test framework | — |

### Hardware Constraints

- **GPU**: Requires `onnxruntime-gpu` + CUDA 11.8 (per install hint). Falls back to CPU silently if GPU unavailable.
- **FP16**: most exports have FP16 IO (input auto-cast by preprocessor). Exception: `Yolov5s_DOTA_Satellite_FP16.onnx` keeps FP32 IO → set `input_dtype: "fp32"`.
- **BF16**: exported as `*_BF16_with_cast.onnx` with FP32 IO (cast inside graph) → no override needed. ONNX Runtime ≥ 1.16. The 4 fine-tuned YOLOv5 models export plain `*_BF16.onnx` (no `_with_cast` suffix) **but still have FP32 IO** (verified) → same, no override.
- **INT8**: two families share the `precision_format: "int8"` label, selected via `weight_path`: `*_Mixed_FP32_INT8.onnx` (static QDQ) and `*_INT8_dynamic_W8A8.onnx` (dynamic MatMul/Gemm quant); both expose FP32 IO. The 4 fine-tuned YOLOv5 models name the dynamic file `*_INT8_dynamic.onnx` (no `_W8A8` suffix) — behavior identical.
- **Smoke (YOLOv5s_Smoke)**: trained on square-stretch input → **requires `resize_mode: "stretch"`** (forwarded via extras to the preprocessor). Under default letterbox the detections collapse (conf 0.43 → 0.017).
- **Session options**: always passed to `InferenceSession` (profiling on). `graph_optimization_level` config key: `disable | basic | extended | all` (default `all`); the CRNN FP16 LSTM export requires `"extended"` (segfault under `all`).

---

## Testing & QA

### Test Framework Status

**No standard test suite exists** (no `pytest`, `unittest`, `tests/` directory, or `*test*.py` files).

QA is **pipeline-centric**:

1. **Evaluation pipelines** (`evaluate_mode`) produce metrics JSON.
2. **Benchmark dashboard** (`Pipeline_Benchmark`) compares two eval runs → plots.
3. **Intermediate comparison** (`Pipeline_Compare_Intermediate`) computes layer-wise MSE.
4. **Standalone scripts** in `AI_Script/Functions/` for data hygiene and ad-hoc checks.

### Verified Weight Matrix

55 inference runs pass on CPU (11 variants × fp32/fp16/mixed/bf16/w8a8, one test image each). Original 7 variants (35 runs): visually verified with consistent predictions. Fine-tuned YOLOv5 variants LPR / Helmet / Seatbelt / Smoke (16 runs, added 2026-11-03): verified by exit codes + per-run logs, IO types confirmed by ONNX inspection. Per-run configs in `test_configs/yolov5_{lpr,helmet,seatbelt,smoke}_<label>.yaml`. Known quirks (see Hardware Constraints): YOLOv5 DOTA FP16 needs `input_dtype: "fp32"`, CRNN FP16 needs `graph_optimization_level: "extended"`, Smoke model needs `resize_mode: "stretch"`.

### How to Run Evaluations

```yaml
# Minimal evaluation config (config_templates/lenet5.yaml style)
input_path: "/path/to/Dataset_Number"   # must contain images/ + labels/
output_path: "/path/to/output"          # REQUIRED
model_name: "lenet5"
precision_format: "fp32"
dataset: "number_character"
evaluate_mode: true
inference_mode: false
debug_mode: false
target:
  device: "cpu"
```

```bash
python main.py config_templates/lenet5.yaml
# Output: <output_path>/evaluate - lenet5 - fp32 - CPU - <timestamp>.json
```

### Sample Datasets (Ready to Use)

| Dataset | Task | Path | Label Format |
|---------|------|------|--------------|
| `Dataset_Number` | Classification (47-class) | `AI_Script/inputs/dataset/Dataset_Number/` | Single integer per `.txt` |
| `Dataset_HTR` | HTR | `AI_Script/inputs/dataset/Dataset_HTR/` | Raw UTF-8 text per `.txt` |
| `COCO_Val_2017` | Detection (YOLO fmt) | `AI_Script/inputs/dataset/COCO_Val_2017/` | `class_id xc yc w h` (normalized) |

Class mappings: `AI_Script/configs/datasets.json` (`coco` 80, `dota_satellite` 16, `dota` 15, `number_character` 47, `eurosat` 10, `imagenet` 1000, `yolo_lpr` 1, `yolo_helmet` 2, `yolo_seatbelt` 4, `yolo_smoke` 1).

### QA Scripts (`AI_Script/Functions/`)

| Script | Purpose |
|--------|---------|
| `check_similarity_images_labels.py` | Sync `images/` ↔ `labels/` folders (delete orphans) |
| `Draw_chart.py` | Standalone benchmark chart generator (hard-coded paths) |
| `draw_box.py` | Quick single-box visualization on COCO image |
| `mv_all_file_subfolder_to_out.py` | Flatten nested subfolders, dedupe names |
| `format_dataset_eurosat.py` | EuroSAT label generation + sampling (commented examples) |
| `get_intermediate_weight_QO.py` | Dump ONNX initializers to JSON (shape/dtype/values) |
| `Download_MNIST_47.py` | Download EMNIST bymerge (47 classes), save as JPG + `.txt` |

### Known Gaps (For Assistant Awareness)

1. **No automated tests** — consider adding `pytest` with fixtures for `Compare_Output_Intermediate`, `check_similarity_images_labels`, metric calculations.
2. **HTR aggregate metrics incomplete** — `benchmark_toltal` returns only exact-match accuracy; `_cer`/`_wer` helpers defined but not aggregated.
3. **Typo: `benchmark_toltal`** — consistent across base + implementations; rename to `benchmark_total` if breaking change acceptable.
4. **`Draw_chart.py` hard-coded paths** — parameterize or wrap as CLI for reuse.
5. **Internal weight paths** in `AI_Script/configs/*.yaml` reference original machine (`/home/bht/...`); users **must** override `weight_path` or replicate directory structure.

---

## Quick Reference: Minimal Configs by Mode

### Inference (Image)
```yaml
input_path: "image.jpg"
weight_path: "model.onnx"
output_path: "outputs/"
model_name: "yolov5"
precision_format: "fp32"
target:
  device: "cpu"
inference_mode: true
debug_mode: false
evaluate_mode: false
```

### Tracking (Video)
```yaml
input_path: "video.mp4"
weight_path: "yolov5.onnx"
output_path: "outputs/"
model_name: "yolov5"
precision_format: "fp32"
target:
  device: "gpu"
inference_mode: true
tracking_option: true
debug_mode: false
evaluate_mode: false
display: false
```

### Debug (Single Image)
```yaml
input_path: "image.jpg"
weight_path: "model.onnx"
output_path: "outputs/"
model_name: "resnet50"
precision_format: "fp32"
target:
  device: "cpu"
debug_mode: true
inference_mode: false
evaluate_mode: false
```

### Evaluate (Dataset Folder)
```yaml
input_path: "AI_Script/inputs/dataset/Dataset_Number"
weight_path: "lenet5.onnx"
output_path: "outputs/"
model_name: "lenet5"
precision_format: "fp32"
dataset: "number_character"
evaluate_mode: true
inference_mode: false
debug_mode: false
target:
  device: "cpu"
```