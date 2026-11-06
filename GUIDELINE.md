# GUIDELINE — Run Guide

> Setup → Config → Run → Outputs. All paths in examples are placeholders (`/path/to/...`) — point them to **your** locations; the project has no preferred storage layout.
> Architecture & dev conventions → [AGENTS.md](AGENTS.md) · Overview & structure → [README.md](README.md)

## 1. Environment Setup

### Python version

**Python 3.12** (enforced by `environment.yml`).

### Environment creation & activation

```bash
./run.sh config.yaml            # auto-creates the conda env 'xaip' from environment.yml if missing, then runs

# or manually
conda env create -f environment.yml
conda activate xaip
```

### Dependency installation

`environment.yml` pins everything: `onnx`, `onnxruntime`, `opencv-python`, `numpy`, `PyYAML`, `Pillow`, `scikit-learn`, `matplotlib`, `pandas`, `torch`, `torchvision`, `lap`, `cython_bbox` — CPU setup needs nothing extra.

GPU runs additionally require the CUDA-capable packages (CUDA 11.8):

```bash
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 2. Configuration

Each run = one YAML file, at any location, passed to `main.py`. Ready-made configs: schema-only templates per model in `config_templates/<model_name>.yaml` (pure `/path/to/...` placeholders — fill in your paths). The example configs in `test_configs/` / `eval_configs/` / `debug_configs/` (fine-tuned YOLOv5 run configs in `test_configs/` — quick reference below) contain **absolute paths from the machine they were built on** — edit `input_path`, `weight_path`, `output_path` before reuse.

### Required parameters

| Parameter | Possible values | Notes |
|-----------|----------------|-------|
| `input_path` | image (`.png`/`.jpg`/`.jpeg`) \| image folder \| video (`.mp4`/`.avi`/`.mov`/`.mkv`) \| `.npy` | valid type depends on mode (§ 4) |
| `weight_path` | `/path/to/your/models/<name>.onnx` | always set explicitly (see weight rules below) |
| `output_path` | `/path/to/your/output/` (directory) | auto-created if missing |
| `model_name` | `yolov5` \| `detr` \| `lenet5` \| `resnet50` \| `vit` \| `crnn` | `resnet` is only an alias in code — use `resnet50` in configs |
| `precision_format` | `fp32` \| `fp16` \| `int8` \| `bf16` | a label; the file actually loaded is chosen **by file name** in `weight_path` |
| `dataset` | `coco` \| `dota` \| `dota_satellite` \| `number_character` \| `eurosat` \| `imagenet` \| `yolo_lpr` \| `yolo_helmet` \| `yolo_seatbelt` \| `yolo_smoke` | label mapping only (`AI_Script/configs/datasets.json`); per-model default; not needed by `crnn` |
| `target.device` | `cpu` \| `gpu` | `gpu` requires onnxruntime-gpu (§ 1) |
| mode flags | `inference_mode`, `tracking_option`, `debug_mode`, `evaluate_mode` → `true`\|`false` | all 4 keys must be present; exactly **1** main mode `true` (§ 4) |

Rules:

- If several main modes are `true`, precedence is `inference_mode` → `debug_mode` → `evaluate_mode`.
- Weight selection is **by file name**: `*_FP32.onnx` · `*_FP16.onnx` · `*_Mixed_FP32_INT8.onnx` / `*_INT8_dynamic_W8A8.onnx` · `*_BF16_with_cast.onnx`. Weights may live anywhere.
- Unset `weight_path` silently falls back to the internal `weight_path_<precision>` defaults, which point to the original machine's directories (`AI_Script/configs/*.yaml`) — likely missing on your disk. Treat `weight_path` as always set.
- Keys not in the tables above are forwarded verbatim into the pipeline → they can override model-internal values (`mean`, `std`, `target_size`, `transpose_output`, `weight_path_*`, `default_postprocess`, `task`). Caveat: `mean`/`std` are **inert for YOLO detection preprocessing** — the applied pipeline is RGB `[0,1]` (÷255) and the mean/std math is commented out in `AI_Script/preprocess/detection/yolo.py`; keep them at `[0,0,0]`/`[1,1,1]` (non-default mean/std or 0–255 scale inputs collapse fine-tuned models, mAP → 0).

### Optional parameters

| Parameter | Possible values | Default | Notes |
|-----------|----------------|---------|-------|
| `display` (alias `display_option`) | `true` \| `false` | `false` | real-time preview window, `q` to exit |
| `conf_threshold` | float 0–1 | `0.3` | detection only: box filter by confidence |
| `iou_threshold` | float 0–1 | `0.5` | detection only: NMS + evaluate |
| `resize_mode` | `letterbox` \| `stretch` | `letterbox` | YOLO preprocess + unletterbox when drawing |
| `input_dtype` | `fp32` \| `fp16` | `fp16` if `precision_format: fp16`, else `fp32` | force input tensor dtype |
| `graph_optimization_level` | `disable` \| `basic` \| `extended` \| `all` | `all` | ORT session optimization |
| `inference_workers` | int ≥ 1 | `1` | tracking only: parallel threads per frame |
| `debug_dump_initializers` | `true` \| `false` | `false` | debug only: also dump weights/biases to JSON |
| `method` | `greedy` \| `beam_search` \| `prefix_beam_search` | `beam_search` | CRNN only: CTC decoding |
| `beam_size` | int | `10` | CRNN only: beam decode width |

### Configuration examples

```yaml
# ---------- Inference (image / folder / video / .npy) ----------
input_path: "/path/to/your/input"
weight_path: "/path/to/your/models/yolov5s_FP32.onnx"
output_path: "/path/to/your/output"

model_name: "yolov5"
precision_format: "fp32"
dataset: "coco"

target:
  device: "cpu"

inference_mode: true
tracking_option: false
debug_mode: false
evaluate_mode: false
display: false
```

```yaml
# ---------- Tracking (video + detection model only) ----------
input_path: "/path/to/your/video.mp4"          # video input
weight_path: "/path/to/your/models/yolov5s_FP32.onnx"
output_path: "/path/to/your/output"

model_name: "yolov5"
precision_format: "fp32"
dataset: "coco"

target:
  device: "cpu"

inference_mode: true
tracking_option: true                          # <-- enables ByteTrack
debug_mode: false
evaluate_mode: false
display: false
```

```yaml
# ---------- Debug (single image / .npy only) ----------
input_path: "/path/to/your/single-image.jpg"
weight_path: "/path/to/your/models/resnet50_FP32.onnx"
output_path: "/path/to/your/output"

model_name: "resnet50"
precision_format: "fp32"

target:
  device: "cpu"

inference_mode: false
tracking_option: false
debug_mode: true
evaluate_mode: false
debug_dump_initializers: false                 # optional: also dump weights/biases
```

```yaml
# ---------- Evaluate (folder containing images/ + labels/) ----------
input_path: "/path/to/your/dataset"            # must contain images/ + labels/
weight_path: "/path/to/your/models/lenet5_FP32.onnx"
output_path: "/path/to/your/output"

model_name: "lenet5"
precision_format: "fp32"
dataset: "number_character"

target:
  device: "cpu"

inference_mode: false
tracking_option: false
debug_mode: false
evaluate_mode: true
```

Evaluate label pairing: `labels/<image-name>.txt` — detection: `class_id xc yc w h` (normalized); classification: single integer; HTR: plain text. Sample datasets ship in-repo at `AI_Script/inputs/dataset/`.

Known precision quirks (3 exceptions; full background incl. BF16/INT8 export details → AGENTS.md "Hardware Constraints"):

- YOLOv5 DOTA FP16 keeps FP32 IO → `input_dtype: "fp32"`
- CRNN FP16 (LSTM export) segfaults under full optimization → `graph_optimization_level: "extended"`
- YOLOv5 fine-tuned Smoke & Seatbelt → `resize_mode: "stretch"` (validated); Helmet keeps `letterbox` — measured per model, rationale documented in each config's header

Fine-tuned YOLOv5 quick reference — ready-to-run in `test_configs/`, preprocess validated per model (single-image sweep: letterbox vs stretch vs normalization variants; artifacts + method → `~/Downloads/xaip_preprocess_experiments/`):

| Config | `dataset` | `resize_mode` | Weights (original machine) |
|--------|-----------|---------------|-----------------------------|
| `test_configs/yolov5_helmet_fp32.yaml` | `yolo_helmet` | `letterbox` | `YOLO_HelmetDetection/` |
| `test_configs/yolov5_seatbelt_fp32.yaml` | `yolo_seatbelt` | `stretch` | `YOLO_SeatbeltDetection/` |
| `test_configs/yolov5_smoke_fp32.yaml` | `yolo_smoke` | `stretch` | `YOLO_SmokeDetection/` |

Each config declares all four `weight_path_<precision>` entries → switching precision = edit `precision_format` only. These ONNX exports accept 640×640 only.

## 3. Execution

```bash
python main.py config.yaml       # equivalently: ./run.sh config.yaml
```

- The config path is the only CLI argument.
- Fail-fast: missing required keys (esp. `output_path` — there is no default) exit immediately with an error.
- With `display: true` a preview window opens; press `q` to quit.

## 4. Output

### Output directory

Everything is written to `output_path` (auto-created; required, no default).

### Generated artifacts

| Mode (flags) | Accepted input | Artifacts written to `output_path` |
|--------------|----------------|-------------------------------------|
| Inference (`inference_mode: true`) | image / folder / video / `.npy` | annotated `.jpg` (detection: `<model> \| <precision> \| <timestamp>.jpg`; classification/HTR: `<model> <timestamp>.jpg`); video: `<model> video_detection <timestamp>.mkv` |
| Tracking (`inference_mode: true` + `tracking_option: true`) | video + detection model | `<model> video_tracking <timestamp>.mkv` (FFV1; keeps ByteTrack IDs) |
| Debug (`debug_mode: true`) | single image / `.npy` | postprocessed `.jpg`, `intermediate_outputs/` (per-layer JSONs), ORT profiler JSON (§ 4.3) |
| Evaluate (`evaluate_mode: true`) | folder with `images/` + `labels/` | `evaluate - <model> - <precision> - CPU - <timestamp>.json` (`run_target` is hard-coded `"CPU"`) |

Notes:

- Debug mode also writes a temporary instrumented graph `dump_model.onnx` — removed automatically when the run completes.
- Profiling is enabled on every ORT session, so an `Output profiler - <model> ... .json` appears in `output_path` in **every** mode.

### JSON schemas

**Evaluate JSON** — fixed 4-section skeleton for every task:

```json
{
  "Model_Information": {
    "model_name": "lenet5",
    "run_target": "CPU",
    "task": "classification",
    "weight_path": "/path/to/your/models/...onnx",
    "precision_format": "fp32",
    "model_size_mb": "12.34 MB",
    "target_size": [224, 224]
  },
  "Dataset_Information": {
    "dataset_path": "/path/to/your/dataset",
    "number_images": 100
  },
  "Performance": {
    "total_time_s": 12.3,
    "throughput_fps": 8.1
  },
  "Evaluate": {
    "images": { "<image>.jpg": { "image_path": "...", "metrics": { "...task-specific..." } } },
    "overall_metrics": { "...task-specific..." }
  }
}
```

Metrics per task (`task` values come from the internal model config; the evaluate tasks are registered under these names):

| Task | `images/<img>.metrics` | `overall_metrics` |
|------|------------------------|-------------------|
| `classification` | `correct` (bool) · `predict` (int) · `ground_truth` (int) | `accuracy` · `macro_avg` / `micro_avg` / `weighted_avg` → `{precision, recall, f1-score}` · `mcc` · `Per_class` (per-class report) · `Confusion_matrix` |
| `object_detection` | `mAP@0.5` · `mAP@[0.5:0.95]` · `total_missed_objects` | `avg_mAP@0.5` · `avg_mAP@[0.5:0.95]` |
| `htr` | `correct` (bool) · `predict` (str) · `ground_truth` (str) | `accuracy` only — exact match (CER/WER helpers exist but are not aggregated yet) |

**Debug per-layer JSON** (`intermediate_outputs/NNNN__<layer_name>.json`) — one shared schema for every file:

```json
{
  "layer_name": "<tensor name>",
  "index": 12,
  "dtype": "float32",
  "shape": [1, 64, 112, 112],
  "values": [ "/* nested list matching shape */" ]
}
```

### Comparing two runs (no CLI mode — Python API)

```python
from AI_Script.core.factory import factory_pipeline

factory = factory_pipeline(
    model_name="yolov5", precision_format="fp32",
    inference_mode=False, evaluate_mode=False, debug_mode=False,
    output_path="/path/to/your/output"
)

# benchmark: 2 evaluate JSONs -> charts in Benchmark_Chart_<timestamp>/
# (overall_metrics.png, avg_metrics.png, per_class_metrics.png,
#  confusion_matrices.png, performance_hardware.png)
factory.benchmark_dashboard((
    "/path/to/your/output/evaluate - ... - fp32 - CPU - ....json",
    "/path/to/your/output/evaluate - ... - int8 - CPU - ....json"
))

# precision diff: 2 debug runs (intermediate_outputs/ folder or single JSON) -> MSE per layer
factory.compare_intermediate((
    "/path/to/your/output/intermediate_outputs",
    "/path/to/your/output_int8/intermediate_outputs"
))
```

### Logs and checkpoints

- There are **no checkpoints and no log files** — this is an inference-only project. The console is the runtime log.
- The only saved diagnostics are: the evaluate/per-layer JSONs above and the ORT profiler JSON (`Output profiler - <model> ... .json`) — a native ORT performance trace (Chrome-trace-style events: `cat`, `name`, `ts`, `dur`, …). It records operator latency, not model data.
