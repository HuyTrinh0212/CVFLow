# GUIDELINE — Run Guide

> Setup → Config → Run → Outputs. All paths in examples are placeholders (`/path/to/...`) — point them to **your** locations; the project has no preferred storage layout.

## 0. Quick Start

Goal: one input → one annotated output, before reading any parameter table. You need exactly two things of your own: **one ONNX weight** and **one input image** (the repo ships sample inputs in `AI_Script/inputs/` but **no weights** — see § 1 verification + § 2 config sources).

### Step 1 — copy the minimal config

Copy to e.g. `first_run.yaml` (any location) and edit **only** the three keys marked `# EDIT` — everything else has validated defaults (inference on an image, CPU, fp32):

```yaml
input_path: "/path/to/your/image.jpg"        # EDIT — your image (repo samples: AI_Script/inputs/images/)
weight_path: "/path/to/your/model.onnx"      # EDIT — the ONNX file you want to run
output_path: "/path/to/your/output"          # EDIT — any directory; created if missing

model_name: "yolov5"            # match your weight: yolov5 | detr | lenet5 | resnet50 | vit | crnn
precision_format: "fp32"        # keep fp32 for the first run
dataset: "coco"                 # label set matching model_name (valid values in § 2)

target:
  device: "cpu"

inference_mode: true
tracking_option: false
debug_mode: false
evaluate_mode: false
```

If you start from a shipped config instead of this copy — e.g. `config_templates/templates/<model_name>.yaml` — the same three keys are the only edits.

### Step 2 — run

```bash
./run.sh first_run.yaml          # conda path: creates env 'xaip' if missing, then runs
# no conda: follow the venv path in § 1, then python main.py first_run.yaml
```

### Step 3 — check the result

Annotated image in `output_path` (naming per mode → § 4 artifacts table). If it fails: `python check_env.py` (§ 1) verifies the environment itself.

### First-run rules of thumb

- Start with **fp32** and a **single image**; add other precisions / modes only after one success (non-fp32 weights come with per-file exceptions → § 2 Advanced notes).
- **No other parameter is required.** Optional parameters and advanced keys (`input_dtype`, `resize_mode`, `mean`, `std`, ...) can be omitted entirely — shipped defaults apply.
- A common failure is `model_name` not matching the weight file; the next most common is input type not matching the mode (see the `input_path` row in § 2).

## 1. Environment Setup

> Minimal path: `./run.sh config.yaml` creates the conda env if missing and runs (§ 0). Read this section only if that fails — no conda (venv below) or GPU setup.

### Python version

**Python 3.12** (`environment.yml` conda · header comment in `requirements.txt` for venv).

### Environment creation & activation — conda (project default)

```bash
./run.sh config.yaml            # auto-creates the conda env 'xaip' from environment.yml if missing, then runs

# or manually
conda env create -f environment.yml
conda activate xaip
```

### Alternative — plain venv (no conda)

`requirements.txt` mirrors the exact `environment.yml` pins. Needs any local Python 3.12 (`python3.12` on PATH — pyenv, distro package, deadsnakes). Two steps: the first buys the light CPU torch build, the second fills in the rest:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # ~200 MB, CPU wheels
pip install -r requirements.txt                                                  # +12 packages, torch already satisfied
python check_env.py                                                              # verify — see below
```

Step 1 matters on Linux: the plain PyPI `torch` wheel pulls a multi-GB CUDA-13 stack (`nvidia-*` wheels) that a CPU-only run does not use at all. `torch==2.13.0+cpu` (and `+cu118` for the GPU path) still satisfies the `torch==2.13.0` pin (PEP 440 local-version rule) — verified: `check_env.py` reports READY on such a venv.

### Dependency installation

`environment.yml` (conda) and `requirements.txt` (venv) pin the same set: `onnx`, `onnxruntime`, `opencv-python`, `numpy`, `PyYAML`, `Pillow`, `scikit-learn`, `matplotlib`, `pandas`, `torch`, `torchvision`, `lap`, `cython_bbox` — CPU setup needs nothing extra.

GPU runs additionally require the CUDA-capable packages (CUDA 11.8) — same command for conda and venv:

```bash
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verify installation

Standalone script at the repo root — verifies the active environment (conda or venv alike) with no config or weights needed: Python version, imports + exact versions of every dependency pinned in `environment.yml`, ORT execution providers, and a smoke-run of a tiny synthetic ONNX graph. Exit code `0` only when usable:

```bash
conda activate xaip             # or: source .venv/bin/activate
python check_env.py             # CPU setup
python check_env.py --gpu       # additionally requires a working CUDAExecutionProvider
```

GPU setups must show `CUDAExecutionProvider` in the script's provider list; if missing (or `--gpu` exits non-zero), the onnxruntime-gpu / CUDA 11.8 install is not usable and every run silently stays on CPU despite `target.device: "gpu"` (see AGENTS.md "Hardware Constraints").

End-to-end verification with a real model = running § 3 on the § 0 Quick Start config (`weight_path` pointing to an actual ONNX file — weights are not shipped in-repo).

## 2. Configuration

Each run = one YAML file, at any location, passed to `main.py`. The standard first run is the § 0 Quick Start config; reuse of a shipped config is identical in shape:

> **Using a shipped config: copy it, edit only the keys pointing at your disk — `input_path`, `output_path`, and the weight key(s) it uses (`weight_path`, or the four `weight_path_<precision>` entries in the fine-tuned configs) — and leave everything else as-is.** The path-only edit is the whole setup; no repo config runs unedited, because none ship files on your disk.

Config sources, in order of usefulness: the four per-mode examples below · schema-only templates per model in `config_templates/templates/<model_name>.yaml` (pure `/path/to/...` placeholders — fill in your paths) · ready-to-run configs organized by mode in `config_templates/{inference,eval,debug}/` (all paths are placeholders — edit before reuse; fine-tuned YOLOv5 run configs in `config_templates/inference/` — Advanced notes below apply to them).

### Required parameters

| Parameter | Possible values | Notes |
|-----------|----------------|-------|
| `input_path` | image (`.png`/`.jpg`/`.jpeg`) \| image folder \| video (`.mp4`/`.avi`/`.mov`/`.mkv`) \| `.npy` | input type must match the mode: inference = any listed · tracking = video · debug = single image / `.npy` · evaluate = folder with `images/` + `labels/` (artifacts § 4) |
| `weight_path` | `/path/to/your/models/<name>.onnx` | always set explicitly (see weight rules below) |
| `output_path` | `/path/to/your/output/` (directory) | auto-created if missing |
| `model_name` | `yolov5` \| `detr` \| `lenet5` \| `resnet50` \| `vit` \| `crnn` | `resnet` is only an alias in code — use `resnet50` in configs |
| `precision_format` | `fp32` \| `fp16` \| `int8` \| `bf16` | a label; the file actually loaded is chosen **by file name** in `weight_path` |
| `dataset` | `coco` \| `dota` \| `dota_satellite` \| `number_character` \| `eurosat` \| `imagenet` \| `yolo_lpr` \| `yolo_helmet` \| `yolo_seatbelt` \| `yolo_smoke` | label mapping only (`AI_Script/configs/datasets.json`); per-model default; not needed by `crnn` |
| `target.device` | `cpu` \| `gpu` | `gpu` requires onnxruntime-gpu (§ 1) |
| mode flags | `inference_mode`, `tracking_option`, `debug_mode`, `evaluate_mode` → `true`\|`false` | all 4 keys must be present; exactly **1** main mode `true` (§ 4) |

Two rules matter for every run — the rest is reference detail:

- **Treat `weight_path` as always set** — unset, it silently falls back to the original machine's directories (internal defaults in `AI_Script/configs/*.yaml`), which likely don't exist on your disk. Weight selection is **by file name**: `*_FP32.onnx` · `*_FP16.onnx` · `*_Mixed_FP32_INT8.onnx` / `*_INT8_dynamic_W8A8.onnx` · `*_BF16_with_cast.onnx`. Weights may live anywhere.
- **Mode keys:** all 4 must be present, exactly **1** main mode `true`. If several are `true`, precedence is `inference_mode` → `debug_mode` → `evaluate_mode`. The input type must match the mode (`input_path` row above).

Reference detail — no need to touch on a first run:

- Keys not in the tables above are forwarded verbatim into the pipeline → they can override model-internal values (`mean`, `std`, `target_size`, `transpose_output`, `weight_path_*`, `default_postprocess`, `task`). **Leave them at shipped/default values unless following a documented exception** (§ 2 Advanced notes): e.g. `mean`/`std` are **inert for YOLO detection preprocessing** — the applied pipeline is RGB `[0,1]` (÷255) and the mean/std math is commented out in `AI_Script/preprocess/detection/yolo.py`; keep them at `[0,0,0]`/`[1,1,1]` (non-default mean/std or 0–255 scale inputs collapse fine-tuned models, mAP → 0).

### Optional parameters

All rows below have working defaults — **this table can be skipped entirely on a first run**; the § 0 config and the four examples need none of these keys:

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

### Advanced notes — precision quirks & fine-tuned YOLOv5 (skip on first read)

Everything below applies only to specific weights / precisions — the § 0 config and the four examples need none of it.

Known precision quirks (3 exceptions; full background incl. BF16/INT8 export details → AGENTS.md "Hardware Constraints"):

- YOLOv5 DOTA FP16 keeps FP32 IO → `input_dtype: "fp32"`
- CRNN FP16 (LSTM export) segfaults under full optimization → `graph_optimization_level: "extended"`
- YOLOv5 fine-tuned Smoke & Seatbelt → `resize_mode: "stretch"` (validated); Helmet keeps `letterbox` — measured per model, rationale documented in each config's header

Fine-tuned YOLOv5 quick reference — per-model validated configs in `config_templates/inference/` (edit `input_path` / `output_path` / the `weight_path_<precision>` entries — they use `/path/to/...` placeholders; switch precision by editing `precision_format` only), preprocess validated per model (single-image sweep: letterbox vs stretch vs normalization variants; artifacts + method → `~/Downloads/xaip_preprocess_experiments/`):

| Config | `dataset` | `resize_mode` | Weights directory |
|--------|-----------|---------------|-------------------|
| `config_templates/inference/yolov5_helmet.yaml` | `yolo_helmet` | `letterbox` | `YOLO_HelmetDetection/` |
| `config_templates/inference/yolov5_seatbelt.yaml` | `yolo_seatbelt` | `stretch` | `YOLO_SeatbeltDetection/` |
| `config_templates/inference/yolov5_smoke.yaml` | `yolo_smoke` | `stretch` | `YOLO_SmokeDetection/` |

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
- Profiling is only enabled in debug mode (`debug_mode: true`), where an `Output profiler - <model> ... .json` is written to `output_path`. Inference/tracking/evaluate sessions run without profiler output.

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
  "shape": "(1, 64, 112, 112)",
  "values": [ "/* nested list matching shape */" ]
}
```
