# XAIP - AI Inference Pipeline

Modular Python framework for running, debugging and evaluating computer vision models via **ONNX Runtime** (CPU / CUDA). Six model families across three tasks, four operating modes, and quantization-aware comparison tooling.

> **Setup · Config · Running** → [GUIDELINE.md](GUIDELINE.md)
> **Architecture & dev conventions** → [AGENTS.md](AGENTS.md)
> **Knowledge Base (Source of Truth)** → [wiki/](wiki/index.md) — The `wiki/` directory is the single source-of-truth: always prioritize reading and exploring `wiki/` for every task, problem, or situation.

## Overview

**Tasks & models** (`model_name` → default `dataset`):

| `model_name` | Task | Default `dataset` |
|--------------|------|-------------------|
| `yolov5` | object_detection | `coco` (also `dota`, `dota_satellite` + 4 fine-tuned variants: LPR, helmet, seatbelt, smoke) |
| `detr` | object_detection | `coco` |
| `lenet5` | classification | `number_character` (47 labels) |
| `resnet50` | classification | `eurosat` (10 labels) |
| `vit` | classification | `imagenet` (1000 labels) |
| `crnn` | htr | — |

Label sets live in `AI_Script/configs/datasets.json`. Run configs and sample data live in `config_templates/{templates,inference,eval,debug}/` and `AI_Script/inputs/`.

**Modes** (exactly one active at a time):

| Mode | Flags | Input | Output |
|------|-------|-------|--------|
| Inference | `inference_mode=true` | image / folder / video / `.npy` | annotated `.jpg` / `.mkv` |
| Tracking | `inference_mode=true` + `tracking_option=true` | video (detection models) | tracking `.mkv` with IDs (ByteTrack) |
| Debug | `debug_mode=true` | single image | profiler JSON + `intermediate_outputs/` (per-layer) |
| Evaluate | `evaluate_mode=true` | folder with `images/` + `labels/` | metrics JSON |

**Precision comparison:** `fp32` / `fp16` / `int8` (static QDQ or dynamic W8A8) / `bf16` — the weight is selected per file name via `weight_path`. Python API (`factory_pipeline`) additionally exposes `benchmark_dashboard()` (chart comparison of two eval JSONs) and `compare_intermediate()` (layer-wise MSE between precisions).

**Fine-tuned YOLOv5 (Helmet / Seatbelt / Smoke):** ready-to-run configs in `config_templates/inference/` with per-model validated preprocess — Helmet: `letterbox`; Seatbelt & Smoke: `stretch` (validated sweep, 2026-11). Preprocess is RGB `[0,1]` (÷255); `mean`/`std` config keys are accepted but inert for YOLO — keep defaults. The fine-tuned ONNX exports accept fixed 640×640 input.

## Project Structure

```
xaip/
├── main.py                    # Entry point (accepts config.yaml as argument)
├── run.sh                     # Auto-setup script (conda env + run)
├── check_env.py               # Standalone environment check (GUIDELINE.md)
├── environment.yml            # Conda environment (python=3.12)
├── requirements.txt           # Pip pins (plain venv alternative — GUIDELINE § 1)
├── GUIDELINE.md               # Setup, config, running
├── config.yaml                # Example pipeline configuration
├── config_templates/          # All pipeline configs organized by mode
│   ├── templates/             # 7 schema-only templates (/path/to/...)
│   ├── inference/             # Fine-tuned YOLOv5 ready-to-run (3 configs)
│   ├── eval/                  # Evaluation configs (4 configs)
│   └── debug/                 # Debug configs (7 configs)
└── AI_Script/                 # Core pipeline
    ├── core/                  # Pipelines (inference, tracking, debug, evaluate, benchmark)
    ├── configs/               # Internal model configs + datasets.json (label sets)
    ├── models/                # Model implementations (registry/factory)
    ├── preprocess/            # Preprocessing (YOLO, DETR, CRNN, Classification)
    ├── postprocess/           # Postprocessing (draw boxes/labels, decode CRNN, etc.)
    ├── tracker/               # ByteTrack implementation (detection models only)
    ├── evaluate/              # Evaluation tasks
    ├── inputs/                # Sample datasets & test media
    └── Functions/             # Helper scripts (charts, dataset formatting, etc.)
```
