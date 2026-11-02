# XAIP - AI Inference Pipeline

## Quick Start

### Option 1: Using run.sh (Recommended)

```bash
# Run with config file (auto-creates conda env 'xaip' if missing)
./run.sh config.yaml
```

The script will:
- Check for conda environment `xaip`, create from `environment.yml` if not found
- Activate the environment
- Run `python3 main.py config.yaml`

### Option 2: Manual Setup

#### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate xaip
```

#### 2. Install Dependencies

```bash
pip install onnx onnxruntime opencv-python numpy pyyaml torch torchvision pillow scikit-learn matplotlib pandas
```

For GPU support (CUDA):
```bash
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Run Main Script

```bash
python main.py config.yaml
# Or start from a schema-only template:
python main.py config_templates/yolov5.yaml
```

---

## Project Structure

```
xaip/
├── main.py                    # Entry point (accepts config.yaml as argument)
├── run.sh                     # Auto-setup script (conda env + run)
├── environment.yml            # Conda environment (python=3.12)
├── config.yaml                # Example pipeline configuration
├── config_templates/          # Schema-only templates per model (/path/to/...)
└── AI_Script/                 # Core pipeline
    ├── core/                  # Pipelines (inference, tracking, debug, evaluate, benchmark)
    ├── configs/               # Internal model configs + datasets.json (label sets)
    ├── models/                # Model implementations (yolov5, detr, lenet5, resnet50, vit, crnn)
    ├── preprocess/            # Preprocessing (YOLO, DETR, CRNN, Classification)
    ├── postprocess/           # Postprocessing (draw boxes/labels, decode CRNN, etc.)
    ├── tracker/               # ByteTrack implementation (detection models only)
    ├── evaluate/              # Evaluation tasks
    └── Functions/             # Helper scripts (charts, dataset formatting, etc.)
```

---

## Models

| `model_name` | Task | Default `dataset` |
|--------------|------|-------------------|
| `yolov5` | object_detection | `coco` (`dota_satellite`/`dota` also available) |
| `detr` | object_detection | `coco` |
| `lenet5` | classification | `number_character` (47 labels) |
| `resnet50` | classification | `eurosat` (10 labels) |
| `vit` | classification | `imagenet` (1000 labels) |
| `crnn` | htr | (none - no `dataset` key) |

> **Note**: `resnet` is a code alias only; configs use `resnet50`. Label sets live in `AI_Script/configs/datasets.json` (`coco 80, dota_satellite 16, dota 15, number_character 47, eurosat 10, imagenet 1000`).

---

## Precision Formats

`precision_format` is the label only; the weight file is selected by `weight_path`:

| Label | Weight family | Input IO | Extra keys needed |
|-------|---------------|----------|--------------------|
| `fp32` | `*_FP32.onnx` | fp32 | — |
| `fp16` | `*_FP16.onnx` | fp16 (auto-cast) | — except the 2 files below |
| `int8` | `*_Mixed_FP32_INT8.onnx` (static quant) or `*_INT8_dynamic_W8A8.onnx` (dynamic quant) | fp32 | — |
| `bf16` | `*_BF16_with_cast.onnx` (cast inside graph) | fp32 | — |

All 7 model variants × 5 labels are verified to run on CPU (35/35 configs in `test_configs/`). Two per-file quirks, already set in the templates:

- **YOLOv5 DOTA** FP16 weight keeps FP32 IO → `input_dtype: "fp32"` (default in `config_templates/yolov5_dota.yaml`)
- **CRNN** FP16 weight needs `graph_optimization_level: "extended"` (default in `config_templates/crnn.yaml`)

---

## Modes (in main.py)

| Mode | Description | Accepted input |
|------|-------------|----------------|
| `inference_mode=True` | Standard inference | image, video, folder |
| `tracking_option=True` (+ inference) | ByteTrack object tracking (detection models, video only) | video |
| `debug_mode=True` | Profile model + dump intermediate layers | single image only |
| `evaluate_mode=True` | Run evaluation benchmarks (folder must contain `images/` + `labels/`) | folder only |

> **Note**: Only one mode can be active at a time. If multiple flags are `true`, `inference_mode` wins (`factory.py` precedence: inference → debug → evaluate).

---

## Input Types Supported

- Single image: `image_path`, `npy_path`, `numpy_array`
- Folder of images: `folder_path` (evaluate: must contain `images/` + `labels/` subfolders)
- Video file: `video_path`

---

## Outputs (under `output_path/`)

- Inference image: annotated `.jpg`
- Inference/tracking video: `.mkv` (`video_detection` / `video_tracking` with IDs)
- Debug: annotated `.jpg` + profiler `.json` + `intermediate_outputs/` (per-layer jsons)
- Evaluate: `evaluate - <model> - <precision> - CPU - <timestamp>.json` (+ `Benchmark_Chart_<timestamp>/` for `benchmark_dashboard`)

---

## Requirements

- Python 3.12
- ONNX + ONNX Runtime (CPU or GPU)
- OpenCV, NumPy, PyYAML, Pillow
- PyTorch + torchvision (ByteTrack tracker)
- scikit-learn (evaluation metrics)
- matplotlib + pandas (benchmark charts)

---

## Config.yaml Options

| Key | Description | Default |
|-----|-------------|---------|
| `input_path` | Path to input (image, folder, video, npy; mode-dependent, see Modes) | Required |
| `weight_path` | Path to ONNX model weights (overrides internal weight; `precision_format` then acts as label only) | Required |
| `output_path` | Output directory for results | **Required** (no default) |
| `model_name` | Model name (yolov5, detr, lenet5, resnet50, vit, crnn) | Required |
| `precision_format` | Precision label: `fp32` \| `fp16` \| `int8` (Mixed/W8A8) \| `bf16` — see Precision Formats | Required |
| `input_dtype` | Force model input tensor dtype (`fp32` \| `fp16`); default derived from precision label | Auto |
| `graph_optimization_level` | ORT graph optimization: `disable` \| `basic` \| `extended` \| `all` | `all` |
| `debug_dump_initializers` | Debug mode only: also dump weight/bias initializers as JSON layers | `false` |
| `dataset` | Label set override (optional; defaults per model, see Models) | Per-model default |
| `target.device` | Inference device | `cpu` (`cpu` \| `gpu`) |
| `inference_mode` | Standard inference | `true` |
| `debug_mode` | Profile model performance | `false` |
| `evaluate_mode` | Run evaluation benchmarks | `false` |
| `tracking_option` | Enable ByteTrack tracking (detection + video only) | `false` |
| `display` | Show real-time window (`q` to interrupt); `false` = headless save only | `false` |

> **Note**: Only one mode can be active at a time (inference/debug/evaluate).
> `output_path` is **required** (no default). The directory is created automatically if it doesn't exist. If missing/empty, the run fails with `Error: 'output_path' is required in config.yaml`.
> Benchmark charts (`benchmark_dashboard`) are also saved under `output_path/Benchmark_Chart_{timestamp}/`.
