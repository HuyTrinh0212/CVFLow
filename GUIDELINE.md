# GUIDELINE — Setup, Config, Running

## 1. Setup

```bash
./run.sh config.yaml          # auto-creates the conda env 'xaip' from environment.yml if it doesn't exist

# or manually
conda env create -f environment.yml
conda activate xaip

# only for GPU runs
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 2. Config

Each run = one YAML file. All available parameters:

| Parameter | Possible values | Default | Notes |
|-----------|----------------|---------|-------|
| `input_path` | path: image (`.png`/`.jpg`/`.jpeg`) \| image folder \| video (`.mp4`/`.avi`/`.mov`/`.mkv`) \| `.npy` | — | valid type per mode (section 3) |
| `weight_path` | path to a `.onnx` file | — | always set explicitly |
| `output_path` | path to a directory | — | required, auto-created if missing |
| `model_name` | `yolov5` \| `detr` \| `lenet5` \| `resnet50` \| `vit` \| `crnn` | — | `resnet` is only a code-level alias; use `resnet50` in configs |
| `precision_format` | `fp32` \| `fp16` \| `int8` \| `bf16` | — | a label; the actual weight file is selected by name (below) |
| `dataset` | `coco` \| `dota` \| `dota_satellite` \| `number_character` \| `eurosat` \| `imagenet` \| `yolo_lpr` \| `yolo_helmet` \| `yolo_seatbelt` \| `yolo_smoke` | per model | label sets in `AI_Script/configs/datasets.json`; not needed by `crnn` |
| `target.device` | `cpu` \| `gpu` | `cpu` | `gpu` requires onnxruntime-gpu (+ CUDA) |
| `inference_mode` | `true` \| `false` | — | key required |
| `tracking_option` | `true` \| `false` | `false` | key required; only takes effect with `inference_mode` + video + detection model |
| `debug_mode` | `true` \| `false` | — | key required |
| `evaluate_mode` | `true` \| `false` | — | key required |
| `display` (alias `display_option`) | `true` \| `false` | `false` | `true`: real-time preview window, `q` to exit |
| `conf_threshold` | float 0–1 | `0.3` | detection only: filter boxes by confidence |
| `iou_threshold` | float 0–1 | `0.5` | detection only: NMS + evaluate |
| `resize_mode` | `letterbox` \| `stretch` | `letterbox` | YOLO preprocess + unletterbox when drawing boxes |
| `input_dtype` | `fp32` \| `fp16` | `fp16` if `precision_format: fp16`, otherwise `fp32` | force input tensor dtype |
| `graph_optimization_level` | `disable` \| `basic` \| `extended` \| `all` | `all` | ORT session optimization |
| `inference_workers` | int ≥ 1 | `1` | tracking only: number of parallel threads per frame |
| `debug_dump_initializers` | `true` \| `false` | `false` | debug only: additionally dump weights/biases to JSON |
| `method` | `greedy` \| `beam_search` \| `prefix_beam_search` | `beam_search` | CRNN only: CTC decoding algorithm |
| `beam_size` | int | `10` | CRNN only: beam decode width |

Rules:

- All 4 mode flags (`inference_mode`, `tracking_option`, `debug_mode`, `evaluate_mode`) must always be present in the YAML; exactly **1** main mode `true` — if several are `true` at once, precedence is `inference_mode` → `debug_mode` → `evaluate_mode`.
- The weight file is selected by **file name** in `weight_path`: `*_FP32.onnx`, `*_FP16.onnx`, `*_Mixed_FP32_INT8.onnx` / `*_INT8_dynamic_W8A8.onnx`, `*_BF16_with_cast.onnx`. Weights live under `/home/bht/BHT/AI/Weights/`.
- Keys not listed in the table are forwarded straight into the pipeline → they can override internal values of the model config (`mean`, `std`, `target_size`, `transpose_output`, `weight_path_*`, `default_postprocess`, `task`).

Reference config templates per model (schema-only in `config_templates/`; replace `/path/to/...` with real paths):

| `model_name` | Template |
|--------------|----------|
| `yolov5` | `config_templates/yolov5.yaml` (COCO) · `config_templates/yolov5_dota.yaml` (DOTA Satellite — already includes `input_dtype: "fp32"`) |
| `detr` | `config_templates/detr.yaml` |
| `lenet5` | `config_templates/lenet5.yaml` |
| `resnet50` | `config_templates/resnet50.yaml` |
| `vit` | `config_templates/vit.yaml` |
| `crnn` | `config_templates/crnn.yaml` (already includes `graph_optimization_level: "extended"`) |

## 3. Running

```bash
python main.py <config.yaml>
```

| Mode (flags) | Accepted input | Output |
|--------------|----------------|--------|
| `inference_mode: true` | image / folder / video / `.npy` | `.jpg` or `.mkv` |
| `inference_mode: true` + `tracking_option: true` | video + detection model | `.mkv` (ByteTrack, with IDs) |
| `debug_mode: true` | single image only | image + profiler JSON + `intermediate_outputs/` |
| `evaluate_mode: true` | folder containing `images/` + `labels/` | `evaluate - <model> - <precision> - CPU - <ts>.json` (device is currently hard-coded to `CPU`) |

Minimal config:

```yaml
input_path: "/path/to/input"
weight_path: "/path/to/model.onnx"
output_path: "/path/to/output"

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

Notes:

- Evaluate: `labels/<image-name>.txt` — detection: `class_id xc yc w h`; classification: a single integer; HTR: plain text. Sample datasets: `AI_Script/inputs/dataset/`.
- 3 exceptions: YOLOv5 DOTA fp16 → `input_dtype: "fp32"`; CRNN fp16 → `graph_optimization_level: "extended"`; YOLOv5 Smoke → `resize_mode: "stretch"`.
