# XAIP — Supported Models

**Summary**: The registered model families with their tasks, default datasets, target sizes and preprocess/postprocess routing — counts and keys re-verifiable from the registries.
**Tags**: #models #project
**Created**: 2026-11-09
**Last Updated**: 2026-11-11

---

## Live count

```bash
grep -rn "register(" AI_Script/models/
```

Verified 2026-11-09: **6 model families registered under 7 keys** (single alias), **10 label sets** in `AI_Script/configs/datasets.json`. The internal-config set mirrors it: 6 YAMLs under `AI_Script/configs/`.

## Model matrix

| Registry key | Family | Class file | Task (`task` key) | Default dataset | `target_size` | Preprocess keyed by | Postprocess (`default_postprocess`) |
|---|---|---|---|---|---|---|---|
| `yolov5` | YOLOv5 | `AI_Script/models/detection/yolov5.py` | `object_detection` | `coco` | `[640, 640]` | `yolo`, `yolov5` | `draw_box` |
| `detr` | RT-DETR | `AI_Script/models/detection/detr.py` | `object_detection` | `coco` | `[640, 640]` | `detr` | `draw_box` |
| `lenet5` | LeNet5 | `AI_Script/models/classification/lenet5.py` | `classification` | `number_character` (47 labels) | `[32, 32]` | `lenet5` | `draw_label` |
| `resnet50` / `resnet` | ResNet-50 (+ one alias) | `AI_Script/models/classification/resnet50.py` | `classification` | `eurosat` (10 labels) | `[224, 224]` | `resnet50` (+ `resnet`, `resnet16` preprocess keys) | `draw_label` |
| `vit` | Vision Transformer | `AI_Script/models/classification/vit.py` | `classification` | `imagenet` (1000 labels) | `[224, 224]` | `vit` | `draw_label` |
| `crnn` | CRNN (handwritten text) | `AI_Script/models/HTR/crnn.py` | `htr` | — (no label map; Synth90k charset) | `[100, 32]` (W×H) | `crnn` | `draw_label` |

Notes:

- Model classes are thin wrappers: `session.run` only; all task interpretation happens in the postprocess/evaluate layer. Detection models additionally return nothing heavier than the raw list of output arrays (`List[np.ndarray]` contract).
- Preprocess classes live in `AI_Script/preprocess/…` and are keyed by **model name** (not by task) — see `AI_Script/preprocess/factory_preprocess.py`. The YOLO preprocessor class (`AI_Script/preprocess/detection/yolo.py`) is registered under two keys, `yolo` and `yolov5`; `yolov5` is the key resolved by `model_name`, and the bare `yolo` key is a spare alias no current config uses (`resnet.py` is the other multi-key file: `resnet`, `resnet50`, `resnet16`).
- `resnet16` exists only as a **preprocess** registry key (`AI_Script/preprocess/classification/resnet.py`); there is no `resnet16` model class or internal config.
- `transpose_output` (config, default `False`) handles YOLOv5 exports laid out as `[batch, 5+nc, N]` instead of `[batch, N, 5+nc]`, but it is read **only** by the `draw_box` postprocessor (`AI_Script/postprocess/postprocessor/draw_box/draw_box.py`), which passes it into the detection adapter (`AI_Script/postprocess/Functions/Adapter_Detection.py`). The tracking pipeline (`AI_Script/core/pipeline_tracking.py`) and the evaluate route (`AI_Script/evaluate/tasks/detection.py`) build their own `Adapter` without this key, so a transposed-layout export is not supported there.

## Label sets (`AI_Script/configs/datasets.json`) — verified counts

| Key | Labels | Used by (default) |
|---|---|---|
| `coco` | 80 | `yolov5`, `detr` |
| `dota` | 15 | no in-repo config selects it — legacy label set kept in `AI_Script/configs/datasets.json`; the DOTA route in-repo uses `dota_satellite` (see `config_templates/templates/yolov5_dota.yaml`) |
| `dota_satellite` | 16 | `yolov5` (DOTA Satellite variant) |
| `yolo_lpr` | 1 | no in-repo config selects it (legacy; the shipped fine-tuned configs in `config_templates/inference/` are helmet / seatbelt / smoke only) |
| `yolo_helmet` | 2 | fine-tuned YOLOv5 (helmet) |
| `yolo_seatbelt` | 4 | fine-tuned YOLOv5 (seatbelt) |
| `yolo_smoke` | 1 | fine-tuned YOLOv5 (smoking) |
| `number_character` | 47 | `lenet5` |
| `eurosat` | 10 | `resnet50` |
| `imagenet` | 1000 | `vit` |

Label text → drawn annotation mapping happens in the postprocess layer through `datasets.json[dataset][str(class_id)]` (`AI_Script/postprocess/postprocessor/get_labels/get_labels.py` for classification; same lookup in `draw_box.py` for detection).

## Weights & precision handling

- Selection (verified in `AI_Script/models/base_model.py` → `load_model()`): explicit `weight_path` in the user config wins; otherwise `precision_format` maps onto one of the four `weight_path_fp32/_fp16/_int8/_bf16` keys of the internal model config. Unknown `precision_format` → `FileNotFoundError`.
- **No ONNX weight is tracked in this repository**: `AI_Script/pretrained_models/` — the directory most `weight_path_fp32/_fp16/_int8` keys in `AI_Script/configs/*.yaml` point into — does not exist in the repo, and the remaining keys (all `weight_path_bf16`, plus the whole `detr.yaml` / `vit.yaml` files) point to `/home/bht/BHT/AI/Weights/…` on the machine these configs were written on. Treat every `weight_path*` value in `AI_Script/configs/` as a pointer: set `weight_path` in the user config to your own files. The fine-tuned YOLOv5 variants additionally have no internal config — their ready-to-run configs in `config_templates/inference/` use `/path/to/...` placeholders. Config templates under `config_templates/templates/` use `/path/to/...` placeholders on purpose.
- Which ONNX export file corresponds to which precision label is a **weight-path naming agreement**, not code logic: the code loads whatever path the config points to. The current agreement (annotations in `config_templates/templates/*.yaml`): static-QDQ INT8 files vs dynamic-W8A8 INT8 files both load via `precision_format: int8`, and the static-QDQ filename is **per model** — read `weight_path_int8` in `AI_Script/configs/<model>.yaml` (detr/vit `*_Mixed_FP32_INT8`, resnet50 `*_Q0_QInt8`, lenet5 `*_int8_QO_QInt8`, yolov5/crnn `*_int8`); BF16-with-cast exports keep FP32 IO; most FP16 exports have FP16 IO (auto-cast, see [[project/features|Features]] #9), with the DOTA-Satellite FP16 export the known FP32-IO exception needing `input_dtype: "fp32"`. Full detail → [[project/precision-quantization|Precision & Quantization]].
- Fine-tuned YOLOv5 exports are reported to accept only fixed 640×640 (validated observation — is not enforced by code: the session itself will fail on other sizes); `target_size: [640, 640]` is the internal default for `yolov5`.

## Per-model preprocess specifics (verified in each file)

| Model | Preprocess behavior |
|---|---|
| `yolo`/`yolov5` | BGR→RGB, letterbox (114 pad) **or** `stretch` via `resize_mode`, ÷255; `mean`/`std` keys accepted but **not applied** (math commented out) → `AI_Script/preprocess/detection/yolo.py` |
| `detr` | BGR→RGB, letterbox only (no `stretch`), ÷255 → `AI_Script/preprocess/detection/detr.py` |
| `lenet5` | BGR→gray, resize 32×32, ÷255, `(x-mean)/std` with config defaults 0/1 → `AI_Script/preprocess/classification/lenet5.py` |
| `resnet50` | resize, BGR→RGB, ÷255, **hard-coded EuroSAT statistics** (not read from config) → `AI_Script/preprocess/classification/resnet.py` |
| `vit` | PIL pipeline; square-resize with `(x-0.5)/0.5` normally, torchvision-style (resize-256 + center-crop + ImageNet stats) when `"int8"` appears in `precision_format` → `AI_Script/preprocess/classification/vit.py` |
| `crnn` | BGR→gray, resize `100×32` (W×H), ÷255, `(x-mean)/std` → `AI_Script/preprocess/HTR/CRNN.py` |

All preprocessors add the batch dim and emit `np.ndarray [1,C,H,W]` (channel count per table above: 3 for detection/ResNet/ViT, 1 for LeNet5/CRNN); the FP16-IO cast is applied afterwards in `BasePreprocessor.__call__`.

## Evaluation registration

Evaluate classes are keyed by the same `task` values that the internal configs set (`AI_Script/evaluate/factory_evaluate.py` routes on `config["task"]`):

| Task key | Evaluate class | File |
|---|---|---|
| `object_detection` | `detection_benchmark` | `AI_Script/evaluate/tasks/detection.py` |
| `classification` | `classification_benchmark` | `AI_Script/evaluate/tasks/classification.py` |
| `htr` | `htr_benchmark` | `AI_Script/evaluate/tasks/HTR.py` |

Metric schemas per task are in [[pipeline/output-schemas|Output Schemas]].

## Related Notes

- [[project/overview|Overview]] — registry/factory pattern explanation
- [[project/features|Features]] — grouped capability counts
- [[project/precision-quantization|Precision & Quantization]] — weight-path mapping, IO casting, INT8/BF16 variants, per-model validated preprocess modes
- [[pipeline/mode-flows|Mode Flows]] — how a model ends up called in each mode
- [[pipeline/output-schemas|Output Schemas]] — artifacts per model-task × mode
