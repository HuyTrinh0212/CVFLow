# XAIP — Precision & Quantization

**Summary**: Complete reference for precision labels, weight-path mapping, IO casting rules, INT8/BF16 variants, and per-model validated behaviors — the quantization-aware comparison backbone.
**Tags**: #precision #quantization #project
**Created**: 2026-11-09
**Last Updated**: 2026-11-11

---

## Precision labels & weight-path mapping

**Four precision labels** are recognized in config (`AI_Script/models/base_model.py` → `load_model()`):

| Label | Internal config key | Typical ONNX export naming |
|---|---|---|
| `fp32` | `weight_path_fp32` | `*_FP32.onnx` |
| `fp16` | `weight_path_fp16` | `*_FP16.onnx` / `*_fp16.onnx` |
| `int8` | `weight_path_int8` | static QDQ: **per-model name** (see the INT8 section below — `*_Mixed_FP32_INT8` is only the detr/vit convention) · dynamic W8A8: `*_INT8_dynamic_W8A8.onnx` / fine-tuned `*_INT8_dynamic.onnx` |
| `bf16` | `weight_path_bf16` | `*_BF16_with_cast.onnx` (FP32 IO, cast in graph) or fine-tuned YOLOv5 `*_BF16.onnx` (also FP32 IO) |

**Selection logic** (verified in `BaseModel.load_model()`):
1. Explicit `weight_path` in user config **always wins** (bypasses precision mapping)
2. Otherwise `precision_format` selects one of the four `weight_path_*` keys from the internal model config (`AI_Script/configs/<model>.yaml`)
3. Unknown `precision_format` → `FileNotFoundError`

> The internal configs ship paths from the project's original machine. Using them elsewhere requires setting `weight_path` (or editing/overriding). Config templates under `config_templates/templates/` use `/path/to/...` placeholders on purpose.

---

## FP16 IO auto-cast & override

**Default behavior** (`AI_Script/preprocess/base_preprocess.py` → `__call__`):
- When `precision_format == "fp16"`, the preprocessor **auto-casts** the preprocessed tensor to `float16` before returning it
- This matches FP16 weights exported with **FP16 input/output** (e.g., YOLOv5 COCO, RT-DETR, ViT FP16 exports)

**Override** via config key `input_dtype`:
- `input_dtype: "fp32"` — keeps input as FP32 even for `precision_format: fp16`
- `input_dtype: "fp16"` — forces FP16 cast (default when `precision_format: fp16`)
- Default when `precision_format != "fp16"` is `"fp32"`

**Known exception**: YOLOv5 DOTA Satellite FP16 export keeps **FP32 IO** → requires `input_dtype: "fp32"` (see `config_templates/templates/yolov5_dota.yaml`). Without it, the auto-cast causes a session dtype mismatch.

---

## BF16 exports — two patterns

| Pattern | IO type | Example | Note |
|---|---|---|---|
| `*_BF16_with_cast.onnx` | FP32 input, FP32 output (cast ops inside graph) | RT-DETR, ViT, CRNN, LeNet5, ResNet50, YOLOv5 COCO | No `input_dtype` override needed |
| Fine-tuned YOLOv5 `*_BF16.onnx` | FP32 input, FP32 output (no cast ops) | Helmet, Seatbelt, Smoke (+ legacy LPR: the export-file pattern covers 4 variants, but the LPR config was removed from `config_templates/inference/`) | Also requires no override — auto-cast skipped because `precision_format: bf16` defaults to `input_dtype: fp32` |

The `input_dtype` default logic in `BasePreprocessor.__call__` is:
```python
input_dtype = config.get("input_dtype", "fp16" if precision_format == 'fp16' else "fp32")
```
So `bf16` → `"fp32"` by default, matching both export patterns above.

---

## INT8 — two quantization flavors (same label)

Both load via `precision_format: int8` → `weight_path_int8`:

| Flavor | Typical filename | Quantization type |
|---|---|---|
| Static QDQ | Naming is **per model**, read from `weight_path_int8` in `AI_Script/configs/<model>.yaml`: detr & vit → `*_Mixed_FP32_INT8.onnx` · resnet50 → `*_Q0_QInt8.onnx` · lenet5 → `*_int8_QO_QInt8.onnx` · yolov5 (COCO) & crnn → `*_int8.onnx`. There is no project-wide static-QDQ filename convention. | Per-tensor/per-channel static quantization with QuantizeLinear/DequantizeLinear pairs |
| Dynamic W8A8 | `*_INT8_dynamic_W8A8.onnx` or fine-tuned `*_INT8_dynamic.onnx` | Weights quantized to INT8 (static), activations quantized dynamically at runtime |

The code does not distinguish them — it loads whatever path the config points to. The distinction is a **weight-path naming agreement** documented in `config_templates/templates/*.yaml`.

**Precision-dependent preprocessing (ViT)**: the ViT preprocessor is the only one whose behavior changes with the precision label — when `"int8"` appears in `precision_format`, it switches from the manual path (square resize + `(x-0.5)/0.5` normalization) to torchvision-style preprocessing (resize shortest side to 256 + center-crop + ImageNet mean/std). The `fp32`/`fp16`/`bf16` labels all take the manual path. Verified in `AI_Script/preprocess/classification/vit.py` → `preprocess()` (branch between `_preprocess_manual` and `_preprocess_torchvision`); also flagged in the `config_templates/templates/vit.yaml` precision notes. See [[project/models|Models]] for the full per-preprocessor table.

---

## ORT graph_optimization_level

Config key: `graph_optimization_level` (default `"all"`). Verified in `BaseModel._session_option()`:

| Value | ORT enum |
|---|---|
| `disable` | `ORT_DISABLE_ALL` |
| `basic` | `ORT_ENABLE_BASIC` |
| `extended` | `ORT_ENABLE_EXTENDED` |
| `all` | `ORT_ENABLE_ALL` |

**Known requirement**: CRNN FP16 export segfaults under `ORT_ENABLE_ALL` → must use `graph_optimization_level: "extended"` in config.

---

## Fine-tuned YOLOv5 — fixed 640×640 input constraint

Validated observation (not enforced by code — session fails on other sizes):
- All fine-tuned YOLOv5 exports (Helmet, Seatbelt, Smoke, LPR) accept **only 640×640**
- Internal default for `yolov5` is `target_size: [640, 640]` (see `AI_Script/configs/yolov5.yaml`)
- Other sizes fail at `ort.InferenceSession` creation/invocation level

---

## Per-model validated preprocess resize_mode

Validated by single-image preprocess sweeps with hand-GT (2026-11; sweep artifacts lived on the original machine's Downloads folder — not tracked in this repo):

| Model variant | Recommended `resize_mode` | Evidence |
|---|---|---|
| YOLOv5 Smoke | `stretch` | Square-stretch-trained; mAP@[.5:.95] 0.70 → 0.80 under stretch on the test image, better box quality at strict IoU (letterbox keeps higher raw conf .49 vs .36 but poorer geometry) — see `config_templates/inference/yolov5_smoke.yaml` header |
| YOLOv5 Seatbelt | `stretch` | Detects 3/3 objects vs 1/3 with letterbox |
| YOLOv5 Helmet | `letterbox` | Better post-NMS mAP@[.5:.95] 0.299 vs 0.202; dark driver helmet much stronger under stretch — retest with more labelled images before switching |
| YOLOv5 COCO / DOTA (non-satellite) | `letterbox` (default) | Standard YOLO exports trained with letterbox |
| RT-DETR | `letterbox` (only) | DETR preprocessor is letterbox-only (AI_Script/preprocess/detection/detr.py) |
| YOLOv5 DOTA Satellite | `letterbox` (default) | Follows standard YOLOv5 letterbox default; `resize_mode` configurable via `AI_Script/preprocess/detection/yolo.py` (default `letterbox`) |

**Critical**: `mean`/`std` keys in config are **accepted but not applied** for YOLO preprocessor (math commented out in `AI_Script/preprocess/detection/yolo.py`). ImageNet mean/std or 0–255 scale inputs collapse the models (mAP → 0).

---

## Quick reference: config template annotations

Each `config_templates/templates/*.yaml` contains a "Precision notes" section summarizing the above for that model. Example (YOLOv5 COCO):
```yaml
#   fp32 -> *_FP32.onnx | int8 -> *_Mixed_FP32_INT8.onnx (static QDQ) or *_INT8_dynamic_W8A8.onnx (dynamic) | bf16 -> *_BF16_with_cast.onnx
#   FP16 weight has FP16 input -> input auto-cast to fp16 (no extra key needed)
#   int8/bf16 weights have FP32 input  -> input stays fp32   (no extra key needed)
```

---

## Related Notes

- [[project/overview|Overview]] — config flow: how `precision_format` / `weight_path` reach the model and session
- [[project/models|Models]] — model families, internal configs, preprocess specifics
- [[project/features|Features]] — feature inventory items #8, #9, #10, #11
- [[pipeline/mode-flows|Mode Flows]] — how precision_format reaches the model via config flow
- [[pipeline/debug-internals|Debug Internals]] — ORT profiling + graph_optimization_level in debug mode