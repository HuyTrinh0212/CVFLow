# Repository Guidelines

**Project:** XAIP — AI Inference Pipeline (ONNX Runtime)
**Root:** `/home/bht/CODE/xaip` — **Python:** 3.12 | **Runtime:** ONNX Runtime (CPU/CUDA) | **Config:** YAML
**Docs:** README.md = intro · GUIDELINE.md = user run guide (env/config/run/output schemas) · this file = agent contract (mandatory rules).

XAIP = modular CV inference framework: 6 model families × 3 tasks (detection/classification/HTR) × 4 modes, quantization-aware comparison. Details → README.md.

## Architecture & Data Flow

### Pipeline Dispatch — `AI_Script/core/factory.py`

```python
if self.inference_mode:  return Pipeline_Tracking if tracking_option else Pipeline_Inference
elif self.debug_mode:    return Pipeline_Debug
elif self.evaluate_mode: return Pipeline_Evaluate
else: raise ValueError("Invalid pipeline")
```

| Pipeline | Flags | Input | Output |
|---|---|---|---|
| `Pipeline_Inference` | `inference_mode=true` | image / folder / video / `.npy` | annotated `.jpg` / `<model> video_detection <ts>.mkv` |
| `Pipeline_Tracking` | + `tracking_option=true` | video + detection model | `<model> video_tracking <ts>.mkv` (ByteTrack IDs) |
| `Pipeline_Debug` | `debug_mode=true` | single image only | `.jpg` + `intermediate_outputs/NNNN__<layer>.json` + profiler JSON |
| `Pipeline_Evaluate` | `evaluate_mode=true` | folder with `images/` + `labels/` | `evaluate - <model> - <precision> - CPU - <ts>.json` (`run_target` hard-coded `CPU`) |

All 4 mode keys (`inference_mode`, `tracking_option`, `debug_mode`, `evaluate_mode`) must be present; exactly **1** main mode `true`.

### Registry / Factory pattern (identical in models / preprocess / postprocess / evaluate)

```python
class XRegistry:                # decorator-based; auto-discovered via __init__ imports
    @classmethod def register(cls, name): ...
    @classmethod def get(cls, name): ...

class XFactory:                 # stateless
    @staticmethod def create(config, **kwargs): return XRegistry.get(config["model_name"])(config, **kwargs)

@XRegistry.register("name")
class ConcreteX(BaseX):
    def __init__(self, config): ...
    def __call__(self, input): ...
```

Domains: `ModelRegistry`/`BaseModel` → `models/` · `Preprocess*`/`BasePreprocess` → `preprocess/` · `PostProcessor*`/`BasePostprocess` → `postprocess/` · `Evaluate*`/`BaseEvaluate` → `evaluate/`.

### Data flow

- All modes: `check_file(input)` → preprocessor → `np.ndarray [1,C,H,W]` → model → `List[np.ndarray]`; video = per-frame loop + `VideoWriter` (FFV1).
- Tracking: adapter → conf threshold → NMS → `ByteTrack.update()` → unletterbox → draw ID.
- Debug: `_dump_model()` rewrites the ONNX graph with Identity nodes on every tensor → dumps all layer outputs as JSON `{layer_name, index, dtype, shape, values}`; temp `dump_model.onnx` is removed after the run.
- Evaluate: iterate images → `benchmark_single(pred, img_path, label_path)` → `benchmark_toltal()` aggregate → final JSON.

## Key Directories

| Path | Purpose |
|------|---------|
| `AI_Script/core/` | pipelines + `factory_pipeline` dispatch |
| `AI_Script/{models,preprocess,postprocess,evaluate}/` | registry/factory domains, `factory_*.py` + `base_*.py` |
| `AI_Script/tracker/ByteTrack/` | Kalman + Hungarian matching (detection only) |
| `AI_Script/configs/` | internal per-model defaults + `datasets.json` label maps (reference original machine's weight dirs) |
| `AI_Script/Functions/` | one-off QA/utility scripts |
| `AI_Script/inputs/dataset/` | in-repo sample datasets: `Dataset_Number` (cls 47), `Dataset_HTR`, `COCO_Val_2017` (det) |
| `config_templates/` | 7 schema-only templates (pure `/path/to/...`) |
| `test_configs/` `eval_configs/` `debug_configs/` | ready example configs — **absolute paths from the original machine, edit before reuse** |

Run any pipeline: `python main.py <config.yaml>` (env setup, full parameters, per-mode examples → GUIDELINE.md § 1–3; compare two runs → § 4).

## Code Conventions

- Naming: PascalCase classes · snake_case funcs/keys/files · lowercase registry keys — incl. the kept typo `benchmark_toltal` (renaming = breaking change).
- Base contracts: `BaseModel(config, debug_mode)` → `load_model()` selects weight by `precision_format`/`weight_path` → `__call__(np.ndarray) → List[np.ndarray]`; `BasePreprocess.__call__(image*) → np.ndarray`; `BasePostprocess.__call__(model_output, raw_input=None, save=True)`; `BaseEvaluate.benchmark_single(pred, img_path, label_path)` + `benchmark_toltal(dict)`.
- Error handling: `output_path` validated in `main.py`, `factory_pipeline.__init__`, `BaseModel._session_option`, every pipeline `__init__` — fail-fast, descriptive exceptions, no silent weight fallbacks.
- Config flow: `AI_Script/configs/<model>.yaml` is the single source of truth; `output_path`/`display` injected, then CLI overwrites; unknown YAML keys forwarded via `extras` (this is how `input_dtype`, `graph_optimization_level`, `resize_mode` reach the preprocessor/model).
- Tensor/image conventions: **BGR** · **NCHW** `[1,C,H,W]` · per-model `mean`/`std` in internal config (**inert for YOLO detection** — RGB `[0,1]` ÷255 only, mean/std math commented out in `preprocess/detection/yolo.py`) · detection boxes **xyxy** after adapter + unletterbox.

## Hardware Constraints (verified)

- GPU needs `onnxruntime-gpu` + CUDA 11.8; silently falls back to CPU if unavailable.
- FP16: most exports FP16 IO (auto-cast); exception `Yolov5s_DOTA_Satellite_FP16.onnx` keeps FP32 IO → `input_dtype: "fp32"`.
- BF16: `*_BF16_with_cast.onnx` has FP32 IO (cast in graph) — no override; the 4 fine-tuned YOLOv5 export plain `*_BF16.onnx` **also FP32 IO** — no override.
- INT8 (`precision_format: "int8"`, picked by file name): `*_Mixed_FP32_INT8.onnx` (static QDQ) vs `*_INT8_dynamic_W8A8.onnx` (dynamic W8A8); fine-tuned YOLOv5 dynamic file is `*_INT8_dynamic.onnx` (same behavior).
- Fine-tuned YOLOv5 exports have fixed 640×640 ONNX input → `target_size` must be `[640, 640]` (other sizes fail at session level). Preprocess validated per model (2026-11 sweep, single-image hand-GT): Smoke (square-stretch-trained) & Seatbelt → **stretch** (Seatbelt: detects 3/3 objects vs 1/3 with letterbox); Helmet → **letterbox** (better post-NMS mAP@[.5:.95] 0.299 vs 0.202; detect-strength of the dark driver helmet much better under stretch — retest with more labelled images before switching). `mean`/`std` stay no-ops; ImageNet mean/std or 0–255 scale inputs collapse the models (mAP → 0).
- ORT session: profiling always on → `Output profiler - <model> ....json` in `output_path`; `graph_optimization_level` default `all`, CRNN FP16 needs `"extended"` (segfault under `all`).

## Testing & QA

- **No test suite** (no pytest/unittest/tests/). QA = evaluation JSONs + benchmark charts + layer-wise MSE + `AI_Script/Functions/` scripts.
- Verified: 55 inference runs on CPU (11 variants × fp32/fp16/mixed/bf16/w8a8 — original machine); fine-tuned YOLOv5 configs in `test_configs/yolov5_{helmet,seatbelt,smoke}_fp32.yaml` (preprocess-validated 2026-11; LPR config no longer present; old per-precision test_configs removed in commit `324f7ec`).
- Evaluate label formats: det `class_id xc yc w h` (normalized) · cls single integer · HTR plain text.

### Known Gaps (agent awareness)

1. No automated tests — consider pytest for `Compare_Output_Intermediate`, dataset-folder sync, metric math.
2. HTR `benchmark_toltal` aggregates exact-match accuracy only; `_cer`/`_wer` helpers exist but are not aggregated.
3. `Draw_chart.py` has hard-coded paths.
4. Internal weight defaults point to the original machine — `weight_path` must always be set by the user.
5. Detection evaluate `_adapter_boxes` calls `unletterbox` **without** `resize_mode` → `evaluate_mode` reverse-maps stretch runs with letterbox math (`draw_box` passes `resize_mode` correctly, so inference/tracking visuals are fine). Single-image preprocess sweeps must wrap the eval math themselves (artifacts: `~/Downloads/xaip_preprocess_experiments/`).

Minimal configs by mode + output JSON schemas → GUIDELINE.md § 2 & § 4.
