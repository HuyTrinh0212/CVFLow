# XAIP — Supported Features (verified inventory)

**Summary**: Enumerates every supported feature axis with re-verifiable counts — modes, input types, precision labels, tracking, instrumentation, evaluation, comparison tooling and config options.
**Tags**: #features #project
**Created**: 2026-11-09
**Last Updated**: 2026-11-11

---

## Counted capability axes

| Axis | Count | Verified by reading |
|---|---|---|
| Operating modes | **4** (inference · tracking · debug · evaluate) | `AI_Script/core/factory.py` → `_get_pipeline()`; tracking = `inference_mode` + `tracking_option` |
| Model families | **6** (7 registry keys, one alias) | `grep -rn "register(" AI_Script/models/` — details in [[project/models|Models]] |
| Tasks | **3** (`object_detection` · `classification` · `htr`) | `AI_Script/evaluate/factory_evaluate.py` routes by `task`; registry in `AI_Script/evaluate/registry_evaluate.py` |
| Precision labels in config | **4** (`fp32` · `fp16` · `int8` · `bf16`) | `AI_Script/models/base_model.py` → `load_model()` weight-key selection |
| Input source types | **5** (single image · folder · video · `.npy` · in-memory `np.ndarray`) | `AI_Script/core/utils.py` → `check_file()` |
| Label sets shipped | **10** | `AI_Script/configs/datasets.json` top-level keys |
| Offline analysis tools | **2** (`benchmark_dashboard`, `compare_intermediate`) | `AI_Script/core/factory.py` — methods on `factory_pipeline` |

## Feature inventory (what each axis actually does)

1. **Mode dispatch with fail-fast validation** — `output_path` required (no default) and validated in `main.py`, `factory_pipeline.__init__`, the artifact-writing pipelines' `__init__` (`Pipeline_Inference`/`_Tracking`/`_Debug`/`_Evaluate`; `Pipeline_Benchmark` validates only when `output_path` is in its config and otherwise defers to `run()`) and `BaseModel._session_option`; unknown/empty → descriptive `ValueError`/`FileNotFoundError`. `Pipeline_Compare_Intermediate` has no `__init__` and writes no files, so it never validates. `AI_Script/core/factory.py`, `main.py`.
2. **Detection inference with classical postprocess chain** — model-output adapter (per detection model layout) → confidence mask → NMS (`torchvision.ops.nms`, `AI_Script/postprocess/Functions/Boxes_Steps.py`) → box mapping back to original image (`unletterbox`, honors `resize_mode`) → annotated `.jpg` / video. `AI_Script/postprocess/postprocessor/draw_box/draw_box.py`.
3. **ByteTrack multi-object tracking** — per-frame adapter → conf filter → NMS → `BYTETracker.update` → unletterbox → draw track IDs; optional frame-parallel thread pool (`inference_workers`, default `1`, order-preserving). `AI_Script/core/pipeline_tracking.py`, `AI_Script/tracker/ByteTrack/ByteTrack.py` (wrapper with its own hard-coded `ByteTrackArgs`).
4. **Debug instrumentation** — rewrites the ONNX graph so every tensor becomes a graph output (`debug_dump_initializers` optionally includes weights/initializers, default off), saves a temporary `dump_model.onnx`, streams each layer as a JSON file, runs ONNX profiling, then deletes the temp graph. `AI_Script/core/pipeline_debug.py` → `_dump_model()` and `run()`.
5. **Task-specific evaluation** — dataset folder `images/` + `labels/`; per-image metrics plus task-specific aggregation; skeleton JSON identical for all tasks. Metrics per task in [[pipeline/output-schemas|Output Schemas]].
6. **Benchmark dashboard** — compares two evaluate JSONs; outputs 5 comparison PNGs. De-facto classification inputs: the charts read classification-schema keys (`accuracy` / `mcc` / `Per_class` / `Confusion_matrix`), so classification evaluate runs are needed in practice — `pipeline_benchmark.py` itself checks no task. `AI_Script/core/pipeline_benchmark.py` → `run()`.
7. **Layer-wise MSE comparison** — two debug dumps (directory of per-layer JSONs or a legacy monolithic JSON) → `(per_layer_mse: dict, overall_mse: float)`; requires equal layer-key sets. `AI_Script/postprocess/Functions/Compare_Output_Intermediate.py` → `Compare_Ouput_Intermediate` (name typo is intentional).
8. **Precision-aware weight selection** — `precision_format` maps to `weight_path_fp32/fp16/int8/bf16` keys of the internal model config; explicit `weight_path` in the user config always wins. `AI_Script/models/base_model.py` → `load_model()`.
9. **Precision-aware IO casting** — input tensor is cast to `fp16` automatically when `precision_format == "fp16"` (override via `input_dtype: fp32`), so FP32-IO weights still load. `AI_Script/preprocess/base_preprocess.py` → `__call__`.
10. **Configurable ORT session** — `graph_optimization_level` (`disable|basic|extended|all`, default `all`), profiling enabled only in debug mode. `AI_Script/models/base_model.py` → `_session_option()`.
11. **Resize-mode-aware detection preprocess** — `resize_mode: letterbox` (default, 114 border pad) or `stretch` (anisotropic full resize) for the YOLO preprocessor; letterbox-only for DETR; grayscale pipelines for LeNet5/CNN-HTR; OpenCV pipeline for ResNet; PIL pipeline for ViT. `AI_Script/preprocess/**/*.py`.
12. **Headless by default** — `display: false` runs save-only; `display: true` adds a live window with `q` to interrupt. `main.py` + pipelines.
13. **CTC decode options** — `method` (`beam_search` default) / `beam_size` (/`10`) for CRNN output text. `AI_Script/postprocess/postprocessor/decode_crnn/decode_crnn.py`.
14. **Registry/factory extensibility** — all four domains (models, preprocess, postprocess, evaluate) expose the identical `@XRegistry.register("key")` decorator + factory pattern. A new model normally means a new class in `AI_Script/models/` and a matching preprocess key (the preprocess registry is keyed by `model_name`), while postprocess classes are keyed by `default_postprocess` and evaluate classes by `task` (so they can be reused across models) — route with `default_postprocess`/`task` in the internal config (see [[project/overview|Overview]] § Config flow).
15. **In-repo sample data** — 3 datasets in `AI_Script/inputs/dataset/` (detection/COCO-Val-2017, classification `Dataset_Number`, HTR `Dataset_HTR`), sample images and sample videos under `AI_Script/inputs/`.

## Notes on interpretation

- There is no separate "plugin" registry beyond the four domain registries; features *count* therefore = the axes above + option keys (see [[project/overview|Overview]] § Config flow for the `extras` mechanism that activates them).
- Feature *counts* drift as the code evolves; re-verify with the commands/files listed in the table before quoting the numbers.
- Video input is supported by the pipelines as such, but the drawing postprocessors constrain which task/model can accept which input type — see the input-compatibility matrix in [[pipeline/output-schemas|Output Schemas]].

## Related Notes

- [[project/overview|Overview]] — what the framework is and how config flows
- [[project/models|Models]] — per-family facts, weights, datasets
- [[project/precision-quantization|Precision & Quantization]] — detailed precision labels, weight mapping, IO casting rules, INT8/BF16 variants
- [[pipeline/mode-flows|Mode Flows]] — step order per mode
- [[pipeline/output-schemas|Output Schemas]] — artifacts, file names, JSON schema
