# XAIP — Error Map: Where Invalid Configs Fail Fast

**Summary**: For each class of user mistake, the file/symbol that catches it first and the exception behavior — a fail-fast navigation aid so agents and users do not have to grep when a run dies.
**Tags**: #error-handling #pipeline
**Created**: 2026-11-11
**Last Updated**: 2026-11-11

---

## Design rule

The codebase fails fast with descriptive exceptions — `output_path` and key config inputs are validated at several stacked points rather than once, and there are no silent weight fallbacks (single deliberate exception: the ORT device fallback, below). Exact exception messages are deliberately **not quoted** here — texts are the most volatile part; read the cited file+symbol for the current wording.

## Stage 1 — CLI startup (`main.py`)

| User mistake | Caught where | Behavior |
|---|---|---|
| No command-line argument | `main.py` argv check | prints usage, `sys.exit(1)` |
| `output_path` missing / empty / whitespace | `main.py` output_path block | message + `sys.exit(1)` |
| `output_path` points to an existing file | `main.py` output_path block | message + `sys.exit(1)` |
| Cannot create `output_path` (e.g. permissions) | `os.makedirs` guarded by `PermissionError` | message + `sys.exit(1)` |
| Missing **any** of `model_name`, `precision_format`, `inference_mode`, `debug_mode`, `evaluate_mode`, `tracking_option`, `input_path` | direct index `config["…"]` when building `factory_pipeline` / running | **`KeyError`** — these keys are required literally; there is no friendly message for them. This is why every YAML needs all four mode keys present. |
| Unknown extra YAML keys | `main.py` `_HANDLED_KEYS` set | **not an error** — forwarded verbatim as `extras` (see [[project/overview|Overview]] § Config flow) |

## Stage 2 — `factory_pipeline.__init__` (`AI_Script/core/factory.py`)

| User mistake | Caught where | Behavior |
|---|---|---|
| `output_path` `None`/blank | `__init__` output_path block | `ValueError` |
| `output_path` is an existing file | `__init__` output_path block | `NotADirectoryError` (created otherwise, `exist_ok=True`) |
| `model_name` has no internal config | `load_config` (alias `resnet → resnet50` applied first) | `FileNotFoundError` with the config path |
| All mode flags falsy | `_get_pipeline()` end | `ValueError` ("invalid pipeline / parameter conflicts") |
| Several mode flags `true` at once | **not caught anywhere** | silent precedence: `inference_mode` (→ `tracking_option` ? Tracking : Inference) beats `debug_mode` beats `evaluate_mode` — a convention, not validation |

## Stage 3 — pipeline construction (`AI_Script/core/pipeline_*.py` `__init__`)

| Pipeline | `output_path` check | Other `__init__` failures |
|---|---|---|
| `Pipeline_Inference` | `ValueError` if missing/blank | registry misses (see Stage 5) |
| `Pipeline_Tracking` | same | same |
| `Pipeline_Debug` | same | model path resolution (below) |
| `Pipeline_Evaluate` | same | model path resolution (below) |
| `Pipeline_Benchmark` | validates **only if** `output_path` already in config (`__init__` accepts `config=None`); standalone use re-checks inside `run()` | — |
| `Pipeline_Compare_Intermediate` | none — has no `__init__`, writes no files | — |

## Stage 4 — model / session (`AI_Script/models/base_model.py`)

| User mistake | Caught where | Behavior |
|---|---|---|
| `output_path` missing/blank in config | `_session_option()` | `ValueError` |
| Unknown `graph_optimization_level` | `_session_option()` | `ValueError` listing the 4 valid options |
| Unknown `precision_format` | `load_model()` | `FileNotFoundError` ("no support precision format") |
| Known `precision_format` but its `weight_path_*` key absent from the internal config | `load_model()` — **not** a named error | `str(None)` becomes the path, and ORT fails opening a nonexistent file — the descriptive precision error does **not** fire here. The duplicated mappings in `pipeline_debug.py` → `_get_model_path()` and `pipeline_evaluate.py` → `_get_weight_path()` have the same shape. |
| `weight_path` file does not exist | ORT `InferenceSession` | ORT's own load error (not wrapped) |
| Device `target: {device: gpu}` but CUDA unavailable | `load_model()` provider list | **deliberate silent fallback to CPU** — the one non-fail-fast path; check ORT session providers if a "GPU run" is suspiciously slow |

`precision_format` is duplicated in three places (model layer + debug + evaluate pipelines) — a fix to the mapping must be applied to all three (see [[project/conventions|Conventions]] § Config key names).

## Stage 5 — registry / domain factories

| User mistake | Caught where | Behavior |
|---|---|---|
| `model_name` not in model registry | `XRegistry.get` (`AI_Script/models/registry_model.py`) | `ValueError` listing available keys |
| `model_name` not in preprocess registry | `XRegistry.get` (`AI_Script/preprocess/registry_preprocess.py`) | `ValueError` |
| `default_postprocess` not registered | `XRegistry.get` (`AI_Script/postprocess/registry_postprocess.py`) | `ValueError` |
| `task` not registered | `XRegistry.get` (`AI_Script/evaluate/registry_evaluate.py`) | `ValueError` |
| Duplicate key registered | `XRegistry.register` decorator | `ValueError` at import time |
| Non-`BaseModel` subclass registered | model registry decorator | `TypeError` |
| Detection model with no adapter entry | `Adapter.__init__` (`AI_Script/postprocess/Functions/Adapter_Detection.py`) | `ValueError` listing supported models (`yolov5`, `detr`) |
| Unknown `resize_mode` | YOLO preprocessor `__init__` (`AI_Script/preprocess/detection/yolo.py`) | `ValueError` (`letterbox` / `stretch`) |

## Stage 6 — runtime data

| User mistake | Caught where | Behavior |
|---|---|---|
| `input_path` does not exist | `check_file` (`AI_Script/core/utils.py`) returns `invalid_path` → pipeline dispatch | `ValueError` ("unsupported input type") from the running pipeline |
| Folder with no images (inference) | `Pipeline_Inference.run` folder branch | warning + empty result list (**not** an exception) |
| Per-item failure inside folder loop (inference) | per-item `try/except` | error printed, item skipped, run continues |
| Empty-box edge case | `unletterbox` (`AI_Script/postprocess/Functions/Boxes_Steps.py`) | returns empty `(0,4)` int32 array instead of raising |

## Related Notes

- [[project/overview|Overview]] — dispatch order and config flow feeding these validations
- [[project/conventions|Conventions]] — why these key names are frozen contracts
- [[pipeline/mode-flows|Mode Flows]] — the stage sequence these errors interrupt
- [[project/features|Features]] #1 — fail-fast validation inventory
