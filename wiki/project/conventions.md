# XAIP — Naming Conventions & Kept Typos

**Summary**: Names and config keys that are load-bearing API contracts — renaming any of them is a breaking change, so the wiki states them as fixed; everything else follows standard Python style.
**Tags**: #conventions #project
**Created**: 2026-11-11
**Last Updated**: 2026-11-11

---

## Kept typos (never rename)

These misspellings are public API. They are called from more than one place and there is no test suite or alias layer to protect a rename.

| Kept name | Where it lives | Why it is fixed |
|---|---|---|
| `benchmark_toltal` | `AI_Script/evaluate/base_evaluate.py` (abstract method) | Implemented by all three task classes (`AI_Script/evaluate/tasks/{detection,classification,HTR}.py`) and called by `AI_Script/core/pipeline_evaluate.py` → `run()`. Renaming the abstract method breaks all four files at once. |
| `Compare_Ouput_Intermediate` | `AI_Script/postprocess/Functions/Compare_Output_Intermediate.py` (function; the **file name is spelled correctly**) | Imported by `AI_Script/core/pipeline_compare_intermediate.py` and it is the user-facing return value of `factory_pipeline.compare_intermediate(...)`. |
| `Compare_Output_Intermediate` (file) vs `Compare_Ouput_Intermediate` (symbol) | same module | Documented together so a reader does not "fix" one of the two spellings and break the import. |

## Registry keys are exact-match lowercase strings

Every registry lookup is a plain string equality against the value in the config — no normalization, no case folding, no fuzzy matching:

- **Model registry** — key = `model_name` (`AI_Script/models/factory_model.py`).
- **Preprocess registry** — key = `model_name` (`AI_Script/preprocess/factory_preprocess.py` → `create()`).
- **Postprocess registry** — key = `default_postprocess` (`AI_Script/postprocess/factory_postprocess.py` → `create()`).
- **Evaluate registry** — key = `task` (`AI_Script/evaluate/factory_evaluate.py` → `create()`).

Consequences that are decisions, not accidents:

- Config values such as `model_name: YOLOv5` or `default_postprocess: draw_box ` (trailing space would not survive `str()` in the factory — the lookup uses `str(config.get(key))`, so whitespace **inside** the string still fails) break the lookup with a descriptive `ValueError` from `XRegistry.get` (see [[pipeline/error-map|Error Map]]).
- `resnet` must stay registered as a **preprocess** key even though the internal config file is `resnet50.yaml`: `factory_pipeline.load_config` aliases `resnet → resnet50` only for the **config file** lookup (`AI_Script/core/factory.py` → `load_config`, alias dict `{"resnet": "resnet50"}`); the `model_name` value `"resnet"` then flows unchanged into the preprocess/model registries. Removing the `resnet` preprocess key breaks every config that spells the model `resnet`.
- Multi-key registration files (`yolo`+`yolov5` in `AI_Script/preprocess/detection/yolo.py`; `resnet`+`resnet50`+`resnet16` in `AI_Script/preprocess/classification/resnet.py`) are deliberate aliases — do not prune "unused" keys without checking configs and [[project/models|Models]].

## Adapter dict keys — the detection intermediate contract

`AI_Script/postprocess/Functions/Adapter_Detection.py` → `Adapter.__call__` returns a plain dict with exactly these keys, and every detection consumer is written against them:

| Key | Type | Meaning |
|---|---|---|
| `boxes` | `np.ndarray` `[N,4]` | Boxes in **xyxy**, pixel coordinates in `target_size` space (before unletterbox) |
| `conf` | `np.ndarray` `[N]` | Confidence (YOLOv5: objectness × max class score; DETR: max class score) |
| `class` | `np.ndarray` `[N]` | Integer class ids |

Consumers (all verified against this dict): `AI_Script/postprocess/postprocessor/draw_box/draw_box.py`, the tracking wrapper `AI_Script/tracker/ByteTrack/ByteTrack.py` → `preprocess_box`, and the evaluate route `AI_Script/evaluate/tasks/detection.py` → `_adapter_boxes`. Renaming a key silently breaks all three — there is no shared schema object.

## Config key names are part of the contract

The keys below are read in several independent places, so they cannot be casually renamed without a cross-module sweep:

- `output_path` — validated in `main.py`, `factory_pipeline.__init__`, each artifact-writing pipeline's `__init__`, `BaseModel._session_option`, and `DrawBox.__init__` (fail-fast, see [[pipeline/error-map|Error Map]]).
- `precision_format` / `weight_path*` — read in **three** places: `AI_Script/models/base_model.py` → `load_model()`, `AI_Script/core/pipeline_debug.py` → `_get_model_path()`, `AI_Script/core/pipeline_evaluate.py` → `_get_weight_path()` (duplicated mapping, kept in sync by hand).
- `target`, `display` / `display_option`, `dataset`, `target_size`, `conf_threshold`, `iou_threshold`, `resize_mode`, `input_dtype`, `graph_optimization_level`, `transpose_output` — full flow table in [[project/overview|Overview]] § Config flow.

## General code style (conventions, not contracts)

- Classes PascalCase, functions/variables/keys snake_case, files snake_case.
- Registry keys lowercase (`yolov5`, `draw_box`, `object_detection`, `htr`).
- Registry classes follow the identical decorator pattern in all four domains: `@XRegistry.register("key")` above the class, `XFactory.create(config)` as a stateless static entry point (verified identical shape across `models/`, `preprocess/`, `postprocess/`, `evaluate/` registries).

## Related Notes

- [[project/overview|Overview]] — how the registry/factory pattern resolves config to classes
- [[project/features|Features]] — feature inventory including the extension recipe
- [[project/models|Models]] — exact registered keys and aliases per domain
- [[pipeline/error-map|Error Map]] — what happens when a key is misspelled or missing
