# XAIP — What This Project Is

**Summary**: XAIP ("AI Inference Pipeline") is a modular Python framework for running, debugging and evaluating computer-vision models via ONNX Runtime (CPU/CUDA), driven entirely by YAML configs.
**Tags**: #project #architecture
**Created**: 2026-11-09
**Last Updated**: 2026-11-11

---

## What it is

XAIP is a pipeline runner, not a training framework. A user writes one YAML config, and the app:

1. picks the right **pipeline** from four mode flags,
2. assembles the model-agnostic backbone **preprocess → ONNX model → postprocess** (or → evaluate) from **registry/factory** lookups,
3. writes all artifacts into one required `output_path` directory.

Supported scope: three tasks (`object_detection`, `classification`, `htr`), four precision labels (`fp32`, `fp16`, `int8`, `bf16`) — quantization-aware comparison is a first-class goal. Verified model list: see [[project/models|Models]]. Verified capability list: see [[project/features|Features]].

## Dispatch (single decision point)

`main.py` reads the user YAML and constructs `factory_pipeline(...)`. Mode selection happens in `AI_Script/core/factory.py` → `factory_pipeline._get_pipeline()`:

```
inference_mode  →  tracking_option ? Pipeline_Tracking : Pipeline_Inference
debug_mode      →  Pipeline_Debug
evaluate_mode   →  Pipeline_Evaluate
none of them    →  ValueError
```

Verified order of the `if/elif` chain means **inference_mode wins over debug_mode**, and debug wins over evaluate, when several flags are `true`. "Exactly one main mode `true`" is a repo convention (enforced by templates + `GUIDELINE.md` § 2), not by code.

## Config flow (single source of truth chain)

1. **Internal model config** — one YAML per registered model in `AI_Script/configs/<model_name>.yaml` (see `factory_pipeline.load_config`, which also aliases `resnet` → `resnet50.yaml`; any name without a config fails with `FileNotFoundError`). Holds default weights per precision, `task`, `target_size`, thresholds, `default_postprocess`.
2. `factory_pipeline.__init__` injects `output_path`, `display`, then overwrites from constructor args (`weight_path`, `precision_format`, `target_device`, `dataset`).
3. **Everything else in the user YAML** is forwarded verbatim into the config by `main.py` (`_HANDLED_KEYS` minus the leftover = `extras` → `factory_pipeline(..., extras=...)`). This is how optional keys reach whichever layer consumes them — mostly the preprocessor / model / postprocessor (`input_dtype` → `BasePreprocessor.__call__`, `graph_optimization_level` → `BaseModel._session_option`, `resize_mode` → the YOLO preprocessor and `draw_box`, `method`/`beam_size` → `DecodeCRNN`, `transpose_output` → `draw_box`); the pipeline-level ones land in `AI_Script/core/` (`inference_workers` → `Pipeline_Tracking`, `debug_dump_initializers` → `Pipeline_Debug`). Full option table → [[project/features|Features]].
4. Domain factories resolve concrete classes: preprocess registry keyed by `model_name`, postprocess registry keyed by `default_postprocess`, evaluate registry keyed by `task`, model registry keyed by `model_name`. Every domain follows the same `XRegistry`/`XFactory` decorator pattern (`@XRegistry.register("key")` on a class file).

## Entrypoints

| Method | How | Verified in |
|---|---|---|
| CLI run | `python main.py <config.yaml>` (`main.py`) | `main.py` requires all four mode keys to be literally present in the YAML (indexes them directly) and validates `output_path` |
| Programmatic full run | `factory_pipeline(...)(input_source)` | `AI_Script/core/factory.py` → `factory_pipeline.__call__` |
| Benchmark dashboard | `factory_pipeline(...).benchmark_dashboard((eval_json_a, eval_json_b))` | `AI_Script/core/pipeline_benchmark.py` — not config-driven; sample invocation commented at the bottom of `main.py` |
| Layer-wise MSE | `factory_pipeline(...).compare_intermediate((src_a, src_b))` | `AI_Script/core/pipeline_compare_intermediate.py` — same |

## Where things live

| Role | Location |
|---|---|
| Config templates (schema-only) | `config_templates/templates/` (list with `ls config_templates/templates/`) |
| Ready-to-run example configs (all paths are placeholders — edit before reuse) | `config_templates/inference/`, `config_templates/eval/`, `config_templates/debug/` — per-mode pipeline configs (list with `ls config_templates/<mode>/`) |
| Pipelines + mode dispatch | `AI_Script/core/` |
| Model classes | `AI_Script/models/{detection,classification,HTR}/` |
| Preprocess classes | `AI_Script/preprocess/{detection,classification,HTR}/` |
| Postprocess classes | `AI_Script/postprocess/postprocessor/` (`draw_box`, `draw_label`, `get_labels`, `decode_crnn`), shared helpers in `AI_Script/postprocess/Functions/` |
| Internal configs + label maps | `AI_Script/configs/` (`<model>.yaml`, `datasets.json`) |
| ByteTrack (tracking mode) | `AI_Script/tracker/ByteTrack/` |
| Evaluation task implementations | `AI_Script/evaluate/tasks/` |
| In-repo sample data | `AI_Script/inputs/dataset/` (sample datasets — list with `ls AI_Script/inputs/dataset/`), `AI_Script/inputs/images/`, `AI_Script/inputs/videos/` |

## Environment facts

- Entry script for setup/verification: `run.sh` (auto conda setup + run) and `check_env.py` (standalone check; `--gpu` additionally asserts `CUDAExecutionProvider`).
- Python 3.12 pinned conda env `xaip` in `environment.yml`; pip-only mirror in `requirements.txt` (CPU wheels by default; GPU needs `onnxruntime-gpu` + the `cu118` wheel index — read `requirements.txt` comments and `GUIDELINE.md` § 1 for the current commands).
- Device behavior (verified in `AI_Script/models/base_model.py` → `load_model`): `target: {device: gpu}` requests `[CUDAExecutionProvider, CPUExecutionProvider]`, anything else CPU-only; if CUDA is unusable ORT silently falls back to CPU.
- Runtime options with a verified cross-model effect: `graph_optimization_level` (`disable|basic|extended|all`, default `all`) and precision-IO auto-casting — see [[project/features|Features]] and [[project/models|Models]].

## Repo documentation map

- `README.md` — introduction, task/model table, project structure.
- `GUIDELINE.md` — § 0 quick start, § 1 environment, § 2 configuration, § 3 execution, § 4 Output (output artifacts + JSON schemas).
- `AGENTS.md` — agent/hardware contract, constraints, wider architecture background.

## Related Notes

- [[project/features|Features]] — what is supported, with re-verifiable counts
- [[project/models|Models]] — model families, tasks, datasets, weights
- [[project/precision-quantization|Precision & Quantization]] — precision labels, weight mapping, IO casting, validated per-model behaviors
- [[pipeline/mode-flows|Mode Flows]] — macro flow per mode
- [[pipeline/output-schemas|Output Schemas]] — what each mode writes
