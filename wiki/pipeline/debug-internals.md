# XAIP — Debug Mode Internals

**Summary**: Deep dive into the unique debug pipeline: ONNX graph rewriting, topological layer ordering, memory-safe disk streaming, ORT profiling, and automatic cleanup.
**Tags**: #debug #pipeline #internals
**Created**: 2026-11-09
**Last Updated**: 2026-11-10

---

## Overview

`Pipeline_Debug` (`AI_Script/core/pipeline_debug.py`) is the only mode that **mutates the ONNX graph** before inference. It rewrites the model so every node tensor (initializers excluded by default) becomes a graph output, runs a single inference, streams each layer's output to disk as JSON, captures ORT profiling data, then deletes the temporary model.

**Constraints**:
- Single input only: `image_path`, `npy_path`, or `numpy_array`
- Folder/video inputs raise explicit `ValueError`
- Triggered by `debug_mode: true` in config

---

## Graph mutation: `_dump_model()`

Executed once in `__init__` before any inference:

1. **Load + shape inference**
   ```python
   model = onnx.load(model_path)
   inferred_model = shape_inference.infer_shapes(model)
   ```

2. **Collect tensor dtype/shape info**
   - Dtype resolution does not use a pre-built lookup: the `get_dtype_for_tensor` closure (step 4 below) reads `inferred_model.graph.value_info` first, then `graph.input`, `graph.output`, and `graph.initializer` directly.
   - A `vi_info` dict collecting `value_info` + initializer entries is still built inside `_dump_model()` (`AI_Script/core/pipeline_debug.py`) but never consulted — dead code, not part of the dtype-resolution path. The resulting per-layer JSON's dtype field is described in [[pipeline/output-schemas|Output Schemas]].

3. **Determine tensors to dump**
   - Iterate all `node.input` and `node.output` names
   - **Skip empty names** (`""`) — optional inputs
   - **Skip initializers** by default (weights/biases already in `.onnx` on disk; dumping as JSON bloats RAM ~8× via `tolist()`)
   - Config key `debug_dump_initializers: true` restores old behavior (include initializers)

4. **Resolve dtype for each tensor** (closure `get_dtype_for_tensor`):
   - Priority: `value_info` → `graph.input` → `graph.output` → `initializer`
   - Special handling for `_quantized` suffix tensors (QuantizeLinear/QLinearConv/QGemm): inspect producing node's zero-point initializer to infer dtype
   - Fallback: `INT8` for `_quantized` tensors with unresolved dtype

5. **Append graph outputs**
   ```python
   model.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
   ```
   Shape `None` lets ORT infer at runtime. Only tensors not already in `graph.output` are appended, and a non-`_quantized` tensor whose dtype cannot be resolved is skipped with a warning instead of appended.

6. **Compute topological layer order** (once, stored as `self._layer_order`):
   ```python
   self._layer_order = [str(list(node.output)[0]) for node in model.graph.node if node.output]
   ```
   First output of each node, in node order. Replaces legacy `_sort_right_index` that re-loaded ONNX every run.

7. **Save temp model** → `output_path/dump_model.onnx`
   - Updates `config["weight_path"]` to point to this temp file
   - `ModelFactory.create(debug_mode=True)` loads it

---

## ORT profiling (debug mode only)

In `BaseModel._session_option()`:
```python
sess_options.enable_profiling = self.debug_mode
if self.debug_mode:
    sess_options.profile_file_prefix = os.path.join(output_path, f"Output profiler - {self.model_name} ")
```
After `session.run()`, the model class calls `session.end_profiling()` — this lives in each registered model's `predict()`, not in `pipeline_debug.py` or `base_model.py`: all 6 files under `AI_Script/models/{detection,classification,HTR}/` gate it with `if self.debug_mode:` immediately after `session.run` (verified example: `AI_Script/models/detection/yolov5.py` → `predict()`), writing `Output profiler - <model_name> <timestamp>.json` to `output_path`. Profiling is **disabled** for inference/tracking/evaluate modes.

---

## Inference & layer streaming: `run()`

After preprocess + model inference + postprocess (saves visual `.jpg`):

1. **Get session output names & map to results**
   ```python
   output_names = [o.name for o in self.model.session.get_outputs()]
   name_to_result = dict(zip(output_names, result_1))
   del result_1
   ```

2. **Determine layer order**:
   - **CRNN**: uses session output order directly (no topological re-sort)
   - **Others**: topological order from `_layer_order` + leftover tensors (sorted):
     ```python
     ordered = [n for n in self._layer_order if n in name_to_result]
     leftover = sorted(set(name_to_result) - set(ordered))
     layer_names = ordered + leftover
     ```

3. **Prepare output directory** → `output_path/intermediate_outputs/`
   - Removes stale `*.json` from previous runs

4. **Stream each layer to disk** (memory-safe):
   ```python
   for idx, layer_name in enumerate(layer_names):
       value = name_to_result.pop(layer_name, None)
       if value is None: continue
       payload = {
           "layer_name": layer_name,
           "index": idx,
           "dtype": str(value.dtype),
           "shape": self._get_shape(value),  # stringified tuple e.g. "(1, 64, 112, 112)"
           "values": value.tolist(),
       }
       del value  # free numpy array immediately
       
       sanitized = self._sanitize_layer_name(layer_name)
       # collision handling: append _<md5[:6]>, re-truncate to 200
       
       file_name = f"{idx:04d}__{sanitized}.json"
       with open(file_path, "w", encoding="utf-8") as f:
           json.dump(payload, f, ensure_ascii=False, indent=2)
       del payload  # free Python list immediately
   ```
   - Each layer's `tolist()` and JSON are freed before next layer → **OOM prevention** for large models

5. **Sanitization** (`_sanitize_layer_name`):
   - Regex: `[^A-Za-z0-9._-]` → `_`
   - Strip leading/trailing `_`; fallback `"unnamed"`
   - Truncate to 200 chars
   - Collisions: append `_<md5(layer_name)[:6]>`, re-truncate

---

## Cleanup

After all layers written:
1. Profiler JSON already written by `session.end_profiling()`
2. **Temp model removal**:
   ```python
   dump_path = config.get("weight_path", "")
   if os.path.basename(dump_path) == "dump_model.onnx" and os.path.exists(dump_path):
       os.remove(dump_path)
   ```

---

## Output artifacts summary

| Artifact | Location | Source |
|---|---|---|
| Postprocessed visual | Detection: <output_path>/<model_name> \| <precision_format> \| <timestamp>.jpg<br>Cls/HTR: <output_path>/<model_name> <timestamp>.jpg | Same naming as inference mode |
| ORT profiler | `<output_path>/Output profiler - <model_name> <timestamp>.json` | `session.end_profiling()` in `AI_Script/models/<domain>/<model>.py` → `predict()` (gated `if self.debug_mode:` right after `session.run`); profile prefix from `BaseModel._session_option` |
| Intermediate layers | `<output_path>/intermediate_outputs/NNNN__<sanitized_layer>.json` | One file per layer, topological order (CRNN keeps native session-output order) |
| Temp model (auto-removed) | `output_path/dump_model.onnx` | Created in `_dump_model()`, deleted after run |

---

## Key implementation notes

- **No initializer dump by default** — prevents 8× RAM blowup from `tolist()` on weight tensors. Enable via `debug_dump_initializers: true` in config.
- **CRNN special case** — uses session output order, not topological order (legacy behavior preserved).
- **Shape stored as string** — `_get_shape()` returns `str(obj.shape)` e.g. `"(1, 64, 112, 112)"`, not a JSON array.
- **Profiling prefix includes trailing space** — `f"Output profiler - {self.model_name} "` → ORT appends timestamp.
- **Graph optimization level** — configurable via `graph_optimization_level` (default `all`), same as other modes. CRNN FP16 needs `extended`.

---

## Related Notes

- [[pipeline/mode-flows|Mode Flows]] — macro flow comparison with other modes
- [[pipeline/output-schemas|Output Schemas]] — exact JSON schemas for intermediate layer files and profiler
- [[project/precision-quantization|Precision & Quantization]] — `graph_optimization_level` requirements per precision
- [[project/features|Features]] — feature inventory item #4 (debug instrumentation)