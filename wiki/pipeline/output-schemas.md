# XAIP — Output Schemas & Artifacts

**Summary**: Definitive reference for all generated files and JSON structures across modes, tasks, and model families, cross-checked with the source code.
**Tags**: #pipeline #schema #output
**Created**: 2026-11-09
**Last Updated**: 2026-11-10

---

## Output location rule

All artifacts are written inside the directory specified by `output_path` in the YAML config.
- `output_path` is strictly required (no default). If missing or pointing to an existing file, the pipeline fails immediately with `ValueError` or `NotADirectoryError`.
- Missing directories in `output_path` are automatically created via `os.makedirs(output_path, exist_ok=True)`.

---

## Artifact naming matrix

| Mode | Task / Model | Input Type | Written Artifact (pattern) |
|---|---|---|---|
| **Inference** | `object_detection` (`yolov5`, `detr`) | Image / Folder / Npy | `<output_path>/<model_name> \| <precision_format> \| <YYYY-MM-DD HH:MM:SS.ffffff>.jpg` |
| **Inference** | `object_detection` (`yolov5`, `detr`) | Video | `<output_path>/<model_name> video_detection <YYYY-MM-DD HH:MM:SS>.mkv` (codec: `FFV1`) |
| **Inference** | `classification` & `htr` (`lenet5`, `resnet50`, `vit`, `crnn`) | Image / Folder | `<output_path>/<model_name> <YYYY-MM-DD HH:MM:SS.ffffff>.jpg` |
| **Tracking** | `object_detection` (`yolov5`, `detr`) | Video | `<output_path>/<model_name> video_tracking <YYYY-MM-DD HH:MM:SS>.mkv` (codec: `FFV1`) |
| **Tracking** | `object_detection` | Image / Npy | *No file written* (returns in-memory dictionary `{'ID': [...], 'boxes': [...]}`) |
| **Debug** | Any single image / Npy | Image / Npy | 1. Postprocessed visual `.jpg` (named as in Inference)<br>2. Profiler JSON: `<output_path>/Output profiler - <model_name> <timestamp>.json`<br>3. Directory `<output_path>/intermediate_outputs/` with per-layer JSONs |
| **Evaluate** | Any supported task | Folder (`images/` + `labels/`) | `<output_path>/evaluate - <model_name> - <precision_format> - CPU - <YYYY-MM-DD HH:MM:SS>.json` |
| **Dashboard** *(API)* | `classification` evaluation JSONs | Tuple of 2 JSON paths | Directory `<output_path>/Benchmark_Chart_<YYYY-MM-DD_HH:MM:SS>/` with 5 comparison PNGs |

*Source code references for naming strings*:
- Detection image: `AI_Script/postprocess/postprocessor/draw_box/draw_box.py`
- Detection video: `AI_Script/core/pipeline_inference.py`
- Classification/HTR image: `AI_Script/postprocess/postprocessor/draw_label/draw_label.py`
- Tracking video: `AI_Script/core/pipeline_tracking.py`
- Debug layer files: `AI_Script/core/pipeline_debug.py`
- Debug profiler JSON: `AI_Script/models/base_model.py` (`BaseModel._session_option`, `profile_file_prefix`; ORT appends the timestamp)
- Evaluate JSON: `AI_Script/core/pipeline_evaluate.py`
- Benchmark charts: `AI_Script/core/pipeline_benchmark.py`
- Intermediate MSE comparison: `AI_Script/postprocess/Functions/Compare_Output_Intermediate.py` (`Compare_Ouput_Intermediate` — note the in-repo symbol typo) via `AI_Script/core/pipeline_compare_intermediate.py` (`_load_intermediate_source`)

---

## Visual postprocessing specifications

### 1. Detection (`draw_box.py`)
- **Boxes**: Blue rectangle `(255, 0, 0)` with thickness 1.
- **Labels**: Green text `(0, 255, 0)` formatted as `Label: <dataset_label>`.
- **Confidence**: Magenta text `(255, 0, 255)` formatted as `Score: 0.XX`.
- **Coordinates**: Unletterboxed back to the raw image resolution, honoring `resize_mode` (`"letterbox"` or `"stretch"`).

### 2. Classification & HTR (`draw_label.py`)
- **Canvas modification**: Image is resized to width 1200 maintaining aspect ratio, and a black header banner is added to the top.
- **Text**: Centered green text `(0, 255, 0)` on the black banner formatted as `Predicted: <label_or_decoded_text>`.
- *Source note*: `DrawLabel.draw` accepts only `image_path` inputs and raises `ValueError` for anything else (e.g. a numpy array). Video input additionally fails before drawing, because `Pipeline_Inference.run` passes a `save` flag that `DrawLabel.postprocess` does not accept. Classification/HTR video inference is therefore unsupported in the current drawing code — see `AI_Script/postprocess/postprocessor/draw_label/draw_label.py` (`DrawLabel.draw`) and `AI_Script/core/pipeline_inference.py` (`Pipeline_Inference.run` video branch).

### 3. Tracking (`pipeline_tracking.py`)
- **Boxes**: Blue rectangle `(255, 0, 0)` with thickness 1.
- **Track ID**: Green text `(0, 255, 0)` positioned at `(x1+2, y1+15)` formatted as `ID: <int>`.

---

## JSON Schemas

### 1. Evaluate Output JSON
Fixed 4-section top-level structure written by `Pipeline_Evaluate` across all tasks:

```json
{
  "Model_Information": {
    "model_name": "lenet5",
    "run_target": "CPU",
    "task": "classification",
    "weight_path": "/absolute/path/to/model.onnx",
    "precision_format": "fp32",
    "model_size_mb": "12.34 MB",
    "target_size": [32, 32]
  },
  "Dataset_Information": {
    "dataset_path": "/absolute/path/to/dataset",
    "number_images": 100
  },
  "Performance": {
    "total_time_s": 1.25,
    "throughput_fps": 80.0
  },
  "Evaluate": {
    "images": {
      "<image_filename>": {
        "image_path": "/absolute/path/to/dataset/images/<image_filename>",
        "metrics": { /* task-specific */ }
      }
    },
    "overall_metrics": { /* task-specific */ }
  }
}
```

*Note on `run_target`*: The value `"CPU"` is hard-coded in `AI_Script/core/pipeline_evaluate.py` in both the JSON payload and the file name, regardless of `target_device`.

#### Task-specific fields in `Evaluate`:

##### Task A: `object_detection` (`AI_Script/evaluate/tasks/detection.py`)
- Per-image `metrics`:
  ```json
  {
    "mAP@0.5": 0.852,
    "mAP@[0.5:0.95]": 0.614,
    "total_missed_objects": 0
  }
  ```
- `overall_metrics`:
  ```json
  {
    "avg_mAP@0.5": 0.812,
    "avg_mAP@[0.5:0.95]": 0.589
  }
  ```

##### Task B: `classification` (`AI_Script/evaluate/tasks/classification.py`)
- Per-image `metrics`:
  ```json
  {
    "correct": true,
    "predict": 3,
    "ground_truth": 3
  }
  ```
- `overall_metrics`:
  ```json
  {
    "accuracy": 0.95,
    "macro_avg": {
      "precision": 0.941,
      "recall": 0.932,
      "f1-score": 0.936
    },
    "micro_avg": {
      "precision": 0.95,
      "recall": 0.95,
      "f1-score": 0.95
    },
    "weighted_avg": {
      "precision": 0.952,
      "recall": 0.95,
      "f1-score": 0.95
    },
    "mcc": 0.923,
    "Per_class": {
      "0": { "precision": 0.9, "recall": 0.85, "f1-score": 0.87, "support": 20 },
      "accuracy": 0.95,
      "macro avg": { "precision": 0.941, "recall": 0.932, "f1-score": 0.936, "support": 100 },
      "weighted avg": { "precision": 0.952, "recall": 0.95, "f1-score": 0.95, "support": 100 }
    },
    "Confusion_matrix": [
      [17, 1, 0],
      [0, 18, 2],
      [1, 0, 19]
    ]
  }
  ```

##### Task C: `htr` (`AI_Script/evaluate/tasks/HTR.py`)
- Per-image `metrics`:
  ```json
  {
    "correct": true,
    "predict": "sample text",
    "ground_truth": "sample text"
  }
  ```
- `overall_metrics`:
  ```json
  {
    "accuracy": 0.895
  }
  ```
  *(Exact match ratio computed on stripped, lowercase strings. Raw character and word error rate functions `_cer` and `_wer` are defined in the class but not aggregated in `benchmark_toltal`).*

#### Evaluate label formats
Ground-truth pairing reads `labels/<image-name>.txt` per image — detection: `class_id xc yc w h` (normalized); classification: a single integer; HTR: plain text. Parsers: `AI_Script/evaluate/tasks/detection.py` (`_load_labels`), `classification.py` (`_get_labels`), `HTR.py` (`_get_labels`). See the *Evaluate label pairing* note in `GUIDELINE.md` § 2 (labeled paragraph just before its Advanced notes — not a `§` heading).

---

### 2. Debug Intermediate Output JSON
Each layer is saved to a distinct file under `output_path/intermediate_outputs/`:
- **File name format**: `{index:04d}__{sanitized_layer_name}.json` (e.g. `0015__model_layer1_Conv_output_0.json`).
- **File content schema** (`AI_Script/core/pipeline_debug.py`):

```json
{
  "layer_name": "/model/layer1/Conv_output_0",
  "index": 15,
  "dtype": "float32",
  "shape": "(1, 64, 112, 112)",
  "values": [
    /* nested JSON list matching shape and dtype */
  ]
}
```

*Code precision notes*:
- `shape` is stored as a stringified tuple representation (e.g. `"(1, 64, 112, 112)"`), produced by `_get_shape(value)`.
- Initializer tensors (weights, biases) are excluded from the output dump by default to prevent memory exhaustion; set `debug_dump_initializers: true` in the YAML to include them.

---

### 3. Comparison Tooling Schemas (Python API)

#### `factory.benchmark_dashboard((json_a, json_b))`
De-facto classification inputs, not a code-enforced requirement: the five charts read classification-schema keys (`accuracy` / `mcc` / `Per_class` / `Confusion_matrix` / `macro_avg`–`weighted_avg` from `Evaluate.overall_metrics`, plus `Performance`), so classification evaluate JSONs are needed in practice — `pipeline_benchmark.py` checks no task (verified in `run()`); a non-classification JSON simply breaks the chart code on the classification keys. Generates five visualization images inside `Benchmark_Chart_<timestamp>/`:
1. `overall_metrics.png` — Bar chart of `accuracy` and `mcc` side-by-side.
2. `avg_metrics.png` — Grouped bar chart comparing Macro, Micro, and Weighted averages (`precision`, `recall`, `f1`).
3. `per_class_metrics.png` — Class-by-class comparison of `precision`, `recall`, `f1`.
4. `confusion_matrices.png` — Side-by-side heatmap comparison of the confusion matrices.
5. `performance_hardware.png` — Dual plot comparing `total_time_s` and `throughput_fps`.

#### `factory.compare_intermediate((source_a, source_b))`
Accepts either two `intermediate_outputs/` directory paths or two single legacy JSON files.
- Returns a tuple in Python: `(per_layer_mse: Dict[str, float], overall_mse: float)`.
- Verified formula: mean squared error across flattened layer values. Both sources must have matching tensor key sets.

## Related Notes

- [[project/overview|Overview]] — how `output_path` is propagated through configs
- [[project/models|Models]] — model names, tasks, and dataset bindings
- [[pipeline/mode-flows|Mode Flows]] — pipeline step sequences generating these files
- [[pipeline/debug-internals|Debug Internals]] — how intermediate layer JSONs and profiler JSON are produced
