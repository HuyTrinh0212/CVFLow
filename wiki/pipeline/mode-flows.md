# XAIP — Macro Flow per Mode

**Summary**: High-level execution flow for each of the four operating modes (inference, tracking, debug, evaluate), traced from config ingestion to written artifacts.
**Tags**: #pipeline #flow
**Created**: 2026-11-09
**Last Updated**: 2026-11-10

---

## Universal upstream flow (all modes)

Every run begins identically:

```
CLI / API
   │
   ▼
main.py (loads config YAML, validates output_path, separates extras)
   │
   ▼
AI_Script/core/factory.py: factory_pipeline.__init__
   ├── load_config(model_name)           # reads AI_Script/configs/<model>.yaml
   ├── inject output_path & display
   ├── apply constructor overwrites      # weight_path, precision_format, extras...
   └── _get_pipeline()                   # instantiates exactly one Pipeline_*
```

Inside every pipeline, the per-item inference step follows the identical interface contract:

```
input_source (image, frame, npy)
   │
   ▼
preprocessor(input)      # PreprocessorFactory.create(config) -> [1, C, H, W] ndarray
   │
   ▼
model(preprocessed)      # ModelFactory.create(config) -> List[np.ndarray] (ORT session.run)
```

The difference between modes is what happens **before** (graph mutation, data iteration) and **after** (drawing, tracking, JSON dumping, metric reduction).

---

## 1. Inference Mode (`Pipeline_Inference`)

- **Trigger**: `inference_mode: true` (and `tracking_option: false`).
- **File**: `AI_Script/core/pipeline_inference.py`.
- **Accepted input types**: checked by `AI_Script/core/utils.py` → `check_file(input_source)`:
  - `image_path` / `npy_path` / `numpy_array` (single item)
  - `folder_path` (batch folder of images)
  - `video_path` (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Macro flow

```
run(input_source)
   ├── Case 1: single input ('image_path' | 'npy_path' | 'numpy_array')
   │     ├── preprocessor(input_source)
   │     ├── model(preprocessed_data)
   │     └── postprocessor(raw_output, input_source, save=True)
   │           └── writes annotated .jpg to output_path
   │
   ├── Case 2: folder input ('folder_path')
   │     ├── lists and sorts (.png, .jpg, .jpeg)
   │     └── for each image:
   │           ├── preprocessor(img_path) -> model(...)
   │           └── postprocessor(raw_output, img_path, save=True)
   │                 └── writes annotated .jpg to output_path
   │
   └── Case 3: video input ('video_path')
         ├── cv2.VideoCapture(input_source)
         ├── cv2.VideoWriter(output_path/<model> video_detection <timestamp>.mkv, 'FFV1')
         ├── while True (frame loop):
         │     ├── preprocessor(frame) -> model(...)
         │     ├── postprocessor(raw_output, frame, save=False)  # returns annotated frame
         │     ├── cv2.imshow (if display=True; break on 'q')
         │     └── VideoWriter.write(annotated_frame)
         └── cap.release() & out.release()
```

*Code pointers*: `draw_box.py` for detection models (draws boxes + labels); `draw_label.py` for classification and HTR (draws banner on top of image). Note: `draw_label.py` expects a file path on disk, so video inference is detection-only in practice.

---

## 2. Tracking Mode (`Pipeline_Tracking`)

- **Trigger**: `inference_mode: true` AND `tracking_option: true`.
- **File**: `AI_Script/core/pipeline_tracking.py`.
- **Constraints**: Detection models only (`yolov5`, `detr`). Uses `AI_Script/postprocess/Functions/Adapter_Detection.py`, which rejects other model families.
- **Accepted input types**: `video_path` (primary use case) or single item (returns raw IDs and boxes in memory; does not save video).

### Macro flow (video)

```
run(video_path)
   ├── cv2.VideoCapture(video_path)
   ├── cv2.VideoWriter(output_path/<model> video_tracking <timestamp>.mkv, 'FFV1')
   ├── n_workers = config.get("inference_workers", 1)
   │
   ├── [Serial path (n_workers <= 1)]:
   │     └── while read frame:
   │           ├── preprocessor(frame) -> model(...)
   │           ├── Adapter(outputs)                       # parses model tensor layout to boxes/conf/class
   │           └── _consume(dicts, frame)
   │
   ├── [Frame-parallel path (n_workers > 1)]:
   │     └── ThreadPoolExecutor(n_workers)
   │           ├── worker threads: _infer_frame (preprocessor + model + Adapter)
   │           └── main thread pops in strict submission order -> _consume(dicts, frame)
   │
   └── _consume(dicts, frame):
         ├── ByteTrack.update(dicts, width, height)       # Kalman filter + Hungarian matching
         ├── _draw_box_id(frame, online_targets)          # unletterbox -> cv2.rectangle + ID text
         ├── cv2.imshow (if display=True; break on 'q')
         └── VideoWriter.write(frame)
```

*Code pointers*: `AI_Script/tracker/ByteTrack/ByteTrack.py` (wrapper), `AI_Script/tracker/ByteTrack/tracker/byte_tracker.py` (core BYTETracker).

---

## 3. Debug Mode (`Pipeline_Debug`)

- **Trigger**: `debug_mode: true`.
- **File**: `AI_Script/core/pipeline_debug.py`.
- **Constraints**: Single image input only (`image_path`, `npy_path`, `numpy_array`). Folder and video inputs raise an explicit `ValueError`.

### Macro flow

```
__init__(config)
   └── _dump_model()
         ├── onnx.load(model_path) + shape_inference.infer_shapes
         ├── identifies every intermediate tensor name in graph.node
         ├── appends helper.make_tensor_value_info(name, dtype) to graph.output
         │     (initializers skipped unless debug_dump_initializers: true)
         ├── computes topological _layer_order
         ├── onnx.save(dump_model.onnx) in output_path
         ├── sets config["weight_path"] = temp dump_model.onnx
         └── ModelFactory.create(debug_mode=True)
               └── BaseModel enables ORT profiling: sess_options.enable_profiling = True

run(single_input)
   ├── preprocessor(single_input)
   ├── model(preprocessed_data)
   │     ├── session.run(None, input_feed)                # executes instrumented graph
   │     └── session.end_profiling()                      # writes ORT profiler JSON to output_path
   ├── postprocessor(result_1, single_input)              # saves visual .jpg
   ├── stream intermediate layers:
   │     ├── removes stale *.json in intermediate_outputs/
   │     └── for each layer: _layer_order from _dump_model()
   │           (leftover session outputs appended sorted; crnn keeps native session-output order)
   │           ├── extracts value array and deletes from memory (prevents OOM)
   │           ├── sanitizes layer name (regex + truncation to 200 chars + md5 on collision)
   │           └── writes intermediate_outputs/NNNN__<sanitized_layer>.json
   └── cleanup:
         └── removes temp dump_model.onnx from disk
```

*Code pointers*: Graph mutation in `_dump_model()`, disk-streaming in `run()`, profiling setup in `AI_Script/models/base_model.py` → `_session_option()`.

*Layer ordering detail* → [[pipeline/debug-internals|Debug Internals]].

---

## 4. Evaluate Mode (`Pipeline_Evaluate`)

- **Trigger**: `evaluate_mode: true`.
- **File**: `AI_Script/core/pipeline_evaluate.py`.
- **Constraints**: Folder input only (`folder_path`). The directory must contain `images/` and `labels/` subdirectories.

### Macro flow

```
run(dataset_folder)
   ├── check_file(dataset_folder) == 'folder_path'
   ├── initializes final_results dictionary (Model_Information, Dataset_Information, Performance, Evaluate)
   ├── EvaluateFactory.create(config)                     # resolved by config["task"]
   ├── start_time = time.time()
   │
   ├── for each img in os.listdir(dataset_folder/images):
   │     ├── preprocessor(images/<img>) -> model(...)     # raw prediction
   │     ├── label_path = labels/<img_stem>.txt
   │     ├── evaluate.benchmark_single(pred, img_path, label_path)
   │     │     # task-specific: detection computes AP, cls computes match, HTR decodes text
   │     └── final_results["Evaluate"]["images"][img] = s_evaluate
   │
   ├── end_time = time.time(), elapsed = end_time - start_time
   ├── computes throughput_fps = len(list_images) / elapsed
   ├── evaluate.benchmark_toltal(final_results["Evaluate"])
   │     # reduces per-image metrics to overall_metrics (avg mAP, macro/micro/weighted avg, accuracy)
   │     └── final_results["Evaluate"]["overall_metrics"].update(...)
   │
   └── saves JSON to:
         output_path/evaluate - <model_name> - <precision_format> - CPU - <timestamp>.json
```

*Code pointers*:
- Task routing: `AI_Script/evaluate/factory_evaluate.py` → `EvaluateRegistry.get(task)`.
- Metric calculation: `AI_Script/evaluate/tasks/detection.py` (`detection_benchmark`), `classification.py` (`classification_benchmark`), `HTR.py` (`htr_benchmark`).

---

## Comparison of Mode Properties

| Mode | Input accepted | Multi-threading | Temporary files created | Profiling enabled | Primary artifact |
|---|---|---|---|---|---|
| Inference | image / folder / video / npy | No (serial per frame/image) | None | No | Annotated `.jpg` or `.mkv` |
| Tracking | video (or single image in memory) | Optional (`inference_workers > 1`) | None | No | Annotated `.mkv` with IDs |
| Debug | single item: `image_path` / `npy_path` / `numpy_array` | No | `dump_model.onnx` (auto-removed) | Yes (`session.end_profiling`) | `intermediate_outputs/*.json` + profiler JSON + `.jpg` |
| Evaluate | folder (`images/` + `labels/`) | No | None | No | `evaluate - <model> - ... .json` |

## Related Notes

- [[project/overview|Overview]] — how `_get_pipeline()` chooses between these pipelines
- [[project/features|Features]] — options controlling each mode (`inference_workers`, `debug_dump_initializers`)
- [[pipeline/debug-internals|Debug Internals]] — deep dive: graph rewrite, topological order, disk streaming, profiling, cleanup
- [[pipeline/output-schemas|Output Schemas]] — the exact content of files written by each mode
