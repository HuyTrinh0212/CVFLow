# XAIP — Data Contract Across the Pipeline

**Summary**: The stable invariants that data satisfies at each module boundary — image format, preprocessor tensor layout, the detection adapter dict, box semantics, label lookup and evaluate label files — all verified in source.
**Tags**: #data-contract #project
**Created**: 2026-11-11
**Last Updated**: 2026-11-11

---

## Boundary 1 — raw image (internal color space: BGR)

- Anything read with OpenCV (`cv2.imread`) is **BGR**, everywhere (`AI_Script/postprocess/postprocessor/draw_box/draw_box.py` → `postprocess` reads BGR and saves BGR with `cv2.imwrite`).
- `.npy` / in-memory arrays are normalized defensively toward BGR before drawing: an `[H,W,3]` array is treated as RGB and flipped (`arr[..., ::-1]`), a 2-D array is expanded GRAY→BGR (verified in `draw_box.py` → `postprocess`, input branch).
- Detection preprocessors flip BGR→RGB at the end of the pixel pipeline (`AI_Script/preprocess/detection/yolo.py`, `detr.py`), so the **model** sees RGB while the **drawing/eval side** stays BGR. Any new consumer must respect this split.

## Boundary 2 — preprocessor → model tensor

- Every preprocessor emits `np.ndarray` shaped `[1, C, H, W]` (`np.transpose(2,0,1)` + `expand_dims(0)`); C = 3 for detection/ResNet/ViT, 1 for LeNet5/CRNN. Per-preprocessor pixel math table → [[project/models|Models]] § Per-model preprocess specifics.
- The FP16 cast (when `input_dtype` resolves to `"fp16"`) is applied **after** the concrete `preprocess()` returns, in `AI_Script/preprocess/base_preprocess.py` → `BasePreprocessor.__call__`. Individual preprocessors never need to know the precision label (exception: ViT branches on `precision_format` by design — see [[project/precision-quantization|Precision & Quantization]]).

## Boundary 3 — model → postprocess/evaluate

- `BaseModel.__call__` (`AI_Script/models/base_model.py`) forwards to `predict()`, and model classes are thin `session.run` wrappers returning the raw `List[np.ndarray]` from ONNX. No task interpretation happens in the model layer.
- For **detection**, the first transformation is `AI_Script/postprocess/Functions/Adapter_Detection.py` → `Adapter.__call__`, which converts the raw layout (YOLOv5 `[batch, N, 5+nc]`, optional `transpose_output`; RT-DETR `[1, 300, 84]`) into the dict contract documented in [[project/conventions|Conventions]] § Adapter dict keys: `boxes` (xyxy, pixels, `target_size` space) / `conf` / `class`.
- Boxes are **xywh (center format) until the adapter**, **xyxy afterwards**. `xywh_to_xyxy` lives in `AI_Script/postprocess/Functions/Boxes_Steps.py`; the evaluate label loader also converts its xywh ground truth through the same helper.

## Boundary 4 — the detection chain order (identical in all three consumers)

```
raw List[np.ndarray]
  → Adapter (xyxy, target_size space)
  → conf mask  (strictly greater: `conf_scores > conf_threshold`)
  → NMS        (torchvision.ops.nms via Boxes_Steps.NMS, iou_threshold)
  → unletterbox (map target_size space → original image pixels)
  → draw / track / metric match
```

Verified identical order in `draw_box.py` → `postprocess`, the tracking wrapper `AI_Script/tracker/ByteTrack/ByteTrack.py` → `preprocess_box` (called per frame by `AI_Script/core/pipeline_tracking.py`), and `AI_Script/evaluate/tasks/detection.py` → `_adapter_boxes`. `unletterbox` semantics (verified in `AI_Script/postprocess/Functions/Boxes_Steps.py`):

- Accepts `(N,4)` or `(4,)`; treats boxes ≤ 1.0 as normalized and rescales them to `target_size` first.
- `resize_mode="letterbox"` (default): removes the centered 114-pad with Ultralytics-compatible rounding (`round(pad/2 - 0.1)`, matching the forward letterbox in `AI_Script/preprocess/detection/yolo.py` → `letterbox`), then divides by the aspect ratio `r = min(tgt/orig)`.
- `resize_mode="stretch"`: per-axis inverse scale `orig/tgt` on each coordinate pair.
- Clips to image bounds and returns int32 rounded pixels.

## Boundary 5 — label lookup

- Text label for a class id is always `datasets.json[dataset][str(class_id)]` — the key is a **string**, not an int. Verified in `AI_Script/postprocess/postprocessor/get_labels/get_labels.py` (classification) and `draw_box.py` (detection). Label sets inventory → [[project/models|Models]] § Label sets.

## Boundary 6 — evaluate label files (per task)

| Task | Label file format | Read by |
|---|---|---|
| `object_detection` | One object per line: `class_id xc yc w h` — center xywh, **normalized 0–1** | `AI_Script/evaluate/tasks/detection.py` → `_load_labels` (converts to xyxy, scales by real image width/height) |
| `classification` | Single integer (the class id) | `AI_Script/evaluate/tasks/classification.py` → `_get_labels` |
| `htr` | Plain text (the ground-truth string) | `AI_Script/evaluate/tasks/HTR.py` → `_get_labels` |

Per-task metric aggregation and the JSON schemas built from these files → [[pipeline/output-schemas|Output Schemas]].

## Known asymmetries (verified in code — keep in mind when interpreting results)

- **`resize_mode` is not honored everywhere**: `draw_box.py` passes `resize_mode` into `unletterbox`, but the evaluate route (`detection.py` → `_adapter_boxes`) and the tracking pipeline (`AI_Script/core/pipeline_tracking.py` → `_draw_box_id`) call `unletterbox` **without** `resize_mode`, so they always apply letterbox math. For a `stretch`-preprocessed run, evaluate/tracking reverse-mapping is therefore geometrically wrong (the inference `.jpg` visual is fine). Fix direction exists in code; re-check these call sites before relying on any statement here.
- `transpose_output` is honored only by the `draw_box` path — the tracking and evaluate routes build their own `Adapter` without it (detail in [[project/models|Models]] § notes).

## Related Notes

- [[project/conventions|Conventions]] — the adapter dict as an API contract
- [[project/models|Models]] — per-preprocessor pixel math and registered keys
- [[project/precision-quantization|Precision & Quantization]] — the FP16 cast layer and ViT's precision-dependent branch
- [[pipeline/mode-flows|Mode Flows]] — where each boundary sits in every mode
- [[pipeline/output-schemas|Output Schemas]] — what the evaluate chains produce
