"""CVF CLI - command line interface."""
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cvf.core import load_config, Pipeline, CompatibilityResolver
from cvf.core.artifacts import artifacts_manager
from cvf.plugins.builtin import register_builtin


# COCO 80 class names for YOLO visualization
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def _resolve_output_dir(config) -> Path:
    """Strictly resolve output_dir — ERROR if missing, no silent fallback."""
    raw = getattr(config, "output_dir", None)
    if not raw or not str(raw).strip():
        extra = getattr(config, "model_extra", None) or getattr(config, "__pydantic_extra__", None) or {}
        for alias in ("output", "output_path", "save_dir"):
            if alias in extra and str(extra[alias]).strip():
                raw = extra[alias]
                print(f"[config] Using alias '{alias}' as output_dir: {raw}")
                break
    if not raw or not str(raw).strip():
        raise ValueError(
            "Missing required field 'output_dir' (aliases: 'output', 'output_path'). "
            "Set it in YAML, e.g.: output_dir: ~/Downloads  — refusing to silently use 'outputs/'"
        )
    path = Path(str(raw)).expanduser()
    if path.exists() and path.is_file():
        raise NotADirectoryError(f"output_dir '{path}' exists and is a file, not a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _draw_detections_on_frame(frame: np.ndarray, detections) -> np.ndarray:
    """Draw detections on a BGR frame via letterbox to target shape; returns vis BGR frame."""
    h0, w0 = frame.shape[:2]
    target_h, target_w = detections.image_shape if hasattr(detections, "image_shape") else (640, 640)
    r = min(target_h / h0, target_w / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2
    resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    vis = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    for box, score, cls_id in zip(detections.boxes, detections.scores, detections.class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        name = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else str(cls_id)
        if detections.class_names and 0 <= cls_id < len(detections.class_names):
            name = detections.class_names[cls_id]
        label = f"{name} {float(score):.2f}"
        color = (0, 255, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def _save_detection_vis(image_path: str, detections, output_path: Path) -> Optional[Path]:
    """Draw detection boxes on image and save visualization."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [vis] Could not load image for visualization: {image_path}")
            return None

        h0, w0 = img.shape[:2]
        # Resize with letterbox logic to match model input (640) so boxes align
        # For simplicity, draw on 640x640 letterboxed copy (what model saw)
        # Re-use YOLO letterbox padding color (114,114,114) if possible
        target_h, target_w = detections.image_shape if hasattr(detections, "image_shape") else (640, 640)
        # Simple resize to target for visualization (boxes are in target space)
        # Letterbox the original to target so coordinates match
        r = min(target_h / h0, target_w / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = (target_w - new_unpad[0]) / 2, (target_h - new_unpad[1]) / 2
        if (w0, h0) != (new_unpad[1], new_unpad[0]) and False:
            pass
        # Use simple resize to target for vis (good enough; keeps aspect via letterbox)
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        vis = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Draw boxes (boxes are in target xyxy)
        for box, score, cls_id in zip(detections.boxes, detections.scores, detections.class_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls_id)
            name = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else str(cls_id)
            if detections.class_names and 0 <= cls_id < len(detections.class_names):
                name = detections.class_names[cls_id]
            label = f"{name} {float(score):.2f}"
            color = (0, 255, 0)  # green for all; could hash by cls_id
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)
        return output_path
    except Exception as e:
        print(f"  [vis] Failed to save visualization: {e}")
        return None


def _save_outputs(context, input_path: str, output_dir: Path) -> None:
    """Persist task result, timings and visualization to output_dir."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive stem from input path if file, else generic
        stem = "result"
        if isinstance(input_path, str):
            p = Path(input_path)
            if p.exists() and p.is_file():
                stem = p.stem
            else:
                # fallback: sanitize string
                stem = p.stem or "result"
        elif isinstance(input_path, (list, tuple)) and input_path:
            stem = Path(str(input_path[0])).stem

        # Build serializable result
        result_dict = None
        vis_path = None
        if context.task_result is not None:
            tr = context.task_result
            if hasattr(tr, "to_dict"):
                result_dict = tr.to_dict()
            elif hasattr(tr, "__dict__"):
                try:
                    result_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in tr.__dict__.items()}
                except Exception:
                    result_dict = str(tr)
            else:
                result_dict = str(tr)

            # Visualization for detection
            from cvf.core.contracts.runtime.detection import DetectionOutput
            if isinstance(tr, DetectionOutput) and isinstance(input_path, str) and Path(input_path).exists():
                vis_path = output_dir / f"{stem}_vis.jpg"
                saved = _save_detection_vis(input_path, tr, vis_path)
                if saved:
                    print(f"  Visualization saved: {saved}")

        # Combined JSON
        payload = {
            "input": str(input_path),
            "model": {
                "family": context.config.model.family,
                "variant": context.config.model.variant,
                "path": context.config.model.path,
            },
            "task": context.config.task.model_dump() if hasattr(context.config.task, "model_dump") else str(context.config.task),
            "result": result_dict,
            "stage_timings_ms": {k: v * 1000 for k, v in context.stage_timings.items()},
            "benchmark_metrics": context.benchmark_metrics,
            "evaluation_metrics": context.evaluation_metrics,
        }
        json_path = output_dir / f"{stem}_result.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Results saved: {json_path}")
        if vis_path and vis_path.exists():
            print(f"Output dir: {output_dir.resolve()}")
    except Exception as e:
        print(f"Warning: failed to save outputs: {e}")
        import traceback
        traceback.print_exc()


def _run_video_inference(config, input_path: str, out_dir: Path) -> None:
    """Frame-by-frame video inference: saves vis video + per-frame JSON."""
    import cv2

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 120:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    pipeline = Pipeline(config)
    print(f"Running pipeline with stages: {[s.name for s in pipeline.stages]}")

    stem = Path(input_path).stem
    # Writer uses target size (letterboxed) so boxes align; fallback to 640 if unknown
    target_h, target_w = config.model.target_size or (640, 640)
    # target_size is [H,W]; writer expects (W,H)
    writer_size = (int(target_w), int(target_h))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    vis_path = out_dir / f"{stem}_vis.mp4"
    writer = cv2.VideoWriter(str(vis_path), fourcc, fps, writer_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {vis_path}")

    per_frame = []
    frame_idx = 0
    last_context = None
    # Aggregate timings
    agg_timings: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame is BGR uint8 HxWx3
        context = pipeline(frame)
        last_context = context
        tr = context.task_result
        # Accumulate timings
        for k, v in context.stage_timings.items():
            agg_timings[k] = agg_timings.get(k, 0.0) + v

        # JSON per frame
        if tr is not None and hasattr(tr, "to_dict"):
            fd = tr.to_dict()
        elif tr is not None and hasattr(tr, "__dict__"):
            fd = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in tr.__dict__.items()}
        else:
            fd = str(tr) if tr is not None else None
        per_frame.append({"frame": frame_idx, "result": fd, "timings_ms": {k: v * 1000 for k, v in context.stage_timings.items()}})
        # Draw & write
        try:
            from cvf.core.contracts.runtime.detection import DetectionOutput
            if isinstance(tr, DetectionOutput):
                vis = _draw_detections_on_frame(frame, tr)
                # Ensure vis matches writer_size
                if (vis.shape[1], vis.shape[0]) != writer_size:
                    vis = cv2.resize(vis, writer_size)
            else:
                # No detection vis; letterbox frame to writer_size for consistency
                vis = cv2.resize(frame, writer_size)
            writer.write(vis)
        except Exception as e:
            print(f"  [vis] frame {frame_idx} draw failed: {e}")
            # Fallback: resize frame
            try:
                writer.write(cv2.resize(frame, writer_size))
            except Exception:
                pass

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Processed {frame_idx}/{total} frames...")

    cap.release()
    writer.release()
    print(f"\n--- Video Pipeline Complete: {frame_idx} frames ---")
    if last_context:
        print(f"Last frame result: {last_context.task_result}")
        print("Avg stage timings (ms):")
        for k, v in agg_timings.items():
            print(f"  {k}: {(v/frame_idx)*1000:.2f}ms")

    # Save per-frame JSON
    payload = {
        "input": str(input_path),
        "model": {"family": config.model.family, "variant": config.model.variant, "path": config.model.path},
        "task": config.task.model_dump() if hasattr(config.task, "model_dump") else str(config.task),
        "frames": per_frame,
        "num_frames": frame_idx,
        "fps": fps,
        "avg_stage_timings_ms": {k: (v / frame_idx) * 1000 for k, v in agg_timings.items()} if frame_idx else {},
        "benchmark_metrics": last_context.benchmark_metrics if last_context else None,
    }
    json_path = out_dir / f"{stem}_result.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results saved: {json_path}")
    print(f"Visualization saved: {vis_path}")
    print(f"Output dir: {out_dir.resolve()}")


def run_inference(config_path: str, input_path: Optional[str] = None) -> None:
    """Run inference pipeline."""
    # Register builtin components
    register_builtin()
    
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Resolve input: CLI arg overrides config.input
    input_path = input_path or config.input
    if not input_path:
        raise ValueError(
            "No input provided. Set 'input' in the config YAML "
            "or pass an input path on the command line."
        )
    print(f"Input: {input_path}")
    
    # Validate compatibility
    CompatibilityResolver.validate(config)
    print("Config validation passed")
    
    # Resolve output_dir robustly (handles missing / alias / ~/ / empty)
    out_dir = _resolve_output_dir(config)
    artifacts_manager.output_dir = out_dir

    # Video path -> frame-by-frame pipeline
    if _is_video_path(str(input_path)):
        _run_video_inference(config, str(input_path), out_dir)
        return
    
    # Image / array path
    pipeline = Pipeline(config)
    print(f"Running pipeline with stages: {[s.name for s in pipeline.stages]}")
    
    context = pipeline(input_path)
    
    print("\n--- Pipeline Complete ---")
    print(f"Task result: {context.task_result}")
    if context.benchmark_metrics:
        print(f"Benchmark: {context.benchmark_metrics}")
    if context.evaluation_metrics:
        print(f"Evaluation: {context.evaluation_metrics}")
    
    print(f"\nStage timings:")
    for stage, timing in context.stage_timings.items():
        print(f"  {stage}: {timing*1000:.2f}ms")

    # Save outputs to output_dir (re-resolve in case validator didn't fire)
    _save_outputs(context, input_path, out_dir)


def run_benchmark(config_path: str) -> None:
    """Run benchmark only."""
    register_builtin()
    
    print(f"Loading benchmark config: {config_path}")
    config = load_config(config_path)
    CompatibilityResolver.validate(config)
    
    out_dir = _resolve_output_dir(config)
    artifacts_manager.output_dir = out_dir
    
    # Build dummy input from config (target size + channels)
    import numpy as np
    h, w = config.model.target_size or (640, 640)
    grayscale_variants = {"lenet5", "crnn"}
    channels = 1 if config.model.variant in grayscale_variants else 3
    dummy_input = np.random.randn(1, channels, h, w).astype(np.float32)
    
    pipeline = Pipeline(config)
    context = pipeline(dummy_input)
    
    print("\n--- Benchmark Complete ---")
    if context.benchmark_metrics:
        print(f"Benchmark: {context.benchmark_metrics}")

    # Save benchmark payload
    _save_outputs(context, "dummy_benchmark_input", out_dir)


def main():
    parser = argparse.ArgumentParser(description="CVF - Computer Vision Flow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference pipeline")
    run_parser.add_argument("config", help="Path to config YAML")
    run_parser.add_argument(
        "input", nargs="?", default=None,
        help="Input image/video (overrides 'input' in config)",
    )
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("config", help="Path to config YAML")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_inference(args.config, args.input)
    elif args.command == "benchmark":
        run_benchmark(args.config)


if __name__ == "__main__":
    main()