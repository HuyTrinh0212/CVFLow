import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from AI_Script.models.factory_model import ModelFactory
from AI_Script.preprocess.factory_preprocess import PreprocessorFactory
from AI_Script.tracker.ByteTrack.ByteTrack import bytetrack
from AI_Script.postprocess.Functions.Adapter_Detection import Adapter
from AI_Script.postprocess.Functions.Boxes_Steps import unletterbox
from AI_Script.core.utils import check_file
import numpy as np
from datetime import datetime
import cv2

class Pipeline_Tracking:
    def __init__(self, config):
        self.config = config
        if not self.config.get("output_path") or not str(self.config.get("output_path")).strip():
            raise ValueError("'output_path' is required in config (no default).")
        self.output_path = os.path.abspath(str(self.config.get("output_path")).strip())
        os.makedirs(self.output_path, exist_ok=True)
        self.model_name = str(self.config.get("model_name"))
        self.target_size = tuple(self.config.get("target_size"))
        self.conf_threshold = float(self.config.get("conf_threshold"))
        self.iou_threshold = float(self.config.get("iou_threshold"))
        # Display flag - default False for headless (only save final video)
        display_val = self.config.get("display", self.config.get("display_option", False))
        self.display = bool(display_val) if isinstance(display_val, bool) else str(display_val).lower() in ("true", "1", "yes")

        # pre-process -> AI inference -> post-process
        self.preprocessor = PreprocessorFactory.create(config=config)
        self.model = ModelFactory.create(config=config)
        self.tracker = bytetrack(conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)

        # create adapter
        self.adapter = Adapter(name_model=self.model_name)

    def _process_single_item(self, item_source):
        preprocessed_data = self.preprocessor(item_source)
        model_output = self.model(preprocessed_data)
        return model_output

    def _infer_frame(self, frame):
        """Preprocess + inference + adapter for one frame (thread-safe: preprocessor,
        model session.run and adapter are all stateless; ORT session.run is thread-safe)."""
        return self.adapter(self.model(self.preprocessor(frame)))

    def _draw_box_id(self, frame, online_targets, original_shape):
        # Draw tracking results
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr
            bbox_unletterbox = unletterbox(np.array([bbox]), original_shape=original_shape, target_size=self.target_size)
            bbox_unletterbox = bbox_unletterbox.reshape(-1)
            x1, y1, x2, y2 = bbox_unletterbox[0], bbox_unletterbox[1], bbox_unletterbox[2], bbox_unletterbox[3]
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame,
                        f'ID: {track_id}',
                        (x1+2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1)

    def run(self, input_source, original_shape=None):
        input_type = check_file(input_source)

        # Case 1: single image
        if input_type in ['image_path', 'npy_path', 'numpy_array']:
            # AI model
            outputs = self._process_single_item(input_source)
            # Adapter
            dicts = self.adapter(outputs)
            # Tracker
            height, width = original_shape
            online_targets = self.tracker.update(dicts, width, height)
            # extract id and boxes
            track_id = []
            boxes = []
            for track in online_targets:
                # ID
                track_id.append(track.track_id)
                # Boxse
                bbox = track.tlbr
                bbox_unletterbox = unletterbox(np.array([bbox]), original_shape=original_shape, target_size=self.target_size)
                bbox_unletterbox = bbox_unletterbox.reshape(-1)
                boxes.append(bbox_unletterbox)
            return {'ID': track_id, 'boxes': boxes}
        # Case 2: video
        elif input_type == 'video_path':
            print("Start tracking...")
            # Open video
            cap = cv2.VideoCapture(input_source)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video FPS = {fps}")
            print(f"Width video = {width}")
            print(f"Height video = {height}")

            # Initialize video writer
            output_path = os.path.join(self.output_path, f"{self.model_name} video_tracking {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.mkv")
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Optional frame-parallel inference (config key: inference_workers).
            # Default 1 = original serial loop (identical behavior). >1 = preprocess +
            # inference + adapter run in parallel threads (frame order and ByteTrack
            # state preserved: results are consumed strictly in submission order,
            # tracker/draw/write stay sequential on the main thread).
            n_workers = int(self.config.get("inference_workers", 1) or 1)

            def _consume(dicts, frame):
                # Tracker
                online_targets = self.tracker.update(dicts, width, height)
                # Draw tracking results
                self._draw_box_id(frame, online_targets, original_shape=(height, width))
                # Display frame (optional, default False for headless)
                if self.display:
                    cv2.imshow('ByteTrack Original Implementation', frame)
                # Save frame (always)
                if output_path:
                    out.write(frame)
                # out loop if push 'q' (only when display is enabled)
                if self.display and (cv2.waitKey(1) & 0xFF == ord('q')):
                    return False
                return True

            if n_workers <= 1:
                # Loop video
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # AI model
                    outputs = self._process_single_item(frame)
                    # Adapter
                    dicts = self.adapter(outputs)
                    if not _consume(dicts, frame):
                        break
            else:
                # Frame-parallel: bounded-ahead thread pool keeps memory flat and
                # head-of-line result() guarantees ByteTrack sees frames in order.
                # Errors inside worker threads re-raise on .result() (fail-fast).
                max_ahead = n_workers * 2
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    pending = deque()
                    stop_read = False
                    while True:
                        while len(pending) < max_ahead and not stop_read:
                            ret, frame = cap.read()
                            if not ret:
                                stop_read = True
                                break
                            pending.append((pool.submit(self._infer_frame, frame), frame))
                        if not pending:
                            break
                        fut, cur_frame = pending.popleft()
                        dicts = fut.result()  # ordered (submission order = frame order)
                        if not _consume(dicts, cur_frame):
                            break

            # Cleanup
            cap.release()
            if output_path:
                out.release()
            if self.display:
                cv2.destroyAllWindows()

            # DONE
            print(f"Video are saved in {output_path}")

        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def __call__(self, input_source, original_shape=None):
        return self.run(input_source, original_shape=original_shape)