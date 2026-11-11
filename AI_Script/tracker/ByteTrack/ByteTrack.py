from AI_Script.tracker.ByteTrack.tracker.byte_tracker import BYTETracker
from AI_Script.postprocess.Functions.Boxes_Steps import NMS
import numpy as np

class ByteTrackArgs:
    def __init__(self):
        self.track_thresh = 0.3
        self.track_buffer = 80 # time store track
        self.match_thresh = 0.7 # IOU new box and old box
        self.aspect_ratio_thresh = 60
        self.min_box_area = 1
        self.mot20 = True

class bytetrack:
    def __init__(self, frame_rate=25, conf_threshold=0.5, iou_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Initialize ByteTracker
        self.args = ByteTrackArgs()
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    def preprocess_box(self, input_data, width, height):
        # Load raw outputs
        boxes = input_data["boxes"]
        conf_scores = input_data["conf"]
        class_ids = input_data["class"]

        # Filter detections with low confidence (e.g., score > 0.5)
        mask = conf_scores > self.conf_threshold
        boxes = boxes[mask]
        conf_scores = conf_scores[mask]
        class_ids = class_ids[mask]

        # Apply NMS
        if len(boxes) > 0:
            keep = NMS(boxes, conf_scores, iou_threshold=self.iou_threshold)
            boxes = boxes[keep]
            conf_scores = conf_scores[keep]
            class_ids = class_ids[keep]

        # # scale box to real size image
        # scale_w = width / 640
        # scale_h = height / 640
        # boxes[:, [0, 2]] *= scale_w
        # boxes[:, [1, 3]] *= scale_h

        detections = []
        for box, score, cls in zip(boxes, conf_scores, class_ids):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])

        # Convert to numpy array
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        return detections

    def update(self, input_data, width, height):
        detections = self.preprocess_box(input_data, width, height)
        return self.tracker.update(detections, [height, width], [height, width])

