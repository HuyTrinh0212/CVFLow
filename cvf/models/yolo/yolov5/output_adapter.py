"""YOLOv5 output adapter - converts raw ONNX output to canonical DetectionOutput."""
import numpy as np
from typing import Any, Dict, Optional
from cvf.core.contracts.runtime.detection import DetectionOutput


class YOLOv5DetectionAdapter:
    """Adapter for YOLOv5 raw output -> canonical DetectionOutput."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.transpose_output = self.config.get("transpose_output", False)
    
    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from xywh to xyxy format."""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def adapt(self, raw_output: Any) -> DetectionOutput:
        """
        Convert raw YOLOv5 output to canonical DetectionOutput.
        
        Args:
            raw_output: List of numpy arrays from ONNX Runtime, typically [1, N, 5+num_classes]
            
        Returns:
            DetectionOutput with boxes (xyxy), scores, class_ids
        """
        # Handle different output formats
        if isinstance(raw_output, (list, tuple)):
            outputs = raw_output[0]
        else:
            outputs = raw_output
        
        # Transpose if needed (some ONNX exports have different layouts)
        if self.transpose_output and outputs.ndim == 3:
            outputs = np.transpose(outputs, (0, 2, 1))
        
        # Cast FP16 to FP32 to avoid overflow in NMS/postprocess (onnxruntime returns float16 for FP16 model)
        if isinstance(outputs, np.ndarray) and outputs.dtype == np.float16:
            outputs = outputs.astype(np.float32)
        # Squeeze batch dimension
        prediction = outputs.squeeze(0)
        
        # Boxes: (x_center, y_center, width, height) - first 4 columns
        boxes = prediction[:, :4]
        boxes_xyxy = self._xywh_to_xyxy(boxes)
        
        # Objectness score
        objectness = prediction[:, 4:5]
        
        # Class scores (after objectness)
        class_scores = prediction[:, 5:]
        
        # Apply sigmoid to class scores if needed (depends on export)
        if self.config.get("apply_sigmoid", True):
            class_scores = self._sigmoid(class_scores)
        
        # Get max class score and class id for each prediction
        max_class_scores = np.max(class_scores, axis=1, keepdims=True)
        class_ids = np.argmax(class_scores, axis=1, keepdims=True)
        
        # Final confidence = objectness * max_class_score
        conf = objectness * max_class_scores
        
        return DetectionOutput(
            boxes=boxes_xyxy,
            scores=conf.squeeze(1),
            class_ids=class_ids.squeeze(1).astype(np.int32),
            image_shape=self.config.get("image_shape", (640, 640)),
        )


def create_adapter(config: Optional[Dict] = None) -> YOLOv5DetectionAdapter:
    """Factory function to create YOLOv5 detection adapter."""
    return YOLOv5DetectionAdapter(config)