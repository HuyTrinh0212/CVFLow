"""Detection task - schema, postprocess, metrics."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from cvf.core.contracts.runtime.detection import DetectionOutput
from cvf.core.contracts.config.task import TaskConfig


@dataclass
class DetectionConfig:
    """Detection-specific configuration."""
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    max_det: int = 100
    class_names: Optional[List[str]] = None


class DetectionPostprocessor:
    """Detection postprocessing: confidence filter, NMS, format results."""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
    
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        
        return np.array(keep, dtype=int)
    
    def __call__(self, output: DetectionOutput, task_config: TaskConfig) -> DetectionOutput:
        """Apply postprocessing to detection output."""
        # Filter by confidence
        mask = output.scores >= self.config.conf_threshold
        filtered = output.filter(mask)
        
        if len(filtered) == 0:
            return filtered
        
        # Apply NMS per class
        final_masks = np.zeros(len(filtered), dtype=bool)
        for class_id in np.unique(filtered.class_ids):
            class_mask = filtered.class_ids == class_id
            class_indices = np.where(class_mask)[0]
            class_boxes = filtered.boxes[class_mask]
            class_scores = filtered.scores[class_mask]
            
            keep = self._nms(class_boxes, class_scores, self.config.iou_threshold)
            final_masks[class_indices[keep]] = True
        
        result = filtered.filter(final_masks)
        
        # Limit max detections
        if len(result) > self.config.max_det:
            top_indices = np.argsort(result.scores)[::-1][:self.config.max_det]
            result = result.filter(np.zeros(len(result), dtype=bool))
            result = DetectionOutput(
                boxes=result.boxes[top_indices],
                scores=result.scores[top_indices],
                class_ids=result.class_ids[top_indices],
                image_shape=result.image_shape,
                class_names=result.class_names,
            )
        
        return result


class DetectionMetrics:
    """Detection evaluation metrics: mAP, precision, recall."""
    
    def __init__(self, iou_thresholds: Optional[np.ndarray] = None):
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes (xyxy format)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def compute_ap(self, preds: DetectionOutput, targets: DetectionOutput) -> Dict[str, float]:
        """Compute Average Precision."""
        # Simplified AP computation - can be extended
        results = {}
        
        for iou_thr in self.iou_thresholds:
            tp = 0
            fp = 0
            matched = set()
            
            for pred_box, pred_score, pred_cls in zip(preds.boxes, preds.scores, preds.class_ids):
                best_iou = 0
                best_idx = -1
                
                for idx, (tgt_box, tgt_cls) in enumerate(zip(targets.boxes, targets.class_ids)):
                    if idx in matched or tgt_cls != pred_cls:
                        continue
                    iou = self._iou(pred_box, tgt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_iou >= iou_thr and best_idx >= 0:
                    tp += 1
                    matched.add(best_idx)
                else:
                    fp += 1
            
            fn = len(targets) - len(matched)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"AP@{iou_thr:.2f}"] = precision  # Simplified
            results[f"Precision@{iou_thr:.2f}"] = precision
            results[f"Recall@{iou_thr:.2f}"] = recall
            results[f"F1@{iou_thr:.2f}"] = f1
        
        # mAP@0.5:0.95
        ap_values = [results[k] for k in results if k.startswith("AP@")]
        results["mAP@0.5:0.95"] = np.mean(ap_values) if ap_values else 0.0
        results["mAP@0.5"] = results.get("AP@0.50", 0.0)
        
        return results


def create_detection_postprocess(config: Optional[Dict] = None) -> DetectionPostprocessor:
    """Factory for detection postprocessor."""
    if config:
        # Filter out non-DetectionConfig fields
        det_config_dict = {k: v for k, v in config.items() 
                          if k in ['conf_threshold', 'iou_threshold', 'max_det', 'class_names']}
        det_config = DetectionConfig(**det_config_dict)
    else:
        det_config = DetectionConfig()
    return DetectionPostprocessor(det_config)


def create_detection_metrics(config: Optional[Dict] = None) -> DetectionMetrics:
    """Factory for detection metrics."""
    return DetectionMetrics()