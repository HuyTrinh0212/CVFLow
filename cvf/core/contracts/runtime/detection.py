from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class DetectionOutput:
    """Canonical detection output - model agnostic."""
    boxes: np.ndarray          # [N, 4] in xyxy format
    scores: np.ndarray         # [N] confidence scores
    class_ids: np.ndarray      # [N] class indices
    image_shape: Tuple[int, int]  # (H, W) original image shape
    class_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError(f"boxes must be [N, 4], got {self.boxes.shape}")
        if self.scores.ndim != 1:
            raise ValueError(f"scores must be [N], got {self.scores.shape}")
        if self.class_ids.ndim != 1:
            raise ValueError(f"class_ids must be [N], got {self.class_ids.shape}")
        n = len(self.boxes)
        if len(self.scores) != n or len(self.class_ids) != n:
            raise ValueError(f"boxes, scores, class_ids must have same length: {n}, {len(self.scores)}, {len(self.class_ids)}")
    
    def __len__(self) -> int:
        return len(self.boxes)
    
    def filter(self, mask: np.ndarray) -> "DetectionOutput":
        """Return filtered detection output."""
        return DetectionOutput(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            class_ids=self.class_ids[mask],
            image_shape=self.image_shape,
            class_names=self.class_names,
        )
    
    def to_dict(self) -> dict:
        return {
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "class_ids": self.class_ids.tolist(),
            "image_shape": self.image_shape,
            "class_names": self.class_names,
        }