from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class SegmentationOutput:
    """Canonical segmentation output - model agnostic."""
    masks: np.ndarray            # [N, H, W] binary or probability masks
    scores: np.ndarray           # [N] confidence scores per mask
    class_ids: np.ndarray        # [N] class indices per mask
    image_shape: Tuple[int, int] # (H, W) original image shape
    class_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.masks.ndim != 3:
            raise ValueError(f"masks must be [N, H, W], got {self.masks.shape}")
        if self.scores.ndim != 1:
            raise ValueError(f"scores must be [N], got {self.scores.shape}")
        if self.class_ids.ndim != 1:
            raise ValueError(f"class_ids must be [N], got {self.class_ids.shape}")
        n = len(self.masks)
        if len(self.scores) != n or len(self.class_ids) != n:
            raise ValueError(f"masks, scores, class_ids must have same length: {n}, {len(self.scores)}, {len(self.class_ids)}")
    
    def __len__(self) -> int:
        return len(self.masks)
    
    def to_dict(self) -> dict:
        return {
            "masks": self.masks.tolist(),
            "scores": self.scores.tolist(),
            "class_ids": self.class_ids.tolist(),
            "image_shape": self.image_shape,
            "class_names": self.class_names,
        }