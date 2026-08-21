from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class ClassificationOutput:
    """Canonical classification output - model agnostic."""
    logits: np.ndarray           # [C] or [1, C] raw logits
    probabilities: np.ndarray    # [C] or [1, C] softmax probabilities
    class_ids: np.ndarray        # [K] top-K class indices
    scores: np.ndarray           # [K] top-K scores
    class_names: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.logits.ndim == 2 and self.logits.shape[0] == 1:
            self.logits = self.logits.squeeze(0)
        if self.probabilities.ndim == 2 and self.probabilities.shape[0] == 1:
            self.probabilities = self.probabilities.squeeze(0)
    
    @property
    def top1_id(self) -> int:
        return int(self.class_ids[0]) if len(self.class_ids) > 0 else -1
    
    @property
    def top1_score(self) -> float:
        return float(self.scores[0]) if len(self.scores) > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "logits": self.logits.tolist(),
            "probabilities": self.probabilities.tolist(),
            "class_ids": self.class_ids.tolist(),
            "scores": self.scores.tolist(),
            "class_names": self.class_names,
        }