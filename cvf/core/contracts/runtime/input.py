from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class ImageInput:
    """Single image input - path or numpy array."""
    path: Optional[str] = None
    array: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.path is None and self.array is None:
            raise ValueError("Either path or array must be provided")
        if self.path is not None and self.array is not None:
            raise ValueError("Only one of path or array should be provided")


@dataclass
class VideoInput:
    """Video file input."""
    path: str
    fps: Optional[int] = None
    frame_limit: Optional[int] = None


@dataclass
class TensorInput:
    """Preprocessed tensor input ready for inference."""
    data: np.ndarray
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchInput:
    """Batch of inputs for batched inference."""
    items: list
    batch_size: int = 1
    
    def __post_init__(self):
        self.batch_size = len(self.items)