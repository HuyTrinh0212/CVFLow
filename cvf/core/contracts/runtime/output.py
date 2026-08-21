from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class ModelOutput:
    """Raw model output before adapter conversion."""
    data: Any
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CanonicalOutput:
    """Base canonical output after adapter conversion."""
    pass