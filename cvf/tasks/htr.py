"""HTR (Handwritten Text Recognition) task - postprocess."""
from typing import Any, Dict, Optional
from dataclasses import dataclass

from cvf.core.contracts.runtime.output import CanonicalOutput


@dataclass
class HTRConfig:
    """HTR-specific configuration."""
    pass


class HTRPostprocessor:
    """HTR postprocessing - passes through the decoded text."""
    
    def __init__(self, config: Optional[HTRConfig] = None):
        self.config = config or HTRConfig()
    
    def __call__(self, output: CanonicalOutput, task_config) -> CanonicalOutput:
        """Pass through HTR output (already decoded by adapter)."""
        return output


def create_htr_postprocess(config: Optional[Dict] = None) -> HTRPostprocessor:
    """Factory for HTR postprocessor."""
    return HTRPostprocessor()