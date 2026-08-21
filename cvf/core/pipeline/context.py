"""Pipeline context - carries data between stages."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import numpy as np
import time

from cvf.core.contracts.config.main import CVFConfig
from cvf.core.contracts.runtime.input import ImageInput, VideoInput, TensorInput
from cvf.core.contracts.runtime.output import ModelOutput
from cvf.core.contracts.runtime.detection import DetectionOutput
from cvf.core.contracts.runtime.classification import ClassificationOutput
from cvf.core.contracts.runtime.segmentation import SegmentationOutput


@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""
    
    # Config
    config: CVFConfig
    
    # Input
    input_data: Any = None  # ImageInput, VideoInput, TensorInput, etc.
    
    # Preprocessed tensor
    preprocessed: Optional[TensorInput] = None
    
    # Raw model output
    raw_output: Optional[ModelOutput] = None
    
    # Canonical output (after adapter)
    canonical_output: Any = None  # DetectionOutput, ClassificationOutput, etc.
    
    # Task result (after postprocess)
    task_result: Any = None
    
    # Metrics
    benchmark_metrics: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    
    # Timing
    stage_timings: Dict[str, float] = field(default_factory=dict)
    _stage_start: float = field(default=0.0, init=False)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Convenience properties for model/task info
    @property
    def model_family(self) -> str:
        return self.config.model.family
    
    @property
    def model_variant(self) -> str:
        return self.config.model.variant
    
    @property
    def task_type(self) -> str:
        tt = self.config.task.type
        return tt.value if hasattr(tt, 'value') else tt
    
    @property
    def backend_type(self) -> str:
        bt = self.config.backend.type
        return bt.value if hasattr(bt, 'value') else bt
    
    @property
    def device_type(self) -> str:
        dt = self.config.device.type
        return dt.value if hasattr(dt, 'value') else dt
    
    def start_stage(self, stage_name: str) -> None:
        """Mark stage start for timing."""
        self._stage_start = time.perf_counter()
        self.metadata["current_stage"] = stage_name
    
    def end_stage(self, stage_name: str) -> float:
        """Mark stage end and record timing."""
        elapsed = time.perf_counter() - self._stage_start
        self.stage_timings[stage_name] = elapsed
        return elapsed