from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class StageType(str, Enum):
    PREPROCESS = "preprocess"
    INFERENCE = "inference"
    ADAPTER = "adapter"
    POSTPROCESS = "postprocess"
    BENCHMARK = "benchmark"
    EVALUATION = "evaluation"


class PipelineConfig(BaseModel):
    stages: List[StageType] = Field(
        default_factory=lambda: [
            StageType.PREPROCESS,
            StageType.INFERENCE,
            StageType.ADAPTER,
            StageType.POSTPROCESS,
        ],
        description="Pipeline stages to execute"
    )
    
    # Benchmark config
    warmup_runs: Optional[int] = Field(10, description="Warmup iterations")
    benchmark_runs: Optional[int] = Field(100, description="Benchmark iterations")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True