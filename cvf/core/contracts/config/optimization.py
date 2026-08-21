from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class OptimizationType(str, Enum):
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    GRAPH_OPTIMIZATION = "graph_optimization"


class OptimizationConfig(BaseModel):
    type: OptimizationType = Field(..., description="Optimization type")
    
    # Optimization-specific config
    config: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")
    
    # Output
    output_path: Optional[str] = Field(None, description="Output path for optimized model")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True