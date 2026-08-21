from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class BackendType(str, Enum):
    ONNXRUNTIME = "onnxruntime"
    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TFLITE = "tflite"


class BackendConfig(BaseModel):
    type: BackendType = Field(..., description="Inference backend")
    
    # Backend-specific config
    providers: Optional[List[str]] = Field(None, description="Execution providers (ONNX Runtime)")
    device_id: Optional[int] = Field(0, description="GPU device ID")
    
    # Optimization options
    graph_optimization: Optional[str] = Field("all", description="Graph optimization level")
    enable_profiling: Optional[bool] = Field(False, description="Enable profiling")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True