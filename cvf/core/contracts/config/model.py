from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from pathlib import Path


class ModelConfig(BaseModel):
    family: str = Field(..., description="Model family: yolo, detr, resnet, crnn")
    variant: str = Field(..., description="Model variant: yolov5, yolov12, resnet50, lenet5, crnn")
    path: str = Field(..., description="Path to model file (.onnx, .pt)")
    
    # Optional preprocessing overrides
    target_size: Optional[List[int]] = Field(None, description="Target size [H, W]")
    mean: Optional[List[float]] = Field(None, description="Normalization mean")
    std: Optional[List[float]] = Field(None, description="Normalization std")
    
    # Model-specific config
    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra model-specific config")
    
    class Config:
        extra = "allow"