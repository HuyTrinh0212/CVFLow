from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class TaskType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    HTR = "htr"
    POSE = "pose"


class TaskConfig(BaseModel):
    type: TaskType = Field(..., description="Task type")
    
    # Task-specific config
    conf_threshold: Optional[float] = Field(0.3, description="Confidence threshold")
    iou_threshold: Optional[float] = Field(0.5, description="IoU threshold for NMS")
    top_k: Optional[int] = Field(5, description="Top-K for classification")
    
    # Dataset info
    dataset: Optional[str] = Field(None, description="Dataset name for class labels")
    class_names: Optional[List[str]] = Field(None, description="Class names")
    
    # Extra task config
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True