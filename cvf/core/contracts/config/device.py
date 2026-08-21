from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DeviceConfig(BaseModel):
    type: DeviceType = Field(..., description="Device type")
    device_id: Optional[int] = Field(0, description="Device ID for multi-GPU")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True