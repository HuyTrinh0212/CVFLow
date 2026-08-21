from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from cvf.core.contracts.config.model import ModelConfig
from cvf.core.contracts.config.task import TaskConfig
from cvf.core.contracts.config.backend import BackendConfig
from cvf.core.contracts.config.device import DeviceConfig
from cvf.core.contracts.config.pipeline import PipelineConfig
from cvf.core.contracts.config.optimization import OptimizationConfig


class CVFConfig(BaseModel):
    """Root CVF configuration."""
    model: ModelConfig
    task: TaskConfig
    backend: BackendConfig
    device: DeviceConfig
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    optimization: Optional[OptimizationConfig] = Field(None, description="Optimization config")
    
    # Global options
    output_dir: Optional[str] = Field("outputs", description="Output directory")
    seed: Optional[int] = Field(42, description="Random seed")
    verbose: Optional[bool] = Field(True, description="Verbose logging")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"