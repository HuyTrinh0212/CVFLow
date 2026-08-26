from pydantic import BaseModel, Field, model_validator
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
    input: Optional[str] = Field(None, description="Input image/video path")
    output_dir: Optional[str] = Field(None, description="Output directory - required (use output_dir, output, or output_path)")
    seed: Optional[int] = Field(42, description="Random seed")
    verbose: Optional[bool] = Field(True, description="Verbose logging")
    
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def _alias_output(cls, data: Any) -> Any:
        """Accept `output` / `output_path` as alias for `output_dir`."""
        if isinstance(data, dict) and "output_dir" not in data:
            for alias in ("output", "output_path", "save_dir"):
                if alias in data:
                    data["output_dir"] = data.pop(alias)
                    break
        return data

    @model_validator(mode="after")
    def _resolve_alias_after(self):
        """Validate output_dir required; also adopt Pydantic extra alias if needed."""
        # If output_dir missing but extra alias present (e.g., via unknown field), adopt it
        # Note: when output_dir is required, missing will already have raised ValidationError before reaching here
        # This handles empty-string and extra-alias fallback
        if not self.output_dir or not str(self.output_dir).strip():
            extra = getattr(self, "model_extra", None) or getattr(self, "__pydantic_extra__", None) or {}
            for alias in ("output", "output_path", "save_dir"):
                if alias in extra and str(extra[alias]).strip():
                    self.output_dir = str(extra[alias])
                    break
        if not self.output_dir or not str(self.output_dir).strip():
            raise ValueError(
                "Missing required field 'output_dir' (aliases: 'output', 'output_path'). "
                "Please set output_dir in your YAML, e.g.: output_dir: ~/Downloads"
            )
        return self