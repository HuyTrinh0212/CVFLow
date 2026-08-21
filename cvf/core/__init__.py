"""CVF Core Package."""
from cvf.core.config import load_config, CVFConfig
from cvf.core.contracts import (
    ImageInput, VideoInput, TensorInput,
    ModelOutput, CanonicalOutput,
    DetectionOutput, ClassificationOutput, SegmentationOutput,
)
from cvf.core.compatibility import CompatibilityResolver, CompatibilityError
from cvf.core.registry import (
    model_registry, task_registry, backend_registry, device_registry,
    adapter_registry, optimizer_registry, preprocess_registry, postprocess_registry,
)
from cvf.core.pipeline import Pipeline, PipelineContext, Stage
from cvf.core.artifacts import ArtifactsManager, artifacts_manager

__all__ = [
    # Config
    "load_config",
    "CVFConfig",
    # Contracts
    "ImageInput", "VideoInput", "TensorInput",
    "ModelOutput", "CanonicalOutput",
    "DetectionOutput", "ClassificationOutput", "SegmentationOutput",
    # Compatibility
    "CompatibilityResolver", "CompatibilityError",
    # Registry
    "model_registry", "task_registry", "backend_registry", "device_registry",
    "adapter_registry", "optimizer_registry", "preprocess_registry", "postprocess_registry",
    # Pipeline
    "Pipeline", "PipelineContext", "Stage",
    # Artifacts
    "ArtifactsManager", "artifacts_manager",
]