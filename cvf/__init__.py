"""CVF - Computer Vision Flow.

A config-driven, modular, and extensible Computer Vision execution framework.
"""
from cvf.core import (
    load_config,
    Pipeline,
    PipelineContext,
    CompatibilityResolver,
    CVFConfig,
    ImageInput,
    VideoInput,
    TensorInput,
    DetectionOutput,
    ClassificationOutput,
    SegmentationOutput,
)

__version__ = "1.0.0"

__all__ = [
    "load_config",
    "Pipeline",
    "PipelineContext",
    "CompatibilityResolver",
    "CVFConfig",
    "ImageInput",
    "VideoInput",
    "TensorInput",
    "DetectionOutput",
    "ClassificationOutput",
    "SegmentationOutput",
]