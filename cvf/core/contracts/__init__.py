"""CVF Contracts Package."""
from cvf.core.contracts.runtime import (
    ImageInput, VideoInput, TensorInput, BatchInput,
    ModelOutput, CanonicalOutput,
    DetectionOutput, ClassificationOutput, SegmentationOutput,
)

__all__ = [
    "ImageInput", "VideoInput", "TensorInput", "BatchInput",
    "ModelOutput", "CanonicalOutput",
    "DetectionOutput", "ClassificationOutput", "SegmentationOutput",
]