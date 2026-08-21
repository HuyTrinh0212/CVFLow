from cvf.core.contracts.runtime.input import ImageInput, VideoInput, TensorInput, BatchInput
from cvf.core.contracts.runtime.output import ModelOutput, CanonicalOutput
from cvf.core.contracts.runtime.detection import DetectionOutput
from cvf.core.contracts.runtime.classification import ClassificationOutput
from cvf.core.contracts.runtime.segmentation import SegmentationOutput

__all__ = [
    "ImageInput",
    "VideoInput",
    "TensorInput",
    "BatchInput",
    "ModelOutput",
    "CanonicalOutput",
    "DetectionOutput",
    "ClassificationOutput",
    "SegmentationOutput",
]