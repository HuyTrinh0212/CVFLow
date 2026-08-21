"""Tasks package."""
from cvf.tasks.detection import (
    DetectionPostprocessor,
    DetectionMetrics,
    create_detection_postprocess,
    create_detection_metrics,
)
from cvf.tasks.classification import (
    ClassificationPostprocessor,
    ClassificationMetrics,
    create_classification_postprocess,
    create_classification_metrics,
)
from cvf.tasks.htr import (
    HTRPostprocessor,
    create_htr_postprocess,
)

__all__ = [
    "DetectionPostprocessor",
    "DetectionMetrics",
    "create_detection_postprocess",
    "create_detection_metrics",
    "ClassificationPostprocessor",
    "ClassificationMetrics",
    "create_classification_postprocess",
    "create_classification_metrics",
    "HTRPostprocessor",
    "create_htr_postprocess",
]