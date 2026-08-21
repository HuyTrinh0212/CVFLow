"""Pipeline package."""
from cvf.core.pipeline.pipeline import Pipeline
from cvf.core.pipeline.context import PipelineContext
from cvf.core.pipeline.stage import Stage

__all__ = ["Pipeline", "PipelineContext", "Stage"]