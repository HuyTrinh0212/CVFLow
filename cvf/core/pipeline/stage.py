"""Pipeline stage base class."""
from abc import ABC, abstractmethod
from typing import Any
from cvf.core.pipeline.context import PipelineContext


class Stage(ABC):
    """Base class for pipeline stages."""
    
    name: str = "base"
    
    def __init__(self, config: Any = None):
        self.config = config
    
    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage."""
        pass
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        context.start_stage(self.name)
        try:
            result = self.run(context)
        finally:
            context.end_stage(self.name)
        return result