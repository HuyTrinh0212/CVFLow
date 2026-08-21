"""Postprocess stage - runs task-specific postprocessing."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.core.registry import postprocess_registry


class PostprocessStage(Stage):
    """Run task-specific postprocessing on canonical output."""
    
    name = "postprocess"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.canonical_output is None:
            raise ValueError("Canonical output not available. Run adapter stage first.")
        
        # Get task-specific postprocessor with config
        postprocessor = postprocess_registry.create(
            context.task_type,
            context.config.task.model_dump()  # Pass task config
        )
        
        # Run postprocessing
        result = postprocessor(context.canonical_output, context.config.task)
        
        context.task_result = result
        
        return context