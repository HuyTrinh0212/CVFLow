"""Adapter stage - converts raw model output to canonical format."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.core.contracts.runtime.output import ModelOutput
from cvf.core.registry import adapter_registry


class AdapterStage(Stage):
    """Convert raw model output to canonical output using model+task adapter."""
    
    name = "adapter"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.raw_output is None:
            raise ValueError("Raw output not available. Run inference stage first.")
        
        # Get model+task specific adapter with config
        adapter = adapter_registry.create(
            f"{context.model_family}.{context.model_variant}.{context.task_type}",
            context.config.model.model_dump()
        )
        
        # Convert to canonical format
        canonical = adapter.adapt(context.raw_output.data)
        
        context.canonical_output = canonical
        
        return context