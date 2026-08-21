"""Inference stage - runs model inference via backend."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.core.contracts.runtime.output import ModelOutput
from cvf.core.registry import backend_registry, model_registry


class InferenceStage(Stage):
    """Run inference using registered backend."""
    
    name = "inference"
    
    def __init__(self, config):
        super().__init__(config)
        self._backend = None
        self._model_handle = None
    
    def run(self, context: PipelineContext) -> PipelineContext:
        if self._backend is None:
            # Initialize backend and load model
            backend = backend_registry.create(
                context.backend_type,
                context.config.backend.model_dump()  # Pass backend config
            )
            self._backend = backend
            
            model = model_registry.create(
                f"{context.model_family}.{context.model_variant}",
                context.config.model.model_dump()
            )
            device = self._create_device(context)
            self._model_handle = self._backend.load(model, device)
        
        # Run inference
        if context.preprocessed is None:
            raise ValueError("Preprocessed tensor not available. Run preprocess stage first.")
        input_tensor = context.preprocessed.data
        raw_output = self._backend.infer(self._model_handle, input_tensor)
        
        context.raw_output = ModelOutput(
            data=raw_output,
            metadata={"backend": context.backend_type}
        )
        
        return context
    
    def _create_device(self, context: PipelineContext):
        """Create device instance from config."""
        from cvf.core.registry import device_registry
        return device_registry.create(
            context.device_type,
            context.config.device.model_dump()
        )