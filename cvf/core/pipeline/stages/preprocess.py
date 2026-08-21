"""Preprocess stage - runs model-specific preprocessing."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.core.contracts.runtime.input import ImageInput, TensorInput
from cvf.core.registry import preprocess_registry
import numpy as np


class PreprocessStage(Stage):
    """Run model-specific preprocessing."""
    
    name = "preprocess"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        # Get model-specific preprocessor with config
        preprocessor = preprocess_registry.create(
            f"{context.model_family}.{context.model_variant}",
            context.config.model.model_dump()  # Pass model config
        )
        
        # Run preprocessing
        input_data = context.input_data
        if isinstance(input_data, ImageInput):
            # Load image if path provided
            if input_data.path:
                import cv2
                img = cv2.imread(input_data.path)
                if img is None:
                    raise ValueError(f"Failed to load image: {input_data.path}")
                input_data.array = img
            
            tensor = preprocessor(input_data.array)
        elif isinstance(input_data, TensorInput):
            # Already preprocessed tensor - pass through
            tensor = input_data.data
        else:
            # Assume it's a raw image array (numpy)
            # Check if it needs preprocessing (uint8 -> float conversion)
            if isinstance(input_data, np.ndarray) and input_data.dtype != np.float32:
                tensor = preprocessor(input_data)
            else:
                tensor = input_data
        
        context.preprocessed = TensorInput(
            data=tensor,
            metadata={"preprocessor": f"{context.model_family}.{context.model_variant}"}
        )
        
        return context