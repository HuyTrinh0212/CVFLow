"""Pipeline engine - executes stages dynamically based on config."""
from typing import List, Dict, Any, Optional
from cvf.core.pipeline.context import PipelineContext
from cvf.core.pipeline.stage import Stage
from cvf.core.contracts.config.main import CVFConfig
from cvf.core.contracts.config.pipeline import StageType
from cvf.core.registry import (
    preprocess_registry,
    backend_registry,
    adapter_registry,
    postprocess_registry,
    model_registry,
    task_registry,
)


class Pipeline:
    """Dynamic pipeline executor."""
    
    def __init__(self, config: CVFConfig):
        self.config = config
        self.stages: List[Stage] = []
        self._build_stages()
    
    def _build_stages(self) -> None:
        """Build stage instances from config."""
        stage_map = {
            StageType.PREPROCESS: self._create_preprocess_stage,
            StageType.INFERENCE: self._create_inference_stage,
            StageType.ADAPTER: self._create_adapter_stage,
            StageType.POSTPROCESS: self._create_postprocess_stage,
            StageType.BENCHMARK: self._create_benchmark_stage,
            StageType.EVALUATION: self._create_evaluation_stage,
        }
        
        for stage_type in self.config.pipeline.stages:
            if stage_type in stage_map:
                self.stages.append(stage_map[stage_type]())
            else:
                raise ValueError(f"Unknown stage type: {stage_type}")
    
    def _create_preprocess_stage(self) -> Stage:
        from cvf.core.pipeline.stages.preprocess import PreprocessStage
        return PreprocessStage(self.config)
    
    def _create_inference_stage(self) -> Stage:
        from cvf.core.pipeline.stages.inference import InferenceStage
        return InferenceStage(self.config)
    
    def _create_adapter_stage(self) -> Stage:
        from cvf.core.pipeline.stages.adapter import AdapterStage
        return AdapterStage(self.config)
    
    def _create_postprocess_stage(self) -> Stage:
        from cvf.core.pipeline.stages.postprocess import PostprocessStage
        return PostprocessStage(self.config)
    
    def _create_benchmark_stage(self) -> Stage:
        from cvf.core.pipeline.stages.benchmark import BenchmarkStage
        return BenchmarkStage(self.config)
    
    def _create_evaluation_stage(self) -> Stage:
        from cvf.core.pipeline.stages.evaluation import EvaluationStage
        return EvaluationStage(self.config)
    
    def run(self, input_data: Any) -> PipelineContext:
        """Execute pipeline with input data."""
        from cvf.core.contracts.runtime.input import ImageInput, TensorInput
        import numpy as np
        
        # Wrap input if needed
        if isinstance(input_data, (str, list)):
            input_data = ImageInput(path=input_data)
        elif isinstance(input_data, np.ndarray):
            # Check if it's a raw image (uint8, HWC) vs preprocessed tensor (float32, NCHW)
            if input_data.dtype == np.uint8 and input_data.ndim == 3:
                # Raw image - wrap as ImageInput for preprocessing
                input_data = ImageInput(array=input_data)
            else:
                # Assume preprocessed tensor
                input_data = TensorInput(data=input_data)
        elif hasattr(input_data, "__array__"):
            # Other array-like objects
            input_data = TensorInput(data=np.asarray(input_data))
        
        context = PipelineContext(config=self.config, input_data=input_data)
        
        for stage in self.stages:
            context = stage(context)
        
        return context
    
    def __call__(self, input_data: Any) -> PipelineContext:
        return self.run(input_data)