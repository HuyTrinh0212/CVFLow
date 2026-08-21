"""ONNX Runtime backend implementation."""
import onnxruntime as ort
from typing import Any, List, Dict, Optional

from cvf.backends.base import InferenceBackend
from cvf.devices import CPUDevice, CUDADevice


class ONNXRuntimeBackend(InferenceBackend):
    """ONNX Runtime inference backend."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.graph_optimization = self.config.get("graph_optimization", "all")
        self.enable_profiling = self.config.get("enable_profiling", False)
    
    def get_providers(self, device: Any) -> List[str]:
        """Get ONNX Runtime execution providers for device."""
        if isinstance(device, CUDADevice):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif isinstance(device, CPUDevice):
            return ["CPUExecutionProvider"]
        else:
            # Default to CPU
            return ["CPUExecutionProvider"]
    
    def load(self, model: Any, device: Any) -> Any:
        """Load ONNX model with appropriate providers."""
        providers = self.get_providers(device)
        
        sess_options = ort.SessionOptions()
        if self.graph_optimization == "all":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if self.enable_profiling:
            sess_options.enable_profiling = True
        
        # Load the model
        session = ort.InferenceSession(model.model_path, sess_options, providers=providers)
        
        # Store session in model
        model.session = session
        model.input_name = session.get_inputs()[0].name
        model.output_names = [o.name for o in session.get_outputs()]
        model.input_shape = tuple(session.get_inputs()[0].shape)
        
        return model
    
    def infer(self, handle: Any, input_tensor: Any) -> List[Any]:
        """Run inference using ONNX Runtime."""
        if handle.session is None:
            raise RuntimeError("Model not loaded")
        return handle.session.run(handle.output_names, {handle.input_name: input_tensor})
    
    def unload(self, handle: Any) -> None:
        """Unload model."""
        handle.session = None


def create_onnxruntime_backend(config: Optional[Dict[str, Any]] = None) -> ONNXRuntimeBackend:
    """Factory for ONNX Runtime backend."""
    return ONNXRuntimeBackend(config)