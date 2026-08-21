"""ResNet50 model definition - backend agnostic."""
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple


class ResNet50Model:
    """ResNet50 ONNX model wrapper - backend agnostic."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config.get("path")
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.input_shape: Tuple[int, ...] = ()
        
    def load(self, providers: List[str]) -> None:
        """Load ONNX model with given execution providers."""
        if not self.model_path or not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        
    def unload(self) -> None:
        """Unload model."""
        self.session = None
        
    def infer(self, input_tensor: np.ndarray):
        """Run inference on input tensor."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return list(outputs)
    
    def get_input_info(self) -> Dict[str, Any]:
        """Get model input information."""
        return {
            "name": self.input_name,
            "shape": self.input_shape,
        }
    
    def get_output_info(self) -> List[Dict[str, Any]]:
        """Get model output information."""
        if self.session is None:
            return []
        return [
            {"name": o.name, "shape": o.shape, "type": str(o.type)}
            for o in self.session.get_outputs()
        ]


def create_model(config: Dict[str, Any]) -> ResNet50Model:
    """Factory function to create ResNet50 model."""
    return ResNet50Model(config)