"""Base inference backend interface."""
from abc import ABC, abstractmethod
from typing import Any, List, Dict


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def load(self, model: Any, device: Any) -> Any:
        """Load model for inference.
        
        Args:
            model: Model instance (e.g., YOLOv5Model)
            device: Device instance (e.g., CPUDevice, CUDADevice)
            
        Returns:
            Model handle for inference
        """
        pass
    
    @abstractmethod
    def infer(self, handle: Any, input_tensor: Any) -> List[Any]:
        """Run inference.
        
        Args:
            handle: Model handle from load()
            input_tensor: Input tensor (numpy array)
            
        Returns:
            List of output tensors
        """
        pass
    
    @abstractmethod
    def unload(self, handle: Any) -> None:
        """Unload model."""
        pass
    
    @abstractmethod
    def get_providers(self, device: Any) -> List[str]:
        """Get execution providers for device."""
        pass