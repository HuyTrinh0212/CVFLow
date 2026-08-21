"""Compatibility resolver - validates config combinations before execution."""
from cvf.core.contracts.config.main import CVFConfig
from cvf.core.contracts.config.model import ModelConfig
from cvf.core.contracts.config.task import TaskConfig, TaskType
from cvf.core.contracts.config.backend import BackendConfig, BackendType
from cvf.core.contracts.config.device import DeviceConfig, DeviceType
from cvf.core.contracts.config.optimization import OptimizationConfig, OptimizationType
from pathlib import Path


class CompatibilityError(Exception):
    """Raised when config combination is incompatible."""
    pass


class CompatibilityResolver:
    """Validates config combinations before pipeline construction."""
    
    # Model format -> supported backends
    MODEL_BACKEND_MAP = {
        ".onnx": [BackendType.ONNXRUNTIME, BackendType.TENSORRT, BackendType.OPENVINO],
        ".pt": [BackendType.PYTORCH, BackendType.ONNXRUNTIME],
        ".engine": [BackendType.TENSORRT],
        ".xml": [BackendType.OPENVINO],
        ".tflite": [BackendType.TFLITE],
    }
    
    # Backend -> supported devices
    BACKEND_DEVICE_MAP = {
        BackendType.ONNXRUNTIME: [DeviceType.CPU, DeviceType.CUDA],
        BackendType.PYTORCH: [DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS],
        BackendType.TENSORRT: [DeviceType.CUDA],
        BackendType.OPENVINO: [DeviceType.CPU, DeviceType.CUDA],
        BackendType.TFLITE: [DeviceType.CPU],
    }
    
    # Model family -> supported tasks
    FAMILY_TASK_MAP = {
        "yolo": [TaskType.DETECTION, TaskType.SEGMENTATION, TaskType.POSE],
        "detr": [TaskType.DETECTION],
        "classification": [TaskType.CLASSIFICATION],
        "crnn": [TaskType.HTR],
    }
    
    # Optimization -> supported model formats
    OPTIMIZATION_FORMAT_MAP = {
        OptimizationType.QUANTIZATION: [".onnx"],
        OptimizationType.PRUNING: [".pt", ".onnx"],
        OptimizationType.DISTILLATION: [".pt", ".onnx"],
        OptimizationType.GRAPH_OPTIMIZATION: [".onnx", ".pt"],
    }
    
    @classmethod
    def validate(cls, config: CVFConfig) -> None:
        """
        Validate all compatibility rules.
        
        Args:
            config: CVFConfig to validate
            
        Raises:
            CompatibilityError: If any combination is invalid
        """
        errors = []
        
        # Model format <-> Backend
        model_ext = Path(config.model.path).suffix.lower()
        if model_ext in cls.MODEL_BACKEND_MAP:
            supported = cls.MODEL_BACKEND_MAP[model_ext]
            if config.backend.type not in supported:
                errors.append(
                    f"Model format '{model_ext}' not supported by backend '{config.backend.type}'. "
                    f"Supported: {[b.value for b in supported]}"
                )
        
        # Backend <-> Device
        if config.backend.type in cls.BACKEND_DEVICE_MAP:
            supported = cls.BACKEND_DEVICE_MAP[config.backend.type]
            if config.device.type not in supported:
                errors.append(
                    f"Backend '{config.backend.type}' does not support device '{config.device.type}'. "
                    f"Supported: {[d.value for d in supported]}"
                )
        
        # Model family <-> Task
        family = config.model.family.lower()
        if family in cls.FAMILY_TASK_MAP:
            supported = cls.FAMILY_TASK_MAP[family]
            if config.task.type not in supported:
                errors.append(
                    f"Model family '{family}' does not support task '{config.task.type}'. "
                    f"Supported: {[t.value for t in supported]}"
                )
        
        # Optimization <-> Model format
        if config.optimization:
            opt_type = config.optimization.type
            if opt_type in cls.OPTIMIZATION_FORMAT_MAP:
                supported = cls.OPTIMIZATION_FORMAT_MAP[opt_type]
                if model_ext not in supported:
                    errors.append(
                        f"Optimization '{opt_type}' requires model format in {supported}, "
                        f"got '{model_ext}'"
                    )
        
        if errors:
            raise CompatibilityError("\n".join(errors))