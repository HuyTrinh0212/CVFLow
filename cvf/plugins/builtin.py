"""Built-in plugin registration - registers all CVF components."""
from cvf.core.registry import (
    model_registry,
    preprocess_registry,
    adapter_registry,
    task_registry,
    postprocess_registry,
    backend_registry,
    device_registry,
)


def register_builtin() -> None:
    """Register all built-in components (idempotent)."""
    
    # Clear existing registrations to allow re-registration
    model_registry.clear()
    preprocess_registry.clear()
    adapter_registry.clear()
    task_registry.clear()
    postprocess_registry.clear()
    backend_registry.clear()
    device_registry.clear()
    
    # === MODELS ===
    from cvf.models.yolo.yolov5 import (
        create_model as yolov5_create_model,
        create_preprocessor as yolov5_create_preprocessor,
        create_adapter as yolov5_create_adapter,
    )
    
    model_registry.register("yolo.yolov5", yolov5_create_model)
    preprocess_registry.register("yolo.yolov5", yolov5_create_preprocessor)
    adapter_registry.register("yolo.yolov5.detection", yolov5_create_adapter)
    
    from cvf.models.classification.resnet50 import (
        create_model as resnet50_create_model,
        create_preprocessor as resnet50_create_preprocessor,
        create_adapter as resnet50_create_adapter,
    )
    
    model_registry.register("classification.resnet50", resnet50_create_model)
    preprocess_registry.register("classification.resnet50", resnet50_create_preprocessor)
    adapter_registry.register("classification.resnet50.classification", resnet50_create_adapter)
    
    from cvf.models.classification.lenet5 import (
        create_model as lenet5_create_model,
        create_preprocessor as lenet5_create_preprocessor,
        create_adapter as lenet5_create_adapter,
    )
    
    model_registry.register("classification.lenet5", lenet5_create_model)
    preprocess_registry.register("classification.lenet5", lenet5_create_preprocessor)
    adapter_registry.register("classification.lenet5.classification", lenet5_create_adapter)
    
    from cvf.models.crnn.crnn import (
        create_model as crnn_create_model,
        create_preprocessor as crnn_create_preprocessor,
        create_adapter as crnn_create_adapter,
    )
    
    model_registry.register("crnn.crnn", crnn_create_model)
    preprocess_registry.register("crnn.crnn", crnn_create_preprocessor)
    adapter_registry.register("crnn.crnn.htr", crnn_create_adapter)
    
    # === TASKS ===
    from cvf.tasks import (
        create_detection_postprocess,
        create_classification_postprocess,
        create_htr_postprocess,
    )
    
    task_registry.register("detection", lambda c: None)
    postprocess_registry.register("detection", create_detection_postprocess)
    
    task_registry.register("classification", lambda c: None)
    postprocess_registry.register("classification", create_classification_postprocess)
    
    task_registry.register("htr", lambda c: None)
    postprocess_registry.register("htr", create_htr_postprocess)
    
    # === BACKENDS ===
    from cvf.backends import create_onnxruntime_backend
    
    backend_registry.register("onnxruntime", create_onnxruntime_backend)
    
    # === DEVICES ===
    from cvf.devices import create_cpu_device, create_cuda_device
    
    device_registry.register("cpu", create_cpu_device)
    device_registry.register("cuda", create_cuda_device)
    
    print("Built-in components registered successfully")


# Auto-register on import
register_builtin()