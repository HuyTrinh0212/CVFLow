import os
import os.path
from abc import ABC, abstractmethod
import onnxruntime as ort
import numpy as np

class BaseModel(ABC):
    def __init__(self, config: dict | None = None, debug_mode = False):
        self.config = config or {}
        self.debug_mode = debug_mode
        self.model_name = str(self.config.get("model_name"))
        self.precision_format = str(self.config.get("precision_format"))
        self.onnx_sess_options = self._session_option()
        self.session = self.load_model()

    def _session_option(self):
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        output_path = self.config.get("output_path")
        if not output_path or not str(output_path).strip():
            raise ValueError("'output_path' is required in config (no default).")
        output_path = os.path.abspath(str(output_path).strip())
        os.makedirs(output_path, exist_ok=True)
        sess_options.profile_file_prefix = os.path.join(output_path, f"Output profiler - {self.model_name} ")
        return sess_options

    def load_model(self):
        device = self.config.get("target_device", "cpu").lower()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        
        if self.config.get("weight_path"):
            weight_path = str(self.config.get("weight_path"))
            if self.debug_mode:
                return ort.InferenceSession(weight_path, self.onnx_sess_options, providers=providers)
            return ort.InferenceSession(weight_path, providers=providers)
        else:
            if self.precision_format == 'fp32':
                weight_path_fp32 = str(self.config.get("weight_path_fp32"))
                if self.debug_mode:
                    return ort.InferenceSession(weight_path_fp32, self.onnx_sess_options, providers=providers)
                return ort.InferenceSession(weight_path_fp32, providers=providers)
            elif self.precision_format == 'fp16':
                weight_path_fp16 = str(self.config.get("weight_path_fp16"))
                if self.debug_mode:
                    return ort.InferenceSession(weight_path_fp16, self.onnx_sess_options, providers=providers)
                return ort.InferenceSession(weight_path_fp16, providers=providers)
            elif self.precision_format == 'int8':
                weight_path_int8 = str(self.config.get("weight_path_int8"))
                if self.debug_mode:
                    return ort.InferenceSession(weight_path_int8, self.onnx_sess_options, providers=providers)
                return ort.InferenceSession(weight_path_int8, providers=providers)
            else:
                raise FileNotFoundError(f"No support precision format: {self.precision_format}")

    @abstractmethod
    def predict(self, image: np.ndarray):
        pass

    def __call__(self, image: np.ndarray):
        return self.predict(image)
