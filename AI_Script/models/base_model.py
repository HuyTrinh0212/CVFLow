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
        # ORT profiler JSON ("Output profiler - <model> ....json") is a debug-mode artifact.
        # Profiling stays off for inference/tracking/evaluate sessions so no profiler
        # JSON is written to output_path; debug mode (self.debug_mode=True) keeps it.
        sess_options.enable_profiling = self.debug_mode
        # Graph optimization level is configurable: some models (e.g. fp16 CRNN LSTM export) segfault
        # under ORT_ENABLE_ALL but run fine at ORT_ENABLE_EXTENDED.
        # config key: graph_optimization_level ("disable" | "basic" | "extended" | "all", default "all")
        opt_levels = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        opt_level = str(self.config.get("graph_optimization_level", "all")).lower()
        if opt_level not in opt_levels:
            raise ValueError(f"Unknown graph_optimization_level '{opt_level}'. Options: {list(opt_levels)}")
        sess_options.graph_optimization_level = opt_levels[opt_level]
        output_path = self.config.get("output_path")
        if not output_path or not str(output_path).strip():
            raise ValueError("'output_path' is required in config (no default).")
        output_path = os.path.abspath(str(output_path).strip())
        os.makedirs(output_path, exist_ok=True)
        if self.debug_mode:
            sess_options.profile_file_prefix = os.path.join(output_path, f"Output profiler - {self.model_name} ")
        return sess_options

    def load_model(self):
        device = self.config.get("target_device", "cpu").lower()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"]
        # Always pass session options (incl. configurable graph_optimization_level). Some fp16 exports
        # (e.g. CRNN LSTM) segfault under ORT_ENABLE_ALL -> config key graph_optimization_level.
        sess = lambda path: ort.InferenceSession(path, self.onnx_sess_options, providers=providers)
        if self.config.get("weight_path"):
            return sess(str(self.config.get("weight_path")))
        elif self.precision_format == 'fp32':
            return sess(str(self.config.get("weight_path_fp32")))
        elif self.precision_format == 'fp16':
            return sess(str(self.config.get("weight_path_fp16")))
        elif self.precision_format == 'int8':
            return sess(str(self.config.get("weight_path_int8")))
        elif self.precision_format == 'bf16':
            return sess(str(self.config.get("weight_path_bf16")))
        else:
            raise FileNotFoundError(f"No support precision format: {self.precision_format}")

    @abstractmethod
    def predict(self, image: np.ndarray):
        pass

    def __call__(self, image: np.ndarray):
        return self.predict(image)
