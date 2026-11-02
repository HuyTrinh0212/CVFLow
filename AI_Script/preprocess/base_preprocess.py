# preprocess/base.py
from abc import ABC, abstractmethod
import numpy as np

class BasePreprocessor(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.precision_format = str(self.config.get("precision_format"))

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        pass

    def convert_to_fp16(self, input):
        return input.astype(np.float16)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Default: 'fp16' label => FP16 input tensor (FP16 weights with FP16 IO, e.g. YOLOv5 COCO / RT-DETR / ViT).
        # Some FP16 exports (e.g. YOLOv5 DOTA Satellite) keep FP32 IO -> skip cast via config override:
        #   input_dtype: "fp32" (options: fp32 | fp16, default derived from precision_format)
        input_dtype = self.config.get("input_dtype", "fp16" if self.precision_format == 'fp16' else "fp32")
        if str(input_dtype).lower() == 'fp16':
            return self.convert_to_fp16(self.preprocess(image))
        return self.preprocess(image)