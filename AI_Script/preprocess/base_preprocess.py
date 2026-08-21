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
        if self.precision_format == 'fp16':
            return self.convert_to_fp16(self.preprocess(image))
        return self.preprocess(image)