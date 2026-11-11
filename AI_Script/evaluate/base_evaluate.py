from abc import ABC, abstractmethod
import os
import numpy as np

class BaseEvaluate(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def benchmark_single(self, all_image_path, all_predictitons, all_labels):
        pass

    @abstractmethod
    def benchmark_toltal(self, benchmark_dict):
        pass