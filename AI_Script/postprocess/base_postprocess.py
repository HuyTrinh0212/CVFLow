from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BasePostprocessor(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.postprocess(*args, **kwargs)