from typing import Dict, Type
from AI_Script.evaluate.base_evaluate import BaseEvaluate

class EvaluateRegistry:
    _registry: Dict[str, Type[BaseEvaluate]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class: Type[BaseEvaluate]) -> Type[BaseEvaluate]:
            if not issubclass(model_class, BaseEvaluate):
                raise TypeError("Only registered subclass of BaseBenchmark.")
            if name in cls._registry:
                raise ValueError(f"Task '{name}' are registered!")
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseEvaluate]:
        if name not in cls._registry:
            raise ValueError(f"Task '{name}' are not registered."
                             f"Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list:
        return list(cls._registry.keys())