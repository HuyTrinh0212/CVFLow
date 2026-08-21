from typing import Dict, Type
from AI_Script.postprocess.base_postprocess import BasePostprocessor

class PostProcessorRegistry:
    _registry: Dict[str, Type[BasePostprocessor]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(processor_class: Type[BasePostprocessor]) -> Type[BasePostprocessor]:
            if not issubclass(processor_class, BasePostprocessor):
                raise TypeError("Only registered subclass of BasePostProcessor.")
            if name in cls._registry:
                raise ValueError(f"Model '{name}' are registered!")
            cls._registry[name] = processor_class
            return processor_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BasePostprocessor]:
        if name not in cls._registry:
            raise ValueError(f"PostProcessor '{name}' are not registered."
                             f"\nAvailable: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list:
        return list(cls._registry.keys())