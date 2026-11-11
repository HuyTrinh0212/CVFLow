from typing import Dict, Type
from AI_Script.models.base_model import BaseModel

class ModelRegistry:
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if not issubclass(model_class, BaseModel):
                raise TypeError("Only registered subclass of BasePreprocessor.")
            if name in cls._registry:
                raise ValueError(f"Model '{name}' are registered!")
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Lấy class model đã đăng ký theo tên."""
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' are not registered."
                             f"Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list:
        """Trả về danh sách tên các model đã đăng ký."""
        return list(cls._registry.keys())