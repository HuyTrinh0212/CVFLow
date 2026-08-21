"""Generic registry base class."""
from typing import Dict, Type, TypeVar, Generic, Optional, Callable, Any, Union

T = TypeVar("T")
Factory = Callable[..., Any]


class Registry(Generic[T]):
    """Generic registry for dynamic component resolution."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, type] = {}
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Factory] = {}
    
    def register(self, name: str, target: Optional[Union[type, Factory]] = None) -> Union[Callable, type, Factory]:
        """Register a class or factory function."""
        def decorator(t: Union[type, Factory]) -> Union[type, Factory]:
            if name in self._registry or name in self._factories:
                raise ValueError(f"'{name}' already registered in {self.name}")
            
            if isinstance(t, type):
                self._registry[name] = t
            else:
                self._factories[name] = t
            return t
        
        if target is not None:
            return decorator(target)
        return decorator
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """Create an instance by name."""
        if name in self._instances:
            return self._instances[name]
        
        if name in self._registry:
            instance = self._registry[name](*args, **kwargs)
        elif name in self._factories:
            instance = self._factories[name](*args, **kwargs)
        else:
            available = list(self._registry.keys()) + list(self._factories.keys())
            raise ValueError(
                f"'{name}' not found in {self.name}. Available: {available}"
            )
        
        self._instances[name] = instance
        return instance
    
    def get_class(self, name: str) -> type:
        """Get registered class by name."""
        if name not in self._registry:
            raise ValueError(f"'{name}' not found in {self.name}")
        return self._registry[name]
    
    def list(self) -> list[str]:
        """List all registered names."""
        return list(self._registry.keys()) + list(self._factories.keys())
    
    def clear(self) -> None:
        """Clear all registrations (for testing/reloading)."""
        self._registry.clear()
        self._factories.clear()
        self._instances.clear()
    
    def clear_instances(self) -> None:
        """Clear cached instances."""
        self._instances.clear()


# Global registries - typed via string annotations to avoid circular imports
model_registry: "Registry" = Registry("models")
task_registry: "Registry" = Registry("tasks")
backend_registry: "Registry" = Registry("backends")
device_registry: "Registry" = Registry("devices")
adapter_registry: "Registry" = Registry("adapters")
optimizer_registry: "Registry" = Registry("optimizers")
preprocess_registry: "Registry" = Registry("preprocessors")
postprocess_registry: "Registry" = Registry("postprocessors")