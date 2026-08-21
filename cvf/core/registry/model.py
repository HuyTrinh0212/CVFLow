"""Model registry - registers model families and variants."""
from cvf.core.registry.registry import model_registry, Registry

# Re-export for convenience
ModelRegistry = model_registry

__all__ = ["model_registry", "ModelRegistry", "Registry"]