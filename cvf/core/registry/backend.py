"""Backend registry - registers inference backends."""
from cvf.core.registry.registry import backend_registry, Registry

BackendRegistry = backend_registry

__all__ = ["backend_registry", "BackendRegistry", "Registry"]