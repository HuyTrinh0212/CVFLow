"""Adapter registry - registers output adapters (model+task specific)."""
from cvf.core.registry.registry import adapter_registry, Registry

AdapterRegistry = adapter_registry

__all__ = ["adapter_registry", "AdapterRegistry", "Registry"]