"""Optimizer registry - registers model optimizers."""
from cvf.core.registry.registry import optimizer_registry, Registry

OptimizerRegistry = optimizer_registry

__all__ = ["optimizer_registry", "OptimizerRegistry", "Registry"]