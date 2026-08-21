"""Postprocessor registry - registers task-specific postprocessors."""
from cvf.core.registry.registry import postprocess_registry, Registry

PostprocessRegistry = postprocess_registry

__all__ = ["postprocess_registry", "PostprocessRegistry", "Registry"]