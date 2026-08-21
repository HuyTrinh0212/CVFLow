"""Preprocessor registry - registers model-specific preprocessors."""
from cvf.core.registry.registry import preprocess_registry, Registry

PreprocessRegistry = preprocess_registry

__all__ = ["preprocess_registry", "PreprocessRegistry", "Registry"]