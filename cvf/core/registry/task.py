"""Task registry - registers task implementations."""
from cvf.core.registry.registry import task_registry, Registry

TaskRegistry = task_registry

__all__ = ["task_registry", "TaskRegistry", "Registry"]