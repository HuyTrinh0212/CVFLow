"""Device registry - registers device implementations."""
from cvf.core.registry.registry import device_registry, Registry

DeviceRegistry = device_registry

__all__ = ["device_registry", "DeviceRegistry", "Registry"]