"""CPU device implementation."""
from typing import Any, Dict, Optional
from cvf.core.contracts.config.device import DeviceType


class CPUDevice:
    """CPU device."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.type = DeviceType.CPU
        self.device_id = self.config.get("device_id", 0)
    
    def __repr__(self):
        return f"CPUDevice(id={self.device_id})"


def create_cpu_device(config: Optional[Dict[str, Any]] = None) -> CPUDevice:
    """Factory for CPU device."""
    return CPUDevice(config)