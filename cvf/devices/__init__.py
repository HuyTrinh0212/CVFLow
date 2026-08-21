"""Devices package."""
from cvf.devices.cpu import CPUDevice, create_cpu_device
from cvf.devices.cuda import CUDADevice, create_cuda_device

__all__ = [
    "CPUDevice",
    "create_cpu_device",
    "CUDADevice", 
    "create_cuda_device",
]