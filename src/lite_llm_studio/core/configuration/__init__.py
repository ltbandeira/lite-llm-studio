"""
Module core.configuration
-------------------------

This module contains the configuration schemas for the LiteLLM Studio project.
"""

from .base_config import BaseConfigModel
from .hardware_schema import (
    CPUInfoModel,
    DiskInfoModel,
    GPUInfoModel,
    HardwareScanReportModel,
    MemoryInfoModel,
    OSInfoModel,
)

__all__ = [
    "BaseConfigModel",
    "CPUInfoModel",
    "DiskInfoModel",
    "GPUInfoModel",
    "HardwareScanReportModel",
    "MemoryInfoModel",
    "OSInfoModel",
]
