"""
Module lite_llm_studio.configuration.hardware
---------------------------------------------

This module contains the hardware-related configuration schemas for the LiteLLM Studio project.
"""

from typing import Annotated

from pydantic import Field, PositiveInt, StrictStr

from .base_config import BaseConfigModel


class OSInfoModel(BaseConfigModel):
    """
    Model representing the operating system information.
    """

    system: StrictStr = Field(..., description="Operating system name")
    version: StrictStr = Field(..., description="Operating system version")

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class CPUInfoModel(BaseConfigModel):
    """
    Model representing the CPU information.
    """

    brand: StrictStr = Field(..., description="CPU brand name")
    arch: StrictStr = Field(..., description="CPU architecture")
    cores: PositiveInt = Field(..., description="Number of CPU cores")
    threads: PositiveInt = Field(..., description="Number of CPU threads")
    frequency: Annotated[int | float, Field(..., gt=0, description="CPU frequency in GHz")]

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class GPUInfoModel(BaseConfigModel):
    """
    Model representing the GPU information.
    """

    name: StrictStr = Field(..., description="GPU name")
    total_vram: Annotated[int | float, Field(..., gt=0, description="Total GPU memory in GB")]
    driver: str | None = Field(..., description="GPU driver version")
    cuda: str | None = Field(..., description="CUDA version supported by the GPU")

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class MemoryInfoModel(BaseConfigModel):
    """
    Model representing the memory information.
    """

    total_memory: Annotated[int | float, Field(..., gt=0, description="Total memory in GB")]
    used_memory: Annotated[int | float, Field(..., gt=0, description="Used memory in GB")]
    free_memory: Annotated[int | float, Field(..., gt=0, description="Free memory in GB")]

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class DiskInfoModel(BaseConfigModel):
    """
    Model representing the disk information.
    """

    name: StrictStr = Field(..., description="Disk name")
    total_space: Annotated[int | float, Field(..., gt=0, description="Total disk space in GB")]
    used_space: Annotated[int | float, Field(..., gt=0, description="Used disk space in GB")]
    free_space: Annotated[int | float, Field(..., gt=0, description="Free disk space in GB")]

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class HardwareScanReportModel(BaseConfigModel):
    """
    Model representing the hardware scan report.
    """

    os: OSInfoModel = Field(..., description="Operating system information")
    cpu: CPUInfoModel = Field(..., description="CPU information")
    gpus: list[GPUInfoModel] = Field(default_factory=list)
    memory: MemoryInfoModel = Field(..., description="Memory information")
    disks: list[DiskInfoModel] = Field(default_factory=list, description="Disk information")

    @property
    def total_vram_gb(self) -> float:
        return float(sum((g.total_vram or 0.0) for g in self.gpus))

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def total_disk_space_gb(self) -> float:
        return float(sum((d.total_space or 0.0) for d in self.disks))

    @property
    def total_disk_free_gb(self) -> float:
        return float(sum((d.free_space or 0.0) for d in self.disks))

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()
