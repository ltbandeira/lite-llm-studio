"""
Module core.instrumentation.scanner
-----------------------------------

This module defines the `HardwareScanner` class, responsible for collecting
detailed hardware information about the host system.
"""

import logging
import os
import platform
import re
import shutil
import subprocess

import psutil

from ..configuration import (
    CPUInfoModel,
    DiskInfoModel,
    GPUInfoModel,
    HardwareScanReportModel,
    MemoryInfoModel,
    OSInfoModel,
)


class HardwareScanner:
    """
    Hardware scanner for system inspection.

    Attributes:
        logger (logging.Logger): Logger instance for debug and error reporting.
    """

    def __init__(self, logger_name: str = "app.scanner"):
        """
        Initialize the hardware scanner.

        Args:
            logger_name (str): Name of the logger for this component.
        """
        self.logger = logging.getLogger(logger_name)

    def scan(self) -> HardwareScanReportModel:
        """
        Perform a full hardware scan.

        Collects all available information about OS, CPU, GPUs, memory, and disks,
        and returns a structured report model.

        Returns:
            HardwareScanReportModel: Structured report containing hardware details.

        Raises:
            Exception: If any unexpected error occurs during scanning.
        """
        self.logger.info("Starting full hardware scan")

        try:
            self.logger.debug("Collecting operating system information")
            os_info = self._get_os_info()

            self.logger.debug("Collecting CPU information")
            cpu_info = self._get_cpu_info()

            self.logger.debug("Collecting GPU information")
            gpu_info = self._get_gpu_info()

            self.logger.debug("Collecting memory information")
            memory_info = self._get_memory_info()

            self.logger.debug("Collecting disk information")
            disk_info = self._get_disk_info()

            report = HardwareScanReportModel(os=os_info, cpu=cpu_info, gpus=gpu_info, memory=memory_info, disks=disk_info)

            self.logger.info("Hardware scan completed")
            return report

        except Exception as e:
            self.logger.error(f"Error during hardware scan: {str(e)}", exc_info=True)
            raise

    def _get_os_info(self) -> OSInfoModel:
        """
        Collect operating system details.

        Returns:
            OSInfoModel: OS name and version.
        """
        try:
            system = platform.system() or "Unknown"
            version_parts = [platform.release()]
            mach = platform.machine()
            if mach:
                version_parts.append(mach)
            version = " ".join([p for p in version_parts if p]) or "Unknown"

            self.logger.debug(f"Detected OS: {system} {version}")
            return OSInfoModel(system=system, version=version)

        except Exception as e:
            self.logger.warning(f"Failed to collect OS information: {e}")
            return OSInfoModel(system="Unknown", version="Unknown")

    def _get_cpu_info(self) -> CPUInfoModel:
        """
        Collect CPU details.

        Returns:
            CPUInfoModel: Structured CPU information.

        Raises:
            Exception: If CPU information cannot be retrieved.
        """
        try:
            self.logger.debug("Collecting CPU information")

            cpu_brand = platform.processor()
            cpu_cores = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            cpu_arch = platform.machine() or "Unknown"
            cpu_freq: float | None = None

            # Workaround for missing brand info on Windows
            if not cpu_brand or cpu_brand.strip() == "Unknown":
                try:
                    ps_cmd = "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"
                    r = subprocess.run(
                        ["powershell", "-NoProfile", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        cpu_brand = r.stdout.strip()
                except (subprocess.TimeoutExpired, OSError) as e:
                    self.logger.debug(f"Failed to obtain CPU name via CIM: {e}")

            try:
                f = psutil.cpu_freq()
                if f and f.current:
                    cpu_freq = round(float(f.current) / 1000.0, 2)
            except Exception as e:
                self.logger.debug(f"psutil.cpu_freq failed: {e}")

            if cpu_freq is None:
                try:
                    ps_cmd = "(Get-CimInstance Win32_Processor | " "Measure-Object -Property CurrentClockSpeed -Average).Average"
                    r = subprocess.run(
                        ["powershell", "-NoProfile", "-Command", ps_cmd],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        mhz = float(r.stdout.strip())
                        if mhz > 0:
                            cpu_freq = round(mhz / 1000.0, 2)
                except (subprocess.TimeoutExpired, OSError, ValueError) as e:
                    self.logger.debug(f"PowerShell CIM failed: {e}")

            if cpu_freq is None:
                cpu_freq = 0.0

            cpu_info = CPUInfoModel(
                brand=cpu_brand or "Unknown",
                arch=cpu_arch,
                cores=cpu_cores,
                threads=cpu_threads,
                frequency=cpu_freq,
            )

            self.logger.debug(f"CPU detected: {cpu_info.brand} - {cpu_info.cores} cores, {cpu_info.threads} threads")
            return cpu_info

        except Exception as e:
            self.logger.error(f"Error while collecting CPU information: {e}")
            raise

    def _get_gpu_info(self) -> list[GPUInfoModel]:
        """
        Collect GPU information.

        Returns:
            list[GPUInfoModel]: List of detected GPUs.
        """
        self.logger.debug("Detecting available GPUs")
        gpus: list[GPUInfoModel] = []

        if platform.system() != "Windows":
            self.logger.debug("OS is not Windows; skipping GPU detection.")
            return gpus

        # Try detecting NVIDIA GPUs via nvidia-smi
        try:
            smi_candidates = []
            smi_which = shutil.which("nvidia-smi")
            if smi_which:
                smi_candidates.append(smi_which)
            smi_candidates += [
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            ]
            smi_exec = next((p for p in smi_candidates if p and os.path.exists(p)), None)

            if smi_exec:
                result = subprocess.run(
                    [
                        smi_exec,
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    cuda_version = self._get_cuda_version()
                    for line in result.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            name = parts[0]
                            try:
                                vram_gb = round(float(parts[1]) / 1024.0, 2)
                            except Exception:
                                vram_gb = 0.0
                            driver = parts[2]
                            gpus.append(
                                GPUInfoModel(
                                    name=name,
                                    total_vram=vram_gb,
                                    driver=driver,
                                    cuda=cuda_version,
                                )
                            )
                    self.logger.debug(f"NVIDIA GPUs detected: {len(gpus)}")
        except Exception as e:
            self.logger.debug(f"nvidia-smi not available or failed: {e}")

        # Query additional GPUs via CIM
        try:
            ps_cmd = "Get-CimInstance Win32_VideoController | " "Select-Object Name,AdapterRAM,DriverVersion | " "ConvertTo-Csv -NoTypeInformation"
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=7,
            )
            if r.returncode == 0 and r.stdout:
                import csv
                import io

                reader = csv.DictReader(io.StringIO(r.stdout))
                for row in reader:
                    name = (row.get("Name") or "").strip()
                    if not name:
                        continue
                    lname = name.lower()
                    if "microsoft basic" in lname:
                        continue
                    if "nvidia" in lname and any("nvidia" in gpu.name.lower() for gpu in gpus):
                        continue

                    try:
                        adapter_ram = float(row.get("AdapterRAM") or 0.0)
                    except ValueError:
                        adapter_ram = 0.0
                    vram_gb = round(adapter_ram / (1024**3), 2) if adapter_ram > 0 else 0.0
                    driver_version = row.get("DriverVersion")
                    gpu_driver: str | None = None
                    if driver_version:
                        driver_str = str(driver_version).strip()
                        gpu_driver = driver_str if driver_str else None

                    gpus.append(
                        GPUInfoModel(
                            name=name,
                            total_vram=vram_gb,
                            driver=gpu_driver,
                            cuda=None,
                        )
                    )
        except Exception as e:
            self.logger.debug(f"Failed to query Win32_VideoController via CIM: {e}")

        if not gpus:
            self.logger.debug("No GPUs detected")
        else:
            self.logger.debug(f"Total of {len(gpus)} GPU(s) detected")

        return gpus

    def _get_cuda_version(self) -> str | None:
        """
        Try to detect the CUDA version.

        Returns:
            Optional[str]: CUDA version string if available, otherwise None.
        """
        if platform.system() != "Windows":
            return None
        try:
            smi_candidates = []
            smi_which = shutil.which("nvidia-smi")
            if smi_which:
                smi_candidates.append(smi_which)
            smi_candidates += [
                r"C:\Windows\System32\nvidia-smi.exe",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            ]
            for smi in smi_candidates:
                if smi and os.path.exists(smi):
                    r = subprocess.run([smi], capture_output=True, text=True, timeout=5)
                    if r.returncode == 0 and r.stdout:
                        m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
                        if m:
                            return m.group(1)
        except Exception:
            pass
        return None

    def _get_memory_info(self) -> MemoryInfoModel:
        """
        Collect memory usage information.

        Returns:
            MemoryInfoModel: Total, used, and free memory in GB.
        """
        try:
            mem = psutil.virtual_memory()

            total_gb = round(mem.total / (1024**3), 2)
            used_gb = round(mem.used / (1024**3), 2)
            free_gb = round(mem.available / (1024**3), 2)

            self.logger.debug(f"RAM: {total_gb}GB total, {used_gb}GB used, {free_gb}GB free")
            return MemoryInfoModel(total_memory=total_gb, used_memory=used_gb, free_memory=free_gb)

        except Exception as e:
            self.logger.error(f"Error while collecting memory information: {e}")
            return MemoryInfoModel(total_memory=0.0, used_memory=0.0, free_memory=0.0)

    def _get_disk_info(self) -> list[DiskInfoModel]:
        """
        Collect information about disk partitions and usage.

        Returns:
            list[DiskInfoModel]: List of detected disks with usage statistics.
        """
        self.logger.debug("Collecting disk information")
        disks = []

        try:
            partitions = psutil.disk_partitions()
            self.logger.debug(f"Found {len(partitions)} partitions")

            for partition in partitions:
                try:
                    if platform.system() == "Windows":
                        if partition.fstype in ["", "cdfs"]:
                            continue
                        if partition.mountpoint.startswith("\\"):
                            continue

                    disk_usage = shutil.disk_usage(partition.mountpoint)

                    total_gb = round(disk_usage.total / (1024**3), 2)
                    free_gb = round(disk_usage.free / (1024**3), 2)
                    used_gb = round((disk_usage.total - disk_usage.free) / (1024**3), 2)

                    if total_gb < 1.0:
                        continue

                    disk_name = partition.device
                    if platform.system() == "Windows":
                        disk_name = partition.mountpoint

                    disk_info = DiskInfoModel(
                        name=disk_name,
                        total_space=total_gb,
                        used_space=used_gb,
                        free_space=free_gb,
                    )
                    disks.append(disk_info)
                    self.logger.debug(f"Disk detected: {disk_name} - {total_gb}GB total")

                except (PermissionError, OSError) as e:
                    self.logger.debug(f"Inaccessible disk ignored: {partition.device} - {e}")
                    continue

            self.logger.debug(f"Total of {len(disks)} disk(s) detected")

        except Exception as e:
            self.logger.error(f"Error while collecting disk information: {e}")
            disks.append(DiskInfoModel(name="Unknown", total_space=0.0, used_space=0.0, free_space=0.0))

        return disks

    def check_cuda_support(self) -> bool:
        """
        Check if CUDA support is available in llama-cpp-python.

        Returns:
            bool: True if CUDA GPU offload is supported, False otherwise.
        """
        try:
            import llama_cpp.llama_cpp as llama_low

            if hasattr(llama_low, "llama_supports_gpu_offload"):
                return llama_low.llama_supports_gpu_offload()
            else:
                self.logger.warning("llama_supports_gpu_offload function not found")
                return False
        except ImportError:
            self.logger.warning("llama-cpp-python not available")
            return False
        except Exception as e:
            self.logger.error(f"Error checking CUDA support: {e}")
            return False

    def get_gpu_runtime_info(self) -> dict:
        """
        Get basic GPU information for model inference.

        Returns:
            dict: Basic GPU information including name.
        """
        try:
            # Try to get GPU info from nvidia-smi (name only)
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                gpu_info = {}
                for i, line in enumerate(lines):
                    name = line.strip()
                    if name:
                        gpu_info[f"gpu_{i}"] = {"name": name}
                return gpu_info
        except Exception as e:
            self.logger.debug(f"Could not get GPU runtime info: {e}")

        return {}
