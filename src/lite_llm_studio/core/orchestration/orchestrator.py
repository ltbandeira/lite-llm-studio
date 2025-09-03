"""
Module core.orchestration.orchestrator
--------------------------------------

This module defines the Orchestrator class, the central component responsible
for coordinating the main operations of LiteLLM Studio.
"""

import logging
from typing import Any

from ..instrumentation import HardwareScanner


class Orchestrator:
    """
    Main orchestrator class for LiteLLM Studio.

    This class coordinates the execution of different system components.
    """

    def __init__(self, logger_name: str = "app.orchestrator"):
        """
        Initialize the orchestrator instance.

        Args:
            logger_name (str): Name of the logger to be used for this component.
        """
        self.logger = logging.getLogger(logger_name)
        self.hardware_scanner = HardwareScanner()
        self.logger.info("Orchestrator successfully initialized")

    def execute_hardware_scan(self) -> dict[str, Any] | None:
        """
        Perform a hardware scan of the system.

        The scan collects information about the operating system, CPU, memory,
        and available GPUs. The results are logged and returned as a dictionary.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing hardware information,
                                      or None if an error occurs.
        """
        self.logger.info("Starting hardware scan")

        try:
            hardware_report = self.hardware_scanner.scan()
            self.logger.info("Hardware scan completed successfully")

            # Log key information
            self.logger.debug(f"Operating System: {hardware_report.os.system} {hardware_report.os.version}")
            self.logger.debug(f"CPU: {hardware_report.cpu.brand} - {hardware_report.cpu.cores} cores")
            self.logger.debug(f"Total Memory: {hardware_report.memory.total_memory} GB")

            if hardware_report.has_gpu:
                self.logger.debug(f"GPUs detected: {len(hardware_report.gpus)}")
                for i, gpu in enumerate(hardware_report.gpus, 1):
                    self.logger.debug(f"GPU {i}: {gpu.name} - {gpu.total_vram} GB VRAM")
            else:
                self.logger.debug("No GPUs detected")

            return hardware_report.to_dict()

        except Exception as e:
            self.logger.error(f"Error during hardware scan: {str(e)}", exc_info=True)
            return None

    def run_full_pipeline(self) -> bool:
        """
        Run the full LiteLLM Studio pipeline.

        Returns:
            bool: True if the pipeline completes successfully, False otherwise.
        """
        self.logger.info("Starting full LiteLLM Studio pipeline")

        try:
            # Execute hardware scan
            hardware_info = self.execute_hardware_scan()
            if hardware_info is None:
                self.logger.error("Hardware scan failed, aborting pipeline")
                return False

            self.logger.info("Full pipeline executed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
            return False
