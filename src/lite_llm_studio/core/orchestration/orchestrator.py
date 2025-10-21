"""
Module core.orchestration.orchestrator
--------------------------------------

This module defines the Orchestrator class, the central component responsible
for coordinating the main operations of LiteLLM Studio.
"""

import logging
from typing import TYPE_CHECKING, Any

from ..data import DatasetManager, DocumentProcessor
from ..instrumentation import HardwareScanner

if TYPE_CHECKING:
    from ..configuration.data_schema import DataProcessingConfig


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
        self.logger.debug(f"Initializing Orchestrator with logger: {logger_name}")

        try:
            self.logger.info("Orchestrator successfully initialized")
            self.hardware_scanner = HardwareScanner()
            self.logger.debug("Hardware scanner component ready")
            self.document_processor = DocumentProcessor()
            self.logger.debug("Document processor component ready")
            self.dataset_manager = DatasetManager()
            self.logger.debug("Dataset manager component ready")
        except Exception as e:
            self.logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
            raise

    def execute_hardware_scan(self) -> dict[str, Any] | None:
        """
        Perform a hardware scan of the system.

        The scan collects information about the operating system, CPU, memory,
        and available GPUs. The results are logged and returned as a dictionary.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing hardware information,
                                      or None if an error occurs.
        """
        self.logger.info("=" * 40)
        self.logger.info("Starting hardware scan")
        self.logger.info("=" * 40)

        try:
            self.logger.debug("Invoking hardware scanner...")
            hardware_report = self.hardware_scanner.scan()

            if hardware_report is None:
                self.logger.error("Hardware scanner returned None")
                return None

            self.logger.info("Hardware scan completed successfully")

            # Log Operating System information
            self.logger.info(f"Operating System: {hardware_report.os.system} {hardware_report.os.version}")

            # Log CPU information
            self.logger.info(f"CPU: {hardware_report.cpu.brand}")
            self.logger.debug(f"  Cores: {hardware_report.cpu.cores} physical, {hardware_report.cpu.threads} threads")
            self.logger.debug(f"  Architecture: {hardware_report.cpu.arch}")
            if hardware_report.cpu.frequency:
                self.logger.debug(f"  Max Frequency: {hardware_report.cpu.frequency} GHz")

            # Log Memory information
            total_mem_gb = hardware_report.memory.total_memory
            free_mem_gb = hardware_report.memory.free_memory
            used_mem_gb = total_mem_gb - free_mem_gb
            usage_pct = (used_mem_gb / total_mem_gb * 100) if total_mem_gb > 0 else 0

            self.logger.info(f"Memory: {total_mem_gb:.2f} GB total")
            self.logger.debug(f"  Used: {used_mem_gb:.2f} GB ({usage_pct:.1f}%)")
            self.logger.debug(f"  Free: {free_mem_gb:.2f} GB")

            # Log GPU information
            if hardware_report.has_gpu:
                self.logger.info(f"GPUs detected: {len(hardware_report.gpus)}")
                for i, gpu in enumerate(hardware_report.gpus, 1):
                    self.logger.info(f"  GPU {i}: {gpu.name}")
                    self.logger.debug(f"    VRAM: {gpu.total_vram} GB")
                    if gpu.driver:
                        self.logger.debug(f"    Driver: {gpu.driver}")
                    if gpu.cuda:
                        self.logger.debug(f"    CUDA: {gpu.cuda}")
            else:
                self.logger.info("No dedicated GPUs detected (CPU-only mode)")

            # Log Disk information
            if hardware_report.disks and len(hardware_report.disks) > 0:
                self.logger.debug(f"Storage devices: {len(hardware_report.disks)}")
                for i, disk in enumerate(hardware_report.disks, 1):
                    usage_pct = (disk.used_space / disk.total_space * 100) if disk.total_space > 0 else 0
                    self.logger.debug(f"  Disk {i} ({disk.name}): {disk.total_space:.1f} GB total, {disk.used_space:.1f} GB used ({usage_pct:.1f}%)")

            self.logger.info("=" * 40)
            self.logger.info("Hardware scan report generated successfully")
            self.logger.info("=" * 40)

            return hardware_report.to_dict()

        except Exception as e:
            self.logger.error("=" * 40)
            self.logger.error(f"CRITICAL ERROR during hardware scan: {str(e)}")
            self.logger.error("=" * 40, exc_info=True)
            return None

    def execute_document_processing(self, config: "DataProcessingConfig") -> dict[str, Any] | None:
        """
        Process documents using the document processor.

        Args:
            config: Configuration for document processing.

        Returns:
            Optional[Dict[str, Any]]: Processing job results or None if error occurs.
        """

        self.logger.info("=" * 20)
        self.logger.info("Starting document processing")
        self.logger.info("=" * 20)

        try:
            # Create and execute processing job
            job = self.document_processor.create_processing_job(config)
            job = self.document_processor.process_job(job)

            if job.status.value == "completed":
                self.logger.info("Document processing completed successfully")
                self.logger.info(f"Processed {len(job.processed_documents)} documents")
                self.logger.info(f"Total chunks: {job.total_chunks()}")
            else:
                self.logger.error(f"Document processing failed: {job.error_message}")

            self.logger.info("=" * 20)

            return job.model_dump()

        except Exception as e:
            self.logger.error("=" * 20)
            self.logger.error(f"CRITICAL ERROR during document processing: {str(e)}")
            self.logger.error("=" * 20, exc_info=True)
            return None

    def create_dataset(self, chunks_files: list[str], output_dir: str, dataset_name: str, description: str = "") -> dict[str, Any] | None:
        """
        Create a fine-tuning dataset from processed chunks.

        Args:
            chunks_files: List of paths to chunk files.
            output_dir: Directory to save the dataset.
            dataset_name: Name of the dataset.
            description: Optional dataset description.

        Returns:
            Optional[Dict[str, Any]]: Dataset configuration or None if error occurs.
        """
        self.logger.info("=" * 40)
        self.logger.info(f"Creating dataset: {dataset_name}")
        self.logger.info("=" * 40)

        try:
            config = self.dataset_manager.create_dataset_from_chunks(chunks_files, output_dir, dataset_name, description)

            self.logger.info("Dataset created successfully")
            self.logger.info(f"Total samples: {config.total_samples}")
            self.logger.info(f"Approximate tokens: {config.total_tokens}")
            self.logger.info(f"Location: {config.dataset_dir}")
            self.logger.info("=" * 40)

            return config.model_dump()

        except Exception as e:
            self.logger.error("=" * 40)
            self.logger.error(f"CRITICAL ERROR during dataset creation: {str(e)}")
            self.logger.error("=" * 40, exc_info=True)
            return None
