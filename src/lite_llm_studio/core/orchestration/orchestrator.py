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

    def execute_training(self, training_config: dict[str, Any]) -> dict[str, Any] | None:
        """
        Execute model fine-tuning with LoRA adapters.

        This method coordinates the fine-tuning process using the configuration
        provided. It validates the inputs, calls the training module, and returns
        the results.

        Args:
            training_config: Dictionary containing training configuration:
                - dataset_dir: Path to dataset directory (with train.jsonl, validation.jsonl)
                - base_model_path: Path to the base model
                - output_dir: Directory to save the fine-tuned model
                - epochs: Number of training epochs
                - batch_size: Batch size per device
                - learning_rate: Learning rate for optimizer
                - max_seq_length: Maximum sequence length
                - lora_r: LoRA rank
                - lora_alpha: LoRA alpha parameter
                - lora_dropout: LoRA dropout rate
                - gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            Optional[Dict[str, Any]]: Training results containing:
                - trained_model_path: Path to the saved model
                - training_stats: Training statistics
                Or None if error occurs.
        """
        self.logger.info("=" * 40)
        self.logger.info("Starting model fine-tuning")
        self.logger.info("=" * 40)

        try:
            # Import training module
            from ..ml.training import TrainingConfig, train_lora_model

            # Create training configuration
            config = TrainingConfig(
                dataset_dir=training_config["dataset_dir"],
                base_model_path=training_config["base_model_path"],
                output_dir=training_config["output_dir"],
                epochs=training_config.get("epochs", 1),
                batch_size=training_config.get("batch_size", 4),
                learning_rate=training_config.get("learning_rate", 2e-4),
                max_seq_length=training_config.get("max_seq_length", 1024),
                lora_r=training_config.get("lora_r", 8),
                lora_alpha=training_config.get("lora_alpha", 16),
                lora_dropout=training_config.get("lora_dropout", 0.05),
                gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            )

            self.logger.info(f"Dataset: {config.dataset_dir}")
            self.logger.info(f"Base model: {config.base_model_path}")
            self.logger.info(f"Output directory: {config.output_dir}")
            self.logger.info(f"Training parameters:")
            self.logger.info(f"  - Epochs: {config.epochs}")
            self.logger.info(f"  - Batch size: {config.batch_size}")
            self.logger.info(f"  - Learning rate: {config.learning_rate}")
            self.logger.info(f"  - Max sequence length: {config.max_seq_length}")
            self.logger.info(f"  - LoRA r: {config.lora_r}")
            self.logger.info(f"  - LoRA alpha: {config.lora_alpha}")
            self.logger.info(f"  - LoRA dropout: {config.lora_dropout}")
            self.logger.info(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")

            # Execute training
            self.logger.info("Initiating fine-tuning process...")
            trained_model_path = train_lora_model(config)

            self.logger.info("=" * 40)
            self.logger.info("Model fine-tuning completed successfully")
            self.logger.info(f"Fine-tuned model saved to: {trained_model_path}")
            self.logger.info("=" * 40)

            return {
                "trained_model_path": trained_model_path,
                "config": training_config,
                "status": "success",
            }

        except Exception as e:
            self.logger.error("=" * 40)
            self.logger.error(f"CRITICAL ERROR during model training: {str(e)}")
            self.logger.error("=" * 40, exc_info=True)
            return None
