"""
Module core.data.dataset_manager
---------------------------------

This module manages datasets for fine-tuning operations.
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..configuration.data_schema import DatasetConfig, DatasetSample


class DatasetManager:
    """
    Manager for creating and managing fine-tuning datasets.

    This class handles dataset creation, splitting, and validation.
    """

    def __init__(self, logger_name: str = "app.data.dataset_manager"):
        """
        Initialize the dataset manager.

        Args:
            logger_name: Name of the logger for this component.
        """
        self.logger = logging.getLogger(logger_name)

    def create_dataset_from_chunks(self, chunks_files: list[str], output_dir: str, dataset_name: str, description: str = "") -> DatasetConfig:
        """
        Create a dataset from processed chunk files.

        Args:
            chunks_files: List of paths to JSONL chunk files.
            output_dir: Directory to save the dataset.
            dataset_name: Name of the dataset.
            description: Optional description of the dataset.

        Returns:
            DatasetConfig: Configuration for the created dataset.

        Raises:
            Exception: If dataset creation fails.
        """
        self.logger.info(f"Creating dataset '{dataset_name}' from {len(chunks_files)} chunk files")

        try:
            # Create output directory
            dataset_dir = Path(output_dir) / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Load all samples
            all_samples: list[DatasetSample] = []
            total_tokens = 0

            for chunks_file in chunks_files:
                chunks_path = Path(chunks_file)
                if not chunks_path.exists():
                    self.logger.warning(f"Chunks file not found: {chunks_file}")
                    continue

                with chunks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            sample = DatasetSample(**data)
                            all_samples.append(sample)
                            # Rough token count estimation (1 token ≈ 4 characters)
                            total_tokens += len(sample.text) // 4
                        except Exception as e:
                            self.logger.warning(f"Failed to parse sample: {e}")
                            continue

            if not all_samples:
                raise ValueError("No valid samples found in chunk files")

            self.logger.info(f"Loaded {len(all_samples)} samples with ~{total_tokens} tokens")

            # Sort samples by document name and chunk index to preserve sequential order
            all_samples.sort(key=lambda x: (x.id.rsplit("_", 1)[0], x.metadata.get("chunk_index", 0)))  # Document name  # Chunk index within document
            self.logger.info(f"Sorted {len(all_samples)} samples by document and chunk order for sequential split")

            # Create dataset config (format is always JSONL)
            config = DatasetConfig(
                name=dataset_name,
                description=description,
                dataset_dir=str(dataset_dir),
                total_documents=len(chunks_files),
                total_samples=len(all_samples),
                total_tokens=total_tokens,
            )

            # Split and save dataset
            self._split_and_save_dataset(all_samples, config)

            # Save dataset config
            config_path = dataset_dir / "dataset_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config.model_dump(), f, indent=2)

            self.logger.info(f"Dataset created successfully at {dataset_dir}")

            return config

        except Exception as e:
            self.logger.error(f"Failed to create dataset: {e}", exc_info=True)
            raise

    def _split_and_save_dataset(self, samples: list[DatasetSample], config: DatasetConfig) -> None:
        """
        Split dataset into train/validation/test sets and save them.

        Args:
            samples: List of all samples.
            config: Dataset configuration with split ratios.
        """
        total = len(samples)
        train_size = int(total * config.train_split)
        val_size = int(total * config.validation_split)

        train_samples = samples[:train_size]
        val_samples = samples[train_size : train_size + val_size]
        test_samples = samples[train_size + val_size :]

        self.logger.info(f"Dataset split: {len(train_samples)} train, {len(val_samples)} validation, {len(test_samples)} test")

        # Save splits
        if train_samples:
            self._save_samples(train_samples, config.get_train_file())
        if val_samples:
            self._save_samples(val_samples, config.get_validation_file())
        if test_samples:
            self._save_samples(test_samples, config.get_test_file())

    def _save_samples(self, samples: list[DatasetSample], output_path: Path) -> None:
        """
        Save samples to a JSONL file.

        Args:
            samples: List of samples to save.
            output_path: Path to the output file.
        """
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + "\n")

        self.logger.debug(f"Saved {len(samples)} samples to {output_path}")

    def load_dataset_config(self, dataset_dir: str) -> DatasetConfig:
        """
        Load dataset configuration from a directory.

        Args:
            dataset_dir: Path to the dataset directory.

        Returns:
            DatasetConfig: Loaded dataset configuration.

        Raises:
            FileNotFoundError: If config file not found.
        """
        config_path = Path(dataset_dir) / "dataset_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return DatasetConfig(**data)

    def validate_dataset(self, config: DatasetConfig) -> dict[str, Any]:
        """
        Validate a dataset and return validation report.

        Args:
            config: Dataset configuration to validate.

        Returns:
            dict: Validation report with statistics and issues.
        """
        self.logger.info(f"Validating dataset: {config.name}")

        report: dict[str, Any] = {
            "valid": True,
            "issues": [],
            "statistics": {
                "train_samples": 0,
                "val_samples": 0,
                "test_samples": 0,
                "total_samples": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
            },
        }

        try:
            dataset_path = config.get_dataset_path()
            if not dataset_path.exists():
                report["valid"] = False
                report["issues"].append(f"Dataset directory not found: {dataset_path}")
                return report

            # Check train file
            train_samples = self._count_samples(config.get_train_file())
            report["statistics"]["train_samples"] = train_samples
            if train_samples == 0:
                report["issues"].append("No training samples found")

            # Check validation file
            val_samples = self._count_samples(config.get_validation_file())
            report["statistics"]["val_samples"] = val_samples

            # Check test file
            test_samples = self._count_samples(config.get_test_file())
            report["statistics"]["test_samples"] = test_samples

            total = train_samples + val_samples + test_samples
            report["statistics"]["total_samples"] = total

            if total == 0:
                report["valid"] = False
                report["issues"].append("No samples found in dataset")

            # Calculate length statistics from train file
            if train_samples > 0:
                lengths = self._get_sample_lengths(config.get_train_file())
                if lengths:
                    report["statistics"]["avg_length"] = sum(lengths) // len(lengths)
                    report["statistics"]["min_length"] = min(lengths)
                    report["statistics"]["max_length"] = max(lengths)

            if report["issues"]:
                report["valid"] = len([i for i in report["issues"] if "not found" not in i.lower()]) == 0

        except Exception as e:
            report["valid"] = False
            report["issues"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Dataset validation failed: {e}", exc_info=True)

        return report

    def _count_samples(self, file_path: Path) -> int:
        """Count the number of samples in a JSONL file."""
        if not file_path.exists():
            return 0

        count = 0
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception as e:
            self.logger.warning(f"Error counting samples in {file_path}: {e}")

        return count

    def _get_sample_lengths(self, file_path: Path, max_samples: int = 1000) -> list[int]:
        """Get text lengths from samples (limited to max_samples for efficiency)."""
        lengths: list[int] = []

        if not file_path.exists():
            return lengths

        try:
            with file_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        lengths.append(len(text))
                    except Exception:
                        continue
        except Exception as e:
            self.logger.warning(f"Error reading sample lengths from {file_path}: {e}")

        return lengths

    def list_datasets(self, datasets_root: str) -> list[DatasetConfig]:
        """
        List all available datasets in a directory.

        Args:
            datasets_root: Root directory containing datasets.

        Returns:
            list[DatasetConfig]: List of dataset configurations.
        """
        datasets: list[DatasetConfig] = []
        root_path = Path(datasets_root)

        if not root_path.exists():
            self.logger.warning(f"Datasets root directory not found: {datasets_root}")
            return datasets

        # Look for dataset_config.json files
        for config_file in root_path.rglob("dataset_config.json"):
            try:
                config = self.load_dataset_config(str(config_file.parent))
                datasets.append(config)
            except Exception as e:
                self.logger.warning(f"Failed to load dataset config from {config_file}: {e}")

        return sorted(datasets, key=lambda d: d.name)
