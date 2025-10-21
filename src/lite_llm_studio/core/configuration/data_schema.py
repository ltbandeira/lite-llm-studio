"""
Module core.configuration.data_schema
--------------------------------------

This module contains the data processing and dataset configuration schemas
for Causal Language Modeling with JSONL format.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from .base_config import BaseConfigModel


class ChunkingStrategy(str, Enum):
    """Strategies for chunking documents."""

    HYBRID = "hybrid"  # Docling's HybridChunker - hierarchical + tokenization-aware
    HIERARCHICAL = "hierarchical"  # Docling's HierarchicalChunker - one chunk per document element
    FIXED_SIZE = "fixed_size"  # Legacy: simple word-based chunking
    PARAGRAPH = "paragraph"  # Legacy: split by paragraphs


class DataProcessingConfig(BaseConfigModel):
    """Configuration for document processing with Docling for Causal Language Modeling."""

    # Input configuration
    input_files: list[str] = Field(default_factory=list, description="List of input PDF file paths")
    output_dir: str = Field(..., description="Directory to save processed outputs")

    # Processing options (format is always JSONL)
    extract_tables: bool = Field(default=True, description="Whether to extract and format tables")
    ocr_enabled: bool = Field(default=True, description="Enable OCR for scanned documents")

    # Chunking configuration
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.HYBRID, description="Strategy for chunking documents")
    chunk_size: int = Field(default=512, ge=128, le=4096, description="Size of chunks in tokens (for fixed_size strategy)")
    chunk_overlap: int = Field(default=50, ge=0, le=512, description="Overlap between chunks in tokens (for fixed_size strategy)")

    # Advanced chunking options (for Docling native chunkers)
    max_tokens: int = Field(default=512, ge=64, le=8192, description="Maximum tokens per chunk (for hybrid/hierarchical strategies)")
    tokenizer_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace tokenizer model for token-aware chunking")
    merge_peers: bool = Field(default=True, description="Merge undersized peer chunks with same headings (for hybrid strategy)")

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory is a valid path."""
        path = Path(v)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class ProcessedDocument(BaseConfigModel):
    """Represents a processed document."""

    source_file: str = Field(..., description="Path to the source PDF file")
    output_file: str = Field(..., description="Path to the processed output JSONL file")
    page_count: int = Field(default=0, ge=0, description="Number of pages in the document")
    word_count: int = Field(default=0, ge=0, description="Approximate word count")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks created")
    processing_time: float = Field(default=0.0, ge=0, description="Processing time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def get_size_mb(self) -> float:
        """Get the size of the output file in MB."""
        try:
            output_path = Path(self.output_file)
            if output_path.exists():
                return output_path.stat().st_size / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class DatasetConfig(BaseConfigModel):
    """Configuration for a fine-tuning dataset in JSONL format."""

    name: str = Field(..., description="Name of the dataset")
    description: str = Field(default="", description="Description of the dataset")
    dataset_dir: str = Field(..., description="Directory containing the dataset files")

    # Dataset statistics
    total_documents: int = Field(default=0, ge=0, description="Total number of source documents")
    total_samples: int = Field(default=0, ge=0, description="Total number of training samples")
    total_tokens: int = Field(default=0, ge=0, description="Approximate total token count")

    # Training split configuration
    train_split: float = Field(default=0.8, ge=0.0, le=1.0, description="Fraction of data for training")
    validation_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of data for validation")
    test_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of data for testing")

    @field_validator("validation_split", "test_split")
    @classmethod
    def validate_splits_sum(cls, v: float, info) -> float:
        """Ensure splits are valid."""
        if v < 0 or v > 1:
            raise ValueError("Split values must be between 0 and 1")
        return v

    def get_dataset_path(self) -> Path:
        """Get the full path to the dataset directory."""
        return Path(self.dataset_dir)

    def get_train_file(self) -> Path:
        """Get the path to the training data file."""
        return self.get_dataset_path() / "train.jsonl"

    def get_validation_file(self) -> Path:
        """Get the path to the validation data file."""
        return self.get_dataset_path() / "validation.jsonl"

    def get_test_file(self) -> Path:
        """Get the path to the test data file."""
        return self.get_dataset_path() / "test.jsonl"

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class DatasetSample(BaseConfigModel):
    """Represents a single sample in the dataset for Causal Language Modeling."""

    id: str = Field(..., description="Unique identifier for this sample")
    text: str = Field(..., description="The text content for causal language modeling")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()


class ProcessingStatus(str, Enum):
    """Status of document processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJob(BaseConfigModel):
    """Represents a document processing job."""

    job_id: str = Field(..., description="Unique job identifier")
    config: DataProcessingConfig = Field(..., description="Processing configuration")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Processing progress percentage")
    processed_documents: list[ProcessedDocument] = Field(default_factory=list, description="List of processed documents")
    error_message: str | None = Field(default=None, description="Error message if processing failed")
    started_at: str | None = Field(default=None, description="Timestamp when processing started")
    completed_at: str | None = Field(default=None, description="Timestamp when processing completed")

    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]

    def total_chunks(self) -> int:
        """Get the total number of chunks across all documents."""
        return sum(doc.chunk_count for doc in self.processed_documents)

    def to_dict(self):
        return super().to_dict()

    def to_json(self):
        return super().to_json()
