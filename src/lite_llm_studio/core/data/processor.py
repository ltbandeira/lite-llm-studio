"""
Module core.data.processor
---------------------------

This module provides document processing functionality using Docling for
Causal Language Modeling. Output format is always JSONL.
"""

import json
import logging
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..configuration.data_schema import (
    ChunkingStrategy,
    DataProcessingConfig,
    ProcessedDocument,
    ProcessingJob,
    ProcessingStatus,
)


class DocumentProcessor:
    """
    Document processor using Docling for PDF processing.

    This class handles the conversion of PDF documents to JSONL format,
    extraction of text and tables, and chunking for causal language modeling.
    """

    def __init__(self, logger_name: str = "app.data.processor"):
        """
        Initialize the document processor.

        Args:
            logger_name: Name of the logger for this component.
        """
        self.logger = logging.getLogger(logger_name)
        self._docling_available = self._check_docling_availability()

    def _check_docling_availability(self) -> bool:
        """
        Check if Docling is available and properly installed.

        Returns:
            bool: True if Docling is available, False otherwise.
        """
        try:
            import docling  # noqa: F401

            self.logger.info("Docling is available and ready to use")
            return True
        except ImportError:
            self.logger.warning("Docling is not installed. Document processing will be limited.")
            return False

    def _normalize_text(self, text: str) -> str:
        """
        Normalize Unicode text and fix encoding issues.

        Args:
            text: Input text that may have encoding issues.

        Returns:
            str: Normalized and cleaned text.
        """
        # Step 1: Normalize to NFC form (canonical composition)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Try to fix mojibake with ftfy if available
        try:
            import ftfy

            text = ftfy.fix_text(text)
        except ImportError:
            self.logger.debug("ftfy not available, skipping advanced text fixing")

        return text

    def create_processing_job(self, config: DataProcessingConfig) -> ProcessingJob:
        """
        Create a new processing job.

        Args:
            config: Configuration for the processing job.

        Returns:
            ProcessingJob: A new processing job with pending status.
        """
        job = ProcessingJob(job_id=str(uuid4()), config=config, status=ProcessingStatus.PENDING, progress=0.0, processed_documents=[])

        self.logger.info(f"Created processing job {job.job_id} with {len(config.input_files)} files")
        return job

    def process_job(self, job: ProcessingJob) -> ProcessingJob:
        """
        Process a document processing job.

        Args:
            job: The processing job to execute.

        Returns:
            ProcessingJob: Updated job with processing results.
        """
        if not self._docling_available:
            job.status = ProcessingStatus.FAILED
            job.error_message = "Docling is not installed. Please install docling to process documents."
            self.logger.error(job.error_message)
            return job

        job.status = ProcessingStatus.PROCESSING
        job.started_at = datetime.now().isoformat()
        job.progress = 0.0

        self.logger.info(f"Starting processing job {job.job_id}")

        try:
            total_files = len(job.config.input_files)
            if total_files == 0:
                raise ValueError("No input files provided")

            processed_docs: list[ProcessedDocument] = []

            for idx, file_path in enumerate(job.config.input_files):
                self.logger.info(f"Processing file {idx + 1}/{total_files}: {file_path}")

                try:
                    # Process single document
                    processed_doc = self._process_document(file_path, job.config)
                    processed_docs.append(processed_doc)

                    # Update progress
                    job.progress = ((idx + 1) / total_files) * 100
                    self.logger.debug(f"Progress: {job.progress:.1f}%")

                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                    # Continue processing other files
                    continue

            job.processed_documents = processed_docs
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
            job.progress = 100.0

            self.logger.info(f"Job {job.job_id} completed successfully. Processed {len(processed_docs)}/{total_files} files")

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now().isoformat()
            self.logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)

        return job

    def _process_document(self, file_path: str, config: DataProcessingConfig) -> ProcessedDocument:
        """
        Process a single PDF document.

        Args:
            file_path: Path to the PDF file.
            config: Processing configuration.

        Returns:
            ProcessedDocument: Metadata about the processed document.

        Raises:
            Exception: If processing fails.
        """
        start_time = time.time()
        source_path = Path(file_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        if not source_path.suffix.lower() == ".pdf":
            raise ValueError(f"Only PDF files are supported. Got: {source_path.suffix}")

        # Import Docling components
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        self.logger.debug(f"Converting document: {source_path.name}")

        # Configure PDF pipeline options based on OCR settings
        pipeline_options = PdfPipelineOptions()

        if config.ocr_enabled:
            # Enable OCR with Portuguese language support
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options.lang = ["pt"]  # Portuguese language for OCR
            self.logger.debug("OCR enabled with Portuguese language support")
        else:
            # Disable OCR for documents with text layer
            pipeline_options.do_ocr = False
            self.logger.debug("OCR disabled - processing text layer only")

        # Create format options with pipeline configuration
        pdf_options = PdfFormatOption(pipeline_options=pipeline_options)

        # Initialize converter with configured options
        converter = DocumentConverter(format_options={InputFormat.PDF: pdf_options})

        # Convert document
        result = converter.convert(str(source_path))

        # Keep the DoclingDocument for advanced chunking
        docling_document = result.document

        # Extract text for statistics and legacy chunking
        text_content = docling_document.export_to_text()

        # Normalize Unicode and fix encoding issues
        text_content = self._normalize_text(text_content)
        self.logger.debug(f"Text normalized. Length: {len(text_content)} characters")

        # Determine output filename
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = source_path.stem
        output_path = output_dir / f"{output_filename}.jsonl"

        # Calculate statistics from text
        word_count = len(text_content.split())

        # Get page count - try different methods to get the page count
        page_count = 0
        try:
            # Try getting num_pages as a property
            if hasattr(docling_document, "num_pages"):
                num_pages_attr = docling_document.num_pages
                # Check if it's a method or property
                if callable(num_pages_attr):
                    page_count = num_pages_attr()
                else:
                    page_count = num_pages_attr
            # Fallback: count pages from document structure
            elif hasattr(docling_document, "pages") and docling_document.pages:
                page_count = len(docling_document.pages)
            # Another fallback: try _pages
            elif hasattr(docling_document, "_pages") and docling_document._pages:
                page_count = len(docling_document._pages)
        except Exception as e:
            self.logger.debug(f"Could not determine page count: {e}")
            page_count = 0

        # Apply chunking (using Docling document for advanced strategies)
        chunks = self._chunk_document(docling_document, text_content, config)

        # Export to JSONL format
        jsonl_content = self._export_to_jsonl(chunks, str(source_path))

        # Write output
        output_path.write_text(jsonl_content, encoding="utf-8")

        # Create processed document metadata
        processing_time = time.time() - start_time
        processed_doc = ProcessedDocument(
            source_file=str(source_path),
            output_file=str(output_path),
            page_count=page_count,
            word_count=word_count,
            chunk_count=len(chunks),
            processing_time=processing_time,
            metadata={"chunking_strategy": config.chunking_strategy.value},
        )

        # Save chunks separately for dataset creation
        chunks_path = output_dir / f"{output_filename}_chunks.jsonl"
        self._save_chunks(chunks, chunks_path, str(source_path))
        processed_doc.metadata["chunks_file"] = str(chunks_path)

        if not chunks:
            self.logger.warning(f"No chunks generated for {source_path.name}")

        self.logger.info(
            f"Processed {source_path.name}: " f"{page_count} pages, {word_count} words, " f"{len(chunks)} chunks in {processing_time:.2f}s"
        )

        return processed_doc

    def _export_to_jsonl(self, chunks: list[dict], source_file: str) -> str:
        """
        Export chunks to JSONL format.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'.
            source_file: Source file path for metadata.

        Returns:
            str: JSONL formatted content.
        """
        # Convert chunks to JSONL
        lines = []
        for idx, chunk_data in enumerate(chunks):
            # Merge with source file metadata
            metadata = chunk_data.get("metadata", {}).copy()
            metadata["source"] = source_file
            metadata["total_chunks"] = len(chunks)

            entry = {"id": idx, "text": chunk_data["text"], "metadata": metadata}
            lines.append(json.dumps(entry, ensure_ascii=False))

        return "\n".join(lines)

    def _chunk_document(self, docling_doc, text_content: str, config: DataProcessingConfig) -> list[dict]:
        """
        Chunk document using the configured strategy.

        Args:
            docling_doc: The DoclingDocument object from Docling converter.
            text_content: Plain text content for fallback mechanisms.
            config: Processing configuration.

        Returns:
            list[dict]: List of chunk dictionaries with text and metadata.
        """
        self.logger.debug(f"Chunking document with strategy: {config.chunking_strategy}")

        if config.chunking_strategy == ChunkingStrategy.HYBRID:
            return self._chunk_with_hybrid(docling_doc, config)

        elif config.chunking_strategy == ChunkingStrategy.HIERARCHICAL:
            return self._chunk_with_hierarchical(docling_doc, config)

        else:
            # Fallback to paragraph chunking for unknown strategies
            self.logger.warning(f"Unknown chunking strategy: {config.chunking_strategy}, using paragraph fallback")
            return self._chunk_by_paragraph(text_content)

    def _chunk_with_hybrid(self, docling_doc, config: DataProcessingConfig) -> list[dict]:
        """
        Chunk using Docling's HybridChunker (recommended).

        Applies tokenization-aware refinements on top of hierarchical chunking.
        Respects token limits and preserves document structure.

        Args:
            docling_doc: The DoclingDocument object.
            config: Processing configuration.

        Returns:
            list[dict]: List of enriched chunks with metadata.
        """
        try:
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from transformers import AutoTokenizer

            self.logger.info(f"Using HybridChunker with tokenizer: {config.tokenizer_model}")

            # Initialize tokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
            tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=config.max_tokens)

            # Initialize chunker
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=config.merge_peers)

            # Chunk document
            chunks = []
            for idx, chunk in enumerate(chunker.chunk(dl_doc=docling_doc)):
                # Get context-enriched text (includes headings/captions)
                enriched_text = chunker.contextualize(chunk=chunk)

                # Extract metadata
                metadata = {
                    "chunk_index": idx,
                    "chunking_strategy": "hybrid",
                    "tokenizer_model": config.tokenizer_model,
                }

                # Add hierarchical metadata if available
                if hasattr(chunk, "meta") and chunk.meta:
                    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                        metadata["headings"] = chunk.meta.headings
                    if hasattr(chunk.meta, "origin") and chunk.meta.origin:
                        if hasattr(chunk.meta.origin, "page"):
                            metadata["page"] = chunk.meta.origin.page

                # Calculate token count
                token_count = tokenizer.count_tokens(enriched_text)
                metadata["token_count"] = token_count

                chunks.append({"text": enriched_text, "metadata": metadata})

            self.logger.info(f"HybridChunker produced {len(chunks)} chunks")
            return chunks

        except ImportError as e:
            self.logger.error(f"Failed to import HybridChunker dependencies: {e}")
            self.logger.warning("Falling back to paragraph chunking")
            text_content = docling_doc.export_to_text()
            return self._chunk_by_paragraph(text_content)
        except Exception as e:
            self.logger.error(f"Error in HybridChunker: {e}", exc_info=True)
            self.logger.warning("Falling back to paragraph chunking")
            text_content = docling_doc.export_to_text()
            return self._chunk_by_paragraph(text_content)

    def _chunk_with_hierarchical(self, docling_doc, config: DataProcessingConfig) -> list[dict]:
        """
        Chunk using Docling's HierarchicalChunker.

        Creates one chunk per document element, preserving structure.

        Args:
            docling_doc: The DoclingDocument object.
            config: Processing configuration.

        Returns:
            list[dict]: List of chunks with metadata.
        """
        try:
            from docling.chunking import HierarchicalChunker

            self.logger.info("Using HierarchicalChunker")

            # Initialize chunker
            chunker = HierarchicalChunker()

            # Chunk document
            chunks = []
            for idx, chunk in enumerate(chunker.chunk(dl_doc=docling_doc)):
                enriched_text = chunker.contextualize(chunk=chunk)

                metadata = {
                    "chunk_index": idx,
                    "chunking_strategy": "hierarchical",
                }

                # Add metadata if available
                if hasattr(chunk, "meta") and chunk.meta:
                    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                        metadata["headings"] = chunk.meta.headings
                    if hasattr(chunk.meta, "origin") and chunk.meta.origin:
                        if hasattr(chunk.meta.origin, "page"):
                            metadata["page"] = chunk.meta.origin.page

                chunks.append({"text": enriched_text, "metadata": metadata})

            self.logger.info(f"HierarchicalChunker produced {len(chunks)} chunks")
            return chunks

        except ImportError as e:
            self.logger.error(f"Failed to import HierarchicalChunker: {e}")
            self.logger.warning("Falling back to paragraph chunking")
            text_content = docling_doc.export_to_text()
            return self._chunk_by_paragraph(text_content)
        except Exception as e:
            self.logger.error(f"Error in HierarchicalChunker: {e}", exc_info=True)
            self.logger.warning("Falling back to paragraph chunking")
            text_content = docling_doc.export_to_text()
            return self._chunk_by_paragraph(text_content)

    def _chunk_by_paragraph(self, content: str) -> list[dict]:
        """
        Internal fallback: Split content by paragraphs.

        This method is kept as an emergency fallback when Docling
        chunkers fail. Not exposed in the public API.

        Args:
            content: Text content to chunk.

        Returns:
            list[dict]: List of paragraph chunks.
        """
        # Split by double newlines (paragraphs)
        content_normalized = content.replace("\r\n", "\n").replace("\r", "\n")

        # Try double newlines first
        raw_chunks = [p.strip() for p in content_normalized.split("\n\n") if p.strip()]

        # If we get too few chunks, try single newlines with minimum length
        if len(raw_chunks) < 3:
            raw_chunks = [p.strip() for p in content_normalized.split("\n") if p.strip() and len(p.strip()) > 50]

        # If still no chunks, use the whole content
        if not raw_chunks and content.strip():
            raw_chunks = [content.strip()]

        chunks = []
        for idx, text in enumerate(raw_chunks):
            chunks.append({"text": text, "metadata": {"chunk_index": idx, "chunking_strategy": "paragraph"}})

        self.logger.debug(f"Paragraph chunking produced {len(chunks)} chunks")
        return chunks

    def _chunk_by_fixed_size(self, content: str, config: DataProcessingConfig) -> list[dict]:
        """
        Internal fallback: Simple word-based chunking with fixed size and overlap.

        This method is kept for internal use only, not exposed in the public API.
        Uses legacy chunk_size and chunk_overlap parameters if needed.

        Args:
            content: Text content to chunk.
            config: Processing configuration.

        Returns:
            list[dict]: List of fixed-size chunks.
        """
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0

        for word in words:
            current_chunk.append(word)
            current_size += 1

            if current_size >= config.chunk_size:
                text = " ".join(current_chunk)
                chunks.append(
                    {"text": text, "metadata": {"chunk_index": chunk_idx, "chunking_strategy": "fixed_size", "word_count": len(current_chunk)}}
                )
                chunk_idx += 1

                # Apply overlap
                overlap_words = current_chunk[-config.chunk_overlap :] if config.chunk_overlap > 0 else []
                current_chunk = overlap_words
                current_size = len(overlap_words)

        # Add remaining words
        if current_chunk:
            text = " ".join(current_chunk)
            chunks.append({"text": text, "metadata": {"chunk_index": chunk_idx, "chunking_strategy": "fixed_size", "word_count": len(current_chunk)}})

        self.logger.debug(f"Fixed-size chunking produced {len(chunks)} chunks")
        return chunks

    def _save_chunks(self, chunks: list[dict], output_path: Path, source_file: str) -> None:
        """
        Save chunks to a JSONL file with proper encoding.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'.
            output_path: Path to save the chunks.
            source_file: Source file path for metadata.
        """
        with output_path.open("w", encoding="utf-8") as f:
            for idx, chunk_data in enumerate(chunks):
                # Merge metadata
                metadata = chunk_data.get("metadata", {}).copy()
                metadata["source"] = str(Path(source_file).name)
                metadata["total_chunks"] = len(chunks)

                entry = {
                    "id": f"{Path(source_file).stem}_{idx}",
                    "text": chunk_data["text"],
                    "metadata": metadata,
                }
                # Use ensure_ascii=False to preserve Unicode characters properly
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self.logger.debug(f"Saved {len(chunks)} chunks to {output_path}")
