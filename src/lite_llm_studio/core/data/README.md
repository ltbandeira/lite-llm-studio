# Core Data Module

This module provides comprehensive data handling functionality for LiteLLM Studio, including document processing, file upload management, and training dataset creation.

## Features

### Document Processing (`processors.py`)
- **Multi-format Support**: Processes PDF, Word (.docx), PowerPoint (.pptx), Text files, and Markdown
- **Docling Integration**: Uses advanced document conversion with fallback mechanisms
- **Robust Error Handling**: Gracefully handles processing failures with detailed error messages
- **Metadata Extraction**: Captures file statistics, word counts, and processing information

### Upload Management (`upload_manager.py`)
- **Temporary File Storage**: Secure handling of uploaded files with unique identifiers
- **Memory Efficient**: Streams file content without keeping everything in memory
- **Automatic Cleanup**: Manages temporary files and directories lifecycle
- **File Sanitization**: Ensures safe filenames for filesystem storage

### User Interface Components
- **Drag & Drop Interface**: Modern file upload experience
- **Processing Progress**: Real-time feedback on document processing
- **Dataset Statistics**: Comprehensive overview of processed data
- **Data Preview**: Quick content inspection before training
- **Error Reporting**: Clear display of processing issues

## Dependencies

### Required
- `streamlit>=1.49.1`: Web interface framework
- `pydantic==2.11.7`: Data validation and settings management
- `pathlib`: File system operations (built-in)
- `tempfile`: Temporary file management (built-in)

### Optional (Enhanced Processing)
Install with: `pip install -e ".[data]"`

- `docling>=1.0.0`: Advanced document conversion
- `PyPDF2>=3.0.0`: PDF text extraction fallback
- `python-docx>=0.8.11`: Word document processing fallback

## Usage

### Basic Document Processing

```python
from lite_llm_studio.core.data import DocumentProcessor, DatasetBuilder

# Initialize processor
processor = DocumentProcessor()

# Process single file
result = processor.process_file("path/to/document.pdf")
print(f"Extracted {result.word_count} words from {result.filename}")

# Process multiple files
files = ["doc1.pdf", "doc2.docx", "doc3.txt"]
results = processor.process_multiple_files(files)

# Build training dataset
builder = DatasetBuilder()
dataset = builder.build_training_dataset(results)
```

### Upload Management

```python
from lite_llm_studio.core.data import UploadManager

# Initialize upload manager
manager = UploadManager()

# Save uploaded file (from Streamlit)
uploaded_file = manager.save_streamlit_uploaded_file(st_file_object)
print(f"Saved file: {uploaded_file.original_filename}")

# Get upload statistics
stats = manager.get_upload_statistics()
print(f"Total files: {stats['total_files']}, Size: {stats['total_size_mb']} MB")
```

### Integration with Streamlit

```python
import streamlit as st
from lite_llm_studio.app.components.data_components import (
    create_file_upload_area,
    create_upload_progress_display,
    create_dataset_statistics_display
)

# Create upload interface
uploaded_files = create_file_upload_area()

if uploaded_files:
    # Process files
    processor = DocumentProcessor()
    results = []
    
    for file in uploaded_files:
        # Process each file
        result = processor.process_file(file)
        results.append(result)
    
    # Display results
    create_upload_progress_display(results)
    
    # Show dataset statistics
    builder = DatasetBuilder()
    dataset_info = builder.build_training_dataset(results)
    create_dataset_statistics_display(dataset_info['statistics'])
```

## Architecture

### Processing Pipeline

1. **File Upload**: Users drag & drop or select files through Streamlit interface
2. **Temporary Storage**: Files saved securely with unique identifiers
3. **Document Processing**: Content extraction using docling with fallback methods
4. **Text Cleaning**: Optional normalization and filtering based on user preferences
5. **Dataset Building**: Aggregation of processed documents with statistics
6. **Preview & Export**: User can preview content and save training datasets

### Error Handling Strategy

- **Graceful Degradation**: Falls back to simpler extraction methods if docling fails
- **Detailed Logging**: Comprehensive error messages for troubleshooting
- **Partial Success**: Processes what it can, reports what failed
- **User Feedback**: Clear status indicators in the interface

### Security Considerations

- **Filename Sanitization**: Removes unsafe characters from uploaded filenames
- **Temporary File Cleanup**: Automatic cleanup prevents disk space issues
- **Memory Management**: Streaming approach avoids loading large files entirely in memory
- **Input Validation**: Checks file types and sizes before processing

## Configuration

### Processing Options

Users can configure processing behavior through the interface:

- **Text Cleaning**: Remove extra whitespace and normalize punctuation
- **Header/Footer Removal**: Attempt to strip document headers and footers
- **Word Count Filters**: Set minimum and maximum word limits per document
- **Content Preview**: Configure how much content to show in previews

### File Type Support

| Extension | Primary Method | Fallback Method | Status |
|-----------|---------------|-----------------|---------|
| `.pdf` | Docling | PyPDF2 | ✅ Supported |
| `.docx` | Docling | python-docx | ✅ Supported |
| `.doc` | Docling | None | ⚠️ Limited |
| `.pptx` | Docling | None | ✅ Supported |
| `.ppt` | Docling | None | ⚠️ Limited |
| `.txt` | Direct read | None | ✅ Supported |
| `.md` | Direct read | None | ✅ Supported |
| `.rtf` | Docling | None | ⚠️ Limited |

## Performance Considerations

- **Chunked Processing**: Large files processed in manageable chunks
- **Progress Indicators**: Real-time feedback prevents user confusion
- **Async Operations**: Non-blocking processing where possible
- **Resource Monitoring**: Tracks memory and disk usage during processing

## Future Enhancements

- **Batch Processing**: Process multiple files in parallel
- **Content Validation**: Automatic quality checks for training data
- **Format Conversion**: Export datasets in various formats (JSON, CSV, etc.)
- **Integration APIs**: REST endpoints for programmatic access
- **Advanced Filtering**: Content-based filtering and deduplication