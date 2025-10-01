"""
Module app.modules.training_page
--------------------------------

This module contains the model training page content with data upload and processing capabilities.
"""

import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..components.data_components import (
    create_file_upload_area,
    create_upload_progress_display,
    create_dataset_statistics_display,
    create_processing_options,
    create_action_buttons,
    create_data_preview,
    create_error_display,
)

from lite_llm_studio.core.data.processors import DocumentProcessor, DatasetBuilder
from lite_llm_studio.core.data.upload_manager import UploadManager


def initialize_session_state():
    """Initialize session state variables for the training page."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_statistics" not in st.session_state:
        st.session_state.dataset_statistics = {}
    if "upload_manager" not in st.session_state:
        st.session_state.upload_manager = UploadManager()
    if "processing_errors" not in st.session_state:
        st.session_state.processing_errors = []


def process_uploaded_files(uploaded_files: List, processing_options: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process uploaded files and return processing results.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        processing_options: Dictionary containing processing configuration

    Returns:
        List of processing results
    """
    if not uploaded_files:
        return []

    processor = DocumentProcessor()
    upload_manager = st.session_state.upload_manager
    results = []

    with st.spinner("Processing uploaded files..."):
        progress_bar = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Save uploaded file to temporary storage
                saved_file = upload_manager.save_streamlit_uploaded_file(uploaded_file)

                # Process the file
                processed_doc = processor.process_file(saved_file.temp_path)

                # Apply processing options
                if processed_doc.processing_status == "success":
                    content = processed_doc.content

                    # Apply text cleaning if enabled
                    if processing_options.get("clean_text", True):
                        content = clean_text_content(content)

                    # Apply word count filters
                    min_words = processing_options.get("min_word_count", 10)
                    max_words = processing_options.get("max_word_count", 50000)

                    word_count = len(content.split())

                    if word_count < min_words:
                        processed_doc.processing_status = "error"
                        processed_doc.error_message = f"Document too short ({word_count} words, minimum {min_words})"
                    elif word_count > max_words:
                        # Truncate content
                        words = content.split()
                        content = " ".join(words[:max_words])
                        processed_doc.content = content
                        processed_doc.word_count = max_words

                # Convert to dictionary for storage
                result_dict = {
                    "filename": processed_doc.filename,
                    "content": processed_doc.content,
                    "metadata": processed_doc.metadata,
                    "word_count": processed_doc.word_count,
                    "file_type": processed_doc.file_type,
                    "size_bytes": processed_doc.size_bytes,
                    "status": processed_doc.processing_status,
                    "error_message": processed_doc.error_message,
                }

                results.append(result_dict)

            except Exception as e:
                error_dict = {
                    "filename": uploaded_file.name,
                    "content": "",
                    "metadata": {},
                    "word_count": 0,
                    "file_type": Path(uploaded_file.name).suffix.lower(),
                    "size_bytes": uploaded_file.size if hasattr(uploaded_file, "size") else 0,
                    "status": "error",
                    "error_message": str(e),
                }
                results.append(error_dict)

            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))

        progress_bar.empty()

    return results


def clean_text_content(content: str) -> str:
    """
    Clean and normalize text content.

    Args:
        content: Raw text content

    Returns:
        str: Cleaned text content
    """
    import re

    if not content:
        return content

    # Remove excessive whitespace
    content = re.sub(r"\s+", " ", content)

    # Remove excessive newlines
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content


def build_dataset_from_processed_files(processed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a training dataset from processed files.

    Args:
        processed_files: List of processed file information

    Returns:
        Dictionary containing dataset and statistics
    """
    # Convert processed files to ProcessedDocument objects
    from lite_llm_studio.core.data.processors import ProcessedDocument

    documents = []
    for file_info in processed_files:
        doc = ProcessedDocument(
            filename=file_info["filename"],
            content=file_info["content"],
            metadata=file_info["metadata"],
            word_count=file_info["word_count"],
            file_type=file_info["file_type"],
            size_bytes=file_info["size_bytes"],
            processing_status=file_info["status"],
            error_message=file_info.get("error_message"),
        )
        documents.append(doc)

    # Build dataset
    dataset_builder = DatasetBuilder()
    return dataset_builder.build_training_dataset(documents)


def render_training_page():
    """Render the training page with data upload and processing functionality."""
    # Initialize session state
    initialize_session_state()

    st.markdown("---")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Data Upload", "Training Config", "Training Status"])

    with tab1:
        # File upload section
        uploaded_files = create_file_upload_area()

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        # Processing options
        if st.session_state.uploaded_files:
            st.markdown("<br>", unsafe_allow_html=True)
            processing_options = create_processing_options()

            # Action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            actions = create_action_buttons()

            # Process files button
            if actions["process"]:
                if st.session_state.uploaded_files:
                    processed_results = process_uploaded_files(st.session_state.uploaded_files, processing_options)
                    st.session_state.processed_documents = processed_results

                    # Build dataset
                    dataset_result = build_dataset_from_processed_files(processed_results)
                    if dataset_result["status"] == "success":
                        st.session_state.dataset = dataset_result["dataset"]
                        st.session_state.dataset_statistics = dataset_result["statistics"]
                        st.success("Files processed successfully!")
                    else:
                        st.error(f"Dataset building failed: {dataset_result['message']}")
                else:
                    st.warning("Please upload files first.")

            # Clear all button
            if actions["clear"]:
                st.session_state.uploaded_files = []
                st.session_state.processed_documents = []
                st.session_state.dataset = None
                st.session_state.dataset_statistics = {}
                st.session_state.processing_errors = []
                st.session_state.upload_manager.clear_all_files()
                st.success("All data cleared!")
                st.rerun()

        # Display processing results
        if st.session_state.processed_documents:
            st.markdown("<br>", unsafe_allow_html=True)
            create_upload_progress_display(st.session_state.processed_documents)

        # Display dataset statistics
        if st.session_state.dataset_statistics:
            st.markdown("<br>", unsafe_allow_html=True)
            create_dataset_statistics_display(st.session_state.dataset_statistics)

        # Display data preview
        if st.session_state.dataset:
            st.markdown("<br>", unsafe_allow_html=True)
            create_data_preview(st.session_state.dataset, max_documents=2)

        # Display errors
        if st.session_state.processing_errors:
            create_error_display(st.session_state.processing_errors)

    with tab2:
        st.markdown("### Training Configuration")

    with tab3:
        st.markdown("### Training Status")
