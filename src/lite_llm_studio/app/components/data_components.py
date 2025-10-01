"""
Module app.components.data_components
------------------------------------

This module contains UI components for data handling and file uploads.
"""

from typing import List, Dict, Any, Optional
import streamlit as st


def create_file_upload_area() -> Optional[List]:
    """
    Create a drag and drop file upload area using Streamlit's file uploader.

    Returns:
        List of uploaded files or None if no files uploaded
    """
    st.markdown(
        """
        <div class="upload-section">
            <div class="section-title">Upload Training Data</div>
            <div class="section-sub">
                Drag and drop your documents here or click to browse. 
                Supported formats: PDF, Word (.docx), PowerPoint (.pptx), Text files, Markdown.
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "doc", "pptx", "ppt", "txt", "md", "rtf"],
        accept_multiple_files=True,
        help="Upload documents that will be processed and used for training data",
        label_visibility="collapsed",
    )

    return uploaded_files if uploaded_files else None


def create_upload_progress_display(processed_files: List[Dict[str, Any]]):
    """
    Display upload progress and processing results.

    Args:
        processed_files: List of processed file information
    """
    if not processed_files:
        return

    st.markdown(
        """
        <div class="upload-results">
            <div class="section-title">Processing Results</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Create columns for the file list
    for i, file_info in enumerate(processed_files):
        with st.container():
            cols = st.columns([3, 1, 1, 1, 2])

            with cols[0]:
                # File name with status icon
                status_icon = "✅" if file_info["status"] == "success" else "❌"
                st.markdown(f"{status_icon} **{file_info['filename']}**")

            with cols[1]:
                # File type
                st.markdown(f"`{file_info['file_type']}`")

            with cols[2]:
                # File size
                size_mb = file_info["size_bytes"] / (1024 * 1024)
                st.markdown(f"{size_mb:.1f} MB")

            with cols[3]:
                # Word count
                if file_info["status"] == "success":
                    st.markdown(f"{file_info['word_count']:,} words")
                else:
                    st.markdown("—")

            with cols[4]:
                # Status or error message
                if file_info["status"] == "success":
                    st.success("Processed")
                else:
                    st.error(f"Error: {file_info.get('error_message', 'Unknown error')}")

        if i < len(processed_files) - 1:
            st.markdown("<hr style='margin: 8px 0; opacity: 0.3;'>", unsafe_allow_html=True)


def create_dataset_statistics_display(statistics: Dict[str, Any]):
    """
    Display dataset statistics in an attractive format.

    Args:
        statistics: Dictionary containing dataset statistics
    """
    st.markdown(
        """
        <div class="dataset-stats">
            <div class="section-title">Dataset Statistics</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Main statistics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Documents", value=statistics.get("total_documents", 0))

    with col2:
        st.metric(label="Total Words", value=f"{statistics.get('total_words', 0):,}")

    with col3:
        st.metric(label="Avg Words/Doc", value=f"{statistics.get('average_words_per_doc', 0):,.0f}")

    with col4:
        failed_count = statistics.get("failed_documents", 0)
        st.metric(label="Failed Documents", value=failed_count, delta=f"-{failed_count}" if failed_count > 0 else None)

    # File type breakdown
    file_types = statistics.get("file_types", {})
    if file_types:
        st.markdown("**File Types:**")

        file_type_cols = st.columns(len(file_types))
        for i, (file_type, count) in enumerate(file_types.items()):
            with file_type_cols[i]:
                st.markdown(f"**{file_type.upper()}**: {count} files")


def create_processing_options():
    """
    Create UI components for processing options and settings.

    Returns:
        Dict containing selected processing options
    """
    st.markdown(
        """
        <div class="processing-options">
            <div class="section-title">Processing Options</div>
            <div class="section-sub">Configure how your documents will be processed</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Text preprocessing options
        st.markdown("**Text Preprocessing:**")
        clean_text = st.checkbox("Clean and normalize text", value=True, help="Remove extra whitespace, normalize punctuation")

        remove_headers = st.checkbox("Remove headers and footers", value=True, help="Attempt to remove document headers and footers")

    with col2:
        # Content filtering options
        st.markdown("**Content Filtering:**")
        min_word_count = st.number_input(
            "Minimum words per document", min_value=1, max_value=10000, value=10, help="Documents with fewer words will be excluded"
        )

        max_word_count = st.number_input(
            "Maximum words per document", min_value=100, max_value=100000, value=50000, help="Documents will be truncated if they exceed this length"
        )

    return {"clean_text": clean_text, "remove_headers": remove_headers, "min_word_count": min_word_count, "max_word_count": max_word_count}


def create_action_buttons() -> Dict[str, bool]:
    """
    Create action buttons for data processing operations.

    Returns:
        Dict containing button states
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        process_btn = st.button("Process Files", type="primary", help="Process uploaded files and extract content", use_container_width=True)

    with col2:
        preview_btn = st.button("Preview Data", help="Preview processed content", use_container_width=True)

    with col3:
        save_btn = st.button("Save Dataset", help="Save processed data as training dataset", use_container_width=True)

    with col4:
        clear_btn = st.button("Clear All", help="Remove all uploaded files", use_container_width=True)

    return {"process": process_btn, "preview": preview_btn, "save": save_btn, "clear": clear_btn}


def create_data_preview(dataset: List[Dict[str, Any]], max_documents: int = 3):
    """
    Create a preview of the processed dataset.

    Args:
        dataset: List of processed documents
        max_documents: Maximum number of documents to preview
    """
    if not dataset:
        st.info("No data available for preview")
        return

    st.markdown(
        """
        <div class="data-preview">
            <div class="section-title">Data Preview</div>
            <div class="section-sub">Preview of processed document content</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Show preview for up to max_documents
    preview_count = min(len(dataset), max_documents)

    for i in range(preview_count):
        document = dataset[i]

        with st.expander(f"📄 {document['filename']} ({document['word_count']} words)"):
            # Show first 500 characters of content
            content = document["content"]
            preview_content = content[:500] + "..." if len(content) > 500 else content

            st.markdown("**Content Preview:**")
            st.text_area("Content", value=preview_content, height=200, disabled=True, label_visibility="collapsed")

            # Show metadata
            if document.get("metadata"):
                st.markdown("**Metadata:**")
                metadata = document["metadata"]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**File size:** {metadata.get('file_size_mb', 0)} MB")
                    st.markdown(f"**Extension:** {metadata.get('file_extension', 'N/A')}")

                with col2:
                    if "creation_time" in metadata:
                        import datetime

                        creation_date = datetime.datetime.fromtimestamp(metadata["creation_time"])
                        st.markdown(f"**Created:** {creation_date.strftime('%Y-%m-%d %H:%M')}")

    if len(dataset) > max_documents:
        st.info(f"Showing {preview_count} of {len(dataset)} documents. Use 'Save Dataset' to access all processed data.")


def create_error_display(errors: List[str]):
    """
    Display processing errors in a user-friendly format.

    Args:
        errors: List of error messages
    """
    if not errors:
        return

    st.markdown(
        """
        <div class="error-section">
            <div class="section-title">⚠️ Processing Errors</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    for error in errors:
        st.error(error)
