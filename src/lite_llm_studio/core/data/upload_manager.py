"""
Module core.data.upload_manager
-------------------------------

This module handles file upload and temporary file management.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import uuid


@dataclass
class UploadedFile:
    """Represents an uploaded file with metadata."""

    id: str
    original_filename: str
    temp_path: Path
    size_bytes: int
    mime_type: Optional[str] = None
    upload_timestamp: Optional[float] = None


class UploadManager:
    """
    Manages file uploads and temporary file storage.
    """

    def __init__(self, temp_dir: Optional[Path] = None, logger_name: str = "app.data.upload_manager"):
        """
        Initialize the upload manager.

        Args:
            temp_dir: Directory for temporary files. If None, uses system temp.
            logger_name: Name for the logger instance.
        """
        self.logger = logging.getLogger(logger_name)

        # Setup temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="lite_llm_studio_"))

        self.uploaded_files: Dict[str, UploadedFile] = {}
        self.logger.info(f"UploadManager initialized with temp directory: {self.temp_dir}")

    def save_uploaded_file(self, file_content: bytes, filename: str, mime_type: Optional[str] = None) -> UploadedFile:
        """
        Save an uploaded file to temporary storage.

        Args:
            file_content: The file content as bytes
            filename: Original filename
            mime_type: MIME type of the file

        Returns:
            UploadedFile: Object representing the saved file
        """
        # Generate unique ID and safe filename
        file_id = str(uuid.uuid4())
        safe_filename = self._sanitize_filename(filename)
        temp_path = self.temp_dir / f"{file_id}_{safe_filename}"

        try:
            # Write file to temporary location
            with open(temp_path, "wb") as f:
                f.write(file_content)

            # Create UploadedFile object
            uploaded_file = UploadedFile(
                id=file_id,
                original_filename=filename,
                temp_path=temp_path,
                size_bytes=len(file_content),
                mime_type=mime_type,
                upload_timestamp=None,  # Will be set by Streamlit if needed
            )

            # Store in registry
            self.uploaded_files[file_id] = uploaded_file

            self.logger.info(f"File uploaded successfully: {filename} ({len(file_content)} bytes)")
            return uploaded_file

        except Exception as e:
            self.logger.error(f"Failed to save uploaded file {filename}: {str(e)}")
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    def save_streamlit_uploaded_file(self, uploaded_file_obj) -> UploadedFile:
        """
        Save a Streamlit UploadedFile object to temporary storage.

        Args:
            uploaded_file_obj: Streamlit UploadedFile object

        Returns:
            UploadedFile: Object representing the saved file
        """
        # Read content from Streamlit object
        file_content = uploaded_file_obj.read()

        # Reset file pointer if needed
        uploaded_file_obj.seek(0)

        return self.save_uploaded_file(file_content=file_content, filename=uploaded_file_obj.name, mime_type=getattr(uploaded_file_obj, "type", None))

    def get_uploaded_file(self, file_id: str) -> Optional[UploadedFile]:
        """
        Retrieve an uploaded file by ID.

        Args:
            file_id: Unique identifier for the file

        Returns:
            UploadedFile object or None if not found
        """
        return self.uploaded_files.get(file_id)

    def list_uploaded_files(self) -> List[UploadedFile]:
        """
        Get a list of all uploaded files.

        Returns:
            List of UploadedFile objects
        """
        return list(self.uploaded_files.values())

    def remove_uploaded_file(self, file_id: str) -> bool:
        """
        Remove an uploaded file from temporary storage.

        Args:
            file_id: Unique identifier for the file

        Returns:
            bool: True if file was removed, False if not found
        """
        uploaded_file = self.uploaded_files.get(file_id)
        if not uploaded_file:
            return False

        try:
            # Remove physical file
            if uploaded_file.temp_path.exists():
                uploaded_file.temp_path.unlink()

            # Remove from registry
            del self.uploaded_files[file_id]

            self.logger.info(f"Removed uploaded file: {uploaded_file.original_filename}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to remove uploaded file {file_id}: {str(e)}")
            return False

    def clear_all_files(self):
        """Remove all uploaded files from temporary storage."""
        file_ids = list(self.uploaded_files.keys())

        for file_id in file_ids:
            self.remove_uploaded_file(file_id)

        self.logger.info("Cleared all uploaded files")

    def get_upload_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about uploaded files.

        Returns:
            Dict containing upload statistics
        """
        files = self.list_uploaded_files()

        if not files:
            return {"total_files": 0, "total_size_bytes": 0, "total_size_mb": 0, "file_types": {}}

        total_size = sum(f.size_bytes for f in files)
        file_types = {}

        for file in files:
            # Extract file extension
            ext = Path(file.original_filename).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types,
        }

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to be safe for filesystem.

        Args:
            filename: Original filename

        Returns:
            str: Sanitized filename
        """
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename

        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, "_")

        # Limit length
        if len(safe_filename) > 200:
            name, ext = Path(safe_filename).stem, Path(safe_filename).suffix
            safe_filename = name[:190] + ext

        return safe_filename

    def cleanup(self):
        """Clean up all temporary files and directories."""
        try:
            self.clear_all_files()

            # Remove temp directory if it's empty
            if self.temp_dir.exists() and not list(self.temp_dir.iterdir()):
                self.temp_dir.rmdir()
                self.logger.info(f"Removed empty temp directory: {self.temp_dir}")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup on destruction
