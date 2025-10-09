"""
Module core.configuration.desktop_app_config
--------------------------------------------

This module manages application directories for the desktop runtime and ensures
they are created at startup. It also exposes convenience helpers to query and
create the default models directory and related folders.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DesktopAppConfig:
    """
    Desktop application configuration manager.

    This class handles the setup and management of application directories,
    ensuring they are created at startup.
    """

    FIXED_BASE_DIR = Path("C:/LiteLLM-Studio")

    def __init__(self):
        """Initialize the desktop app configuration and its logger."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_user_data_directory(self) -> Path:
        """
        Resolve the base user data directory for the application.

        Returns:
            Path: The resolved user data directory (Windows).
        """
        return self.FIXED_BASE_DIR

    def get_default_models_directory(self) -> Path:
        """
        Get the default directory where local models are stored.

        Returns:
            Path: `<user_data_directory>/models`
        """
        return self.get_user_data_directory() / "models"

    def ensure_directory_exists(self, directory: Path, description: str = "Directory") -> bool:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory (Path): Target directory path to create/ensure.
            description (str): Human-readable label used in log messages.

        Returns:
            bool: True on success, False on failure.
        """
        try:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"{description} created at: {directory}")
            else:
                self.logger.info(f"{description} already exists at: {directory}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create {description.lower()} at {directory}: {e}")
            return False

    def setup_application_directories(self) -> dict[str, Path]:
        """
        Create and return the key application directories.

        Returns:
            Dict[str, Path]: Mapping with keys `user_data`, `models` and `logs`
                for the directories that were successfully ensured/created.
        """
        directories: dict[str, Path] = {}

        # User data directory
        user_data_dir = self.get_user_data_directory()
        if self.ensure_directory_exists(user_data_dir, "User data directory"):
            directories["user_data"] = user_data_dir

        # Models directory
        models_dir = self.get_default_models_directory()
        if self.ensure_directory_exists(models_dir, "Models directory"):
            directories["models"] = models_dir

        # Logs directory (in user data)
        logs_dir = user_data_dir / "logs"
        if self.ensure_directory_exists(logs_dir, "Logs directory"):
            directories["logs"] = logs_dir

        return directories

    def get_models_directory_info(self) -> dict[str, Any]:
        """
        Inspect the models directory and return basic metadata.

        Returns:
            Dict[str, Any]: A dictionary with directory metadata as described above.
        """
        models_dir = self.get_default_models_directory()

        info: dict[str, Any] = {
            "path": str(models_dir),
            "exists": models_dir.exists(),
            "is_writable": False,
            "model_count": 0,
            "total_size": 0,
        }

        if models_dir.exists():
            try:
                # Check writability by trying to create and remove a temp file
                test_file = models_dir / ".write_test"
                test_file.touch()
                test_file.unlink()
                info["is_writable"] = True
            except Exception:
                info["is_writable"] = False

            # Count potential model files
            try:
                model_files = list(models_dir.rglob("*.gguf"))
                info["model_count"] = len(model_files)

                # Calculate total size
                total_size = 0
                for file in model_files:
                    try:
                        total_size += file.stat().st_size
                    except Exception:
                        pass
                info["total_size"] = total_size

            except Exception as e:
                self.logger.warning(f"Could not scan models directory: {e}")

        return info


# Create a singleton instance
_desktop_config = DesktopAppConfig()


# Expose class methods as module-level functions
def get_user_data_directory() -> Path:
    """Return the application user data directory."""
    return _desktop_config.get_user_data_directory()


def get_default_models_directory() -> Path:
    """Return the default path for local models."""
    return _desktop_config.get_default_models_directory()


def ensure_directory_exists(directory: Path, description: str = "Directory") -> bool:
    """Ensure a directory exists by creating it if necessary."""
    return _desktop_config.ensure_directory_exists(directory, description)


def setup_application_directories() -> dict[str, Path]:
    """Create and return the core application directories."""
    return _desktop_config.setup_application_directories()


def get_models_directory_info() -> dict[str, Any]:
    """Return basic metadata for the models directory."""
    return _desktop_config.get_models_directory_info()
