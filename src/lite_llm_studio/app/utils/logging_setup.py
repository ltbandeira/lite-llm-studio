"""
Module app.utils.logging_config
-------------------------------

This module provides logging configuration for the Streamlit application.
It uses the unified logger_config.yaml file for consistency across the
entire application.
"""

import logging
import logging.config
from pathlib import Path

import yaml

from lite_llm_studio.core.configuration import get_user_data_directory


def setup_app_logging() -> None:
    """
    Setup logging for the Streamlit application using logger_config.yaml.
    """
    # Ensure logs directory exists
    logs_dir = get_user_data_directory() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Find the logger config file
    config_paths = [
        Path(__file__).parent.parent.parent.parent / "config" / "logger_config.yaml",  # Development
        Path.cwd() / "config" / "logger_config.yaml",  # Running from project root
    ]

    config_file = None
    for path in config_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        raise FileNotFoundError("Could not find logger_config.yaml in expected locations.")

    # Load YAML configuration
    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Update all file handlers to use absolute paths
    for _, handler_config in config.get("handlers", {}).items():
        if "filename" in handler_config:
            # Convert relative path to absolute path in logs directory
            relative_path = Path(handler_config["filename"])
            absolute_path = logs_dir / relative_path.name
            handler_config["filename"] = str(absolute_path)

    # Apply the configuration
    logging.config.dictConfig(config)

    # Log initialization
    logger = logging.getLogger("app.streamlit")
    logger.info("=" * 40)
    logger.info("Streamlit application logging initialized from logger_config.yaml")
    logger.info(f"Config file: {config_file}")
    logger.info(f"Log directory: {logs_dir}")
    logger.info("=" * 40)


def get_app_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        name: Logger name (e.g., 'app.streamlit', 'app.pages.home')

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)
