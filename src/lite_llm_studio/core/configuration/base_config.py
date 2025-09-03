"""
Module lite_llm_studio.configuration.base_config
------------------------------------------------

This module defines the abstract `BaseConfigModel`, the base interface for all
project configuration objects. It provides a standardized structure for data
validation using Pydantic and specifies the minimal contract that every
configuration class must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseConfigModel(ABC, BaseModel):
    """
    Abstract base class for all configuration settings in the
    LiteLLM Studio project.

    This class inherits from ABC (Abstract Base Class) and BaseModel from Pydantic,
    providing a standardized interface for configuration data validation.

    Attributes:
        model_config (ConfigDict): Pydantic model configuration with rules.
    """

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra attributes not defined in the model
        populate_by_name=True,  # Allow using attribute names as keys
        str_strip_whitespace=True,  # Strip whitespace from string fields
    )

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Converts the configuration model into a serializable dictionary.

        This method must be implemented by all child classes to
        provide a dictionary representation of the configuration.

        Returns:
            dict: Dictionary containing all configuration data.
        """
        return self.model_dump(mode="python", exclude_none=True)

    @abstractmethod
    def to_json(self) -> str:
        """
        Converts the configuration model into a JSON string.

        This method can be overridden by child classes if custom
        serialization is needed.

        Returns:
            str: JSON string representation of the configuration data.
        """
        return self.model_dump_json(exclude_none=True)
