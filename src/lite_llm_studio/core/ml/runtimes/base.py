"""
Module core.ml.runtimes.base
----------------------------

"""

from __future__ import annotations

from abc import ABC, abstractmethod

from lite_llm_studio.core.configuration.model_schema import ModelCard, RuntimeSpec, GenParams


class BaseRuntime(ABC):
    def __init__(self):
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, card: ModelCard, spec: RuntimeSpec) -> None: ...

    @abstractmethod
    def generate(self, history: list[dict], params: GenParams) -> str: ...

    @abstractmethod
    def unload(self) -> None: ...
