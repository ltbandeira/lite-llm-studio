"""
Module core.runtimes.llamacpp_rt
--------------------------------

Runtime baseado em llama-cpp-python.
"""

from typing import Any

from lite_llm_studio.core.configuration.model_schema import GenParams, ModelCard, RuntimeSpec
from .base import BaseRuntime


class LlamaCppRuntime(BaseRuntime):
    def __init__(self):
        super().__init__()
        self._llm = None
        self._card: ModelCard | None = None
        self._spec: RuntimeSpec | None = None

    def load(self, card: ModelCard, spec: RuntimeSpec) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError("llama-cpp-python não está instalado. Adicione `llama-cpp-python` ao seu ambiente.") from e

        model_path = str(card.model_file())
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": spec.n_ctx,
            "n_threads": spec.n_threads,
        }
        # GPU (se desejar/possível)
        if spec.n_gpu_layers and spec.n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = spec.n_gpu_layers

        # Inicializa
        self._llm = Llama(**{k: v for k, v in kwargs.items() if v is not None})
        self._card = card
        self._spec = spec
        self._loaded = True

    def _format_prompt(self, history: list[dict]) -> str:
        """
        Format prompt with proper chat template for Llama models.
        Uses a more standard format: System message + conversation history.
        """
        lines: list[str] = []

        # Add system message if available
        if self._card and self._card.system_prompt:
            lines.append(f"System: {self._card.system_prompt}")

        # Add conversation history
        for msg in history:
            role = msg.get("role", "").lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "system":
                lines.append(f"System: {content}")

        # Add assistant prefix for response generation
        lines.append("Assistant:")

        return "\n\n".join(lines)

    def generate(self, history: list[dict], params: GenParams) -> str:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Runtime não carregado.")
        prompt = self._format_prompt(history)
        out = self._llm(
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop or None,
        )
        # llama-cpp retorna dict; pega texto
        text = (out.get("choices") or [{}])[0].get("text", "")
        return text

    def unload(self) -> None:
        self._llm = None
        self._loaded = False
        self._card = None
        self._spec = None
