"""
Module core.runtimes.llamacpp_rt
--------------------------------

Runtime baseado em llama-cpp-python.
"""

import logging
from typing import Any

from lite_llm_studio.core.configuration.model_schema import GenParams, ModelCard, RuntimeSpec

from .base import BaseRuntime

# Chat templates for different model families
CHAT_TEMPLATES = {
    "LLaMA": {
        "system": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n",
        "user": "{message} [/INST]",
        "assistant": " {message} </s><s>[INST] ",
        "conversation_end": "[/INST]",
        "simple_format": "System: {system_prompt}\n\nUser: {user_message}\nAssistant:",
    },
    "Qwen": {
        "system": "<|im_start|>system\n{system_prompt}<|im_end|>\n",
        "user": "<|im_start|>user\n{message}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{message}<|im_end|>\n",
        "conversation_end": "<|im_start|>assistant\n",
        "simple_format": "System: {system_prompt}\n\nUser: {user_message}\nAssistant:",
    },
    "Mistral": {
        "system": "<s>[INST] {system_prompt}\n\n",
        "user": "{message} [/INST]",
        "assistant": " {message} </s>[INST] ",
        "conversation_end": "[/INST]",
        "simple_format": "System: {system_prompt}\n\nUser: {user_message}\nAssistant:",
    },
    "Phi": {
        "system": "System: {system_prompt}\n\n",
        "user": "User: {message}\n",
        "assistant": "Assistant: {message}\n",
        "conversation_end": "Assistant:",
        "simple_format": "System: {system_prompt}\n\nUser: {user_message}\nAssistant:",
    },
    "Unknown": {
        "system": "System: {system_prompt}\n\n",
        "user": "User: {message}\n",
        "assistant": "Assistant: {message}\n",
        "conversation_end": "Assistant:",
        "simple_format": "System: {system_prompt}\n\nUser: {user_message}\nAssistant:",
    },
}


class LlamaCppRuntime(BaseRuntime):
    def __init__(self):
        super().__init__()
        self._llm = None
        self._card: ModelCard | None = None
        self._spec: RuntimeSpec | None = None
        self.logger = logging.getLogger("app.runtime.llamacpp")

    def load(self, card: ModelCard, spec: RuntimeSpec) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
            
            # Patch for llama-cpp-python 0.3.16 AttributeError bug
            try:
                from llama_cpp._internals import LlamaModel
                original_del = LlamaModel.__del__
                
                def patched_del(self):
                    try:
                        original_del(self)
                    except AttributeError:
                        # Suppress 'sampler' AttributeError in llama-cpp-python 0.3.16
                        pass
                
                LlamaModel.__del__ = patched_del
                self.logger.debug("Applied patch for llama-cpp-python 0.3.16 AttributeError")
            except Exception as e:
                self.logger.debug(f"Could not apply llama-cpp-python patch: {e}")
                
        except Exception:
            # Fallback to mock implementation for testing
            try:
                import os
                import sys

                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
                from mock_llama_cpp import Llama  # type: ignore

                print("Using mock llama-cpp implementation for testing. Install llama-cpp-python for real functionality.")
            except Exception as e:
                raise RuntimeError("llama-cpp-python não está instalado e mock não disponível. Instale llama-cpp-python.") from e

        model_path = str(card.model_file())
        kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": spec.n_ctx,
            "n_threads": spec.n_threads,
            "verbose": True,
        }
        # GPU (se desejar/possível) - permite -1 para todas as camadas
        if spec.n_gpu_layers is not None and spec.n_gpu_layers != 0:
            kwargs["n_gpu_layers"] = spec.n_gpu_layers
            self.logger.info(f"Setting n_gpu_layers = {spec.n_gpu_layers}")
        else:
            self.logger.info(f"CPU mode (n_gpu_layers = {spec.n_gpu_layers})")

        # Debug: mostra todos os parâmetros
        final_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.logger.info(f"Final Llama() parameters: {final_kwargs}")

        # Log model family for debugging
        if card.family:
            self.logger.info(f"Loading model family: {card.family}")

        # Inicializa
        self._llm = Llama(**final_kwargs)
        self._card = card
        self._spec = spec
        self._loaded = True

    def _format_prompt(self, history: list[dict]) -> str:
        """
        Format prompt with proper chat template based on model family.
        Supports multiple model families: LLaMA, Qwen, Mistral, Phi, Gemma, Yi, DeepSeek, Solar.
        """
        if not self._card:
            return self._format_simple_prompt(history)

        # Get the appropriate template for this model family
        family = self._card.family or "Unknown"
        template = CHAT_TEMPLATES.get(family, CHAT_TEMPLATES["Unknown"])

        # Use simple format for single-turn conversations or fallback
        if len(history) <= 1:
            return self._format_simple_conversation(history, template)

        # Use full template for multi-turn conversations
        return self._format_full_conversation(history, template)

    def _format_simple_conversation(self, history: list[dict], template: dict) -> str:
        """Format simple single-turn conversation using simple_format template."""
        system_prompt = ""
        user_message = ""

        # Extract system prompt and last user message
        if self._card and self._card.system_prompt:
            system_prompt = self._card.system_prompt

        for msg in history:
            if msg.get("role") == "user":
                user_message = msg.get("content", "").strip()

        if not user_message:
            user_message = "Hello"

        return template["simple_format"].format(system_prompt=system_prompt, user_message=user_message)

    def _format_full_conversation(self, history: list[dict], template: dict) -> str:
        """Format full multi-turn conversation using model-specific templates."""
        parts = []

        # Add system message if available
        if self._card and self._card.system_prompt:
            parts.append(template["system"].format(system_prompt=self._card.system_prompt))

        # Process conversation history
        for i, msg in enumerate(history):
            role = msg.get("role", "").lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                if i == len(history) - 1:  # Last message (current user input)
                    parts.append(template["user"].format(message=content))
                else:
                    parts.append(template["user"].format(message=content))
            elif role == "assistant":
                parts.append(template["assistant"].format(message=content))

        # Add conversation end marker for response generation
        parts.append(template["conversation_end"])

        return "".join(parts)

    def _format_simple_prompt(self, history: list[dict]) -> str:
        """Fallback simple formatting when no model card is available."""
        lines = []

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

        lines.append("Assistant:")
        return "\n\n".join(lines)

    def generate(self, history: list[dict], params: GenParams) -> str:
        if not self._loaded or self._llm is None:
            raise RuntimeError("Runtime não carregado.")

        # Format prompt using model-specific template
        prompt = self._format_prompt(history)

        # Log template info for debugging
        if self._card:
            self.logger.debug(f"Using chat template for {self._card.family} family")

        # Generate response
        out = self._llm(
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=params.stop or None,
        )

        # Extract text from llama-cpp response
        text = (out.get("choices") or [{}])[0].get("text", "")
        return text.strip()

    def unload(self) -> None:
        """Unload model with proper cleanup to avoid AttributeError in __del__"""
        if self._llm is not None:
            try:
                # Try to close properly if method exists
                if hasattr(self._llm, 'close'):
                    self._llm.close()
                elif hasattr(self._llm, '_model') and hasattr(self._llm._model, 'close'):
                    self._llm._model.close()
            except Exception as e:
                self.logger.debug(f"Error during model cleanup (can be ignored): {e}")
            finally:
                self._llm = None
        
        self._loaded = False
        self._card = None
        self._spec = None
    
    def __del__(self):
        """Destructor with error suppression for llama-cpp-python 0.3.16 bug"""
        try:
            self.unload()
        except Exception:
            pass  # Suppress any errors during cleanup
