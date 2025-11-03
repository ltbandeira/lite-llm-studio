"""
Module core.runtimes.transformers_rt
------------------------------------

Runtime baseado em HuggingFace Transformers para modelos PyTorch.
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
    },
    "Qwen": {
        "system": "<|im_start|>system\n{system_prompt}<|im_end|>\n",
        "user": "<|im_start|>user\n{message}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{message}<|im_end|>\n",
        "conversation_end": "<|im_start|>assistant\n",
    },
    "Mistral": {
        "system": "<s>[INST] {system_prompt}\n\n",
        "user": "{message} [/INST]",
        "assistant": " {message} </s>[INST] ",
        "conversation_end": "[/INST]",
    },
    "Phi": {
        "system": "System: {system_prompt}\n\n",
        "user": "User: {message}\n",
        "assistant": "Assistant: {message}\n",
        "conversation_end": "Assistant:",
    },
    "Unknown": {
        "system": "System: {system_prompt}\n\n",
        "user": "User: {message}\n",
        "assistant": "Assistant: {message}\n",
        "conversation_end": "Assistant:",
    },
}


class TransformersRuntime(BaseRuntime):
    """Runtime for loading and running PyTorch models using HuggingFace Transformers."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._tokenizer = None
        self._card: ModelCard | None = None
        self._spec: RuntimeSpec | None = None
        self.logger = logging.getLogger("app.runtime.transformers")

    def load(self, card: ModelCard, spec: RuntimeSpec) -> None:
        """Load a PyTorch model using Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "❌ transformers ou torch não estão instalados.\n\n"
                "Instale com:\n"
                "  pip install torch transformers accelerate sentencepiece protobuf\n\n"
                "Para GPU NVIDIA:\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ) from e

        model_dir = card.root_dir
        self.logger.info(f"Loading model from: {model_dir}")

        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() and spec.n_gpu_layers != 0 else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.logger.info(f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer - try fast first, fallback to slow
        self.logger.info("Loading tokenizer...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:
            self.logger.warning(f"Fast tokenizer failed ({e}), trying slow tokenizer...")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            except Exception as e2:
                raise RuntimeError(
                    f"❌ Falha ao carregar tokenizer.\n\n"
                    f"Erro: {str(e2)}\n\n"
                    f"Instale as dependências necessárias:\n"
                    f"  pip install sentencepiece protobuf\n\n"
                    f"Se o problema persistir:\n"
                    f"  pip install transformers --upgrade"
                ) from e2
        
        # Set model_max_length if not set
        if not hasattr(self._tokenizer, 'model_max_length') or self._tokenizer.model_max_length > 100000:
            self._tokenizer.model_max_length = spec.n_ctx if spec.n_ctx else 2048
        
        # Load model
        self.logger.info("Loading model...")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if device == "cpu":
            self._model = self._model.to("cpu")

        self._card = card
        self._spec = spec
        self._loaded = True
        self.logger.info(f"Model loaded successfully on {device}")

    def generate(self, history: list[dict], params: GenParams) -> str:
        """Generate text using the loaded model."""
        if not self._loaded or self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError("torch not installed") from e

        # Build prompt from history using chat template
        prompt = self._build_prompt(history)
        
        # Tokenize with truncation
        inputs = self._tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self._spec.n_ctx if self._spec and self._spec.n_ctx else 2048
        )
        
        # Move to same device as model
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                do_sample=params.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def _build_prompt(self, history: list[dict]) -> str:
        """Build prompt from conversation history using model family template."""
        family = self._card.family if self._card else "Unknown"
        template = CHAT_TEMPLATES.get(family, CHAT_TEMPLATES["Unknown"])
        
        system_prompt = self._card.system_prompt if self._card else "You are a helpful assistant."
        
        prompt_parts = []
        
        # Add system prompt
        prompt_parts.append(template["system"].format(system_prompt=system_prompt))
        
        # Add conversation history
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                prompt_parts.append(template["user"].format(message=content))
            elif role == "assistant":
                prompt_parts.append(template["assistant"].format(message=content))
        
        # Add conversation end marker
        prompt_parts.append(template["conversation_end"])
        
        return "".join(prompt_parts)

    def unload(self) -> None:
        """Unload the model and free memory."""
        try:
            import torch  # type: ignore
            
            if self._model is not None:
                del self._model
                self._model = None
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._loaded = False
            self._card = None
            self._spec = None
            self.logger.info("Model unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error unloading model: {e}")
            raise

