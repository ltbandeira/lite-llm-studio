"""
Module app.modules.training_page
--------------------------------

This module contains the model training page content.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import streamlit as st

from lite_llm_studio.core.configuration.data_schema import ChunkingStrategy, DataProcessingConfig
from lite_llm_studio.core.configuration.desktop_app_config import get_default_models_directory, get_user_data_directory

from ..icons import ICONS

# Get logger for training page
logger = logging.getLogger("app.pages.training")

# ============================================================================
# MODEL RECOMMENDATION CONSTANTS AND HELPERS
# ============================================================================

def get_models_dir() -> str:
    """Get the models directory path."""
    return str(get_default_models_directory())

DEFAULT_BYTES_PER_PARAM: float = 2.0
DEFAULT_PRECISION: str = "fp16"

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 1024

# Supported models catalog
MODELS: list[dict[str, Any]] = [
    {"name": "Qwen2-0.5B", "params_b": 0.5, "repo_id": "Qwen/Qwen2-0.5B", "context_length": 32768},
    # {"name": "TinyLlama-1.1B", "params_b": 1.1, "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "context_length": 2048},
    # {"name": "Llama-3.1-70B", "params_b": 70.0, "repo_id": "meta-llama/Llama-3.1-70B", "context_length": 131072},
    # {"name": "Qwen3-4B-Instruct", "params_b": 4.0, "repo_id": "Qwen/Qwen3-4B-Instruct-2507", "context_length": 262144},
    # {"name": "Mistral-7B-Instruct-v0.3", "params_b": 7.0, "repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "context_length": 32768},
]


def _estimate_model_memory(params_b: float, bytes_per_param: float = DEFAULT_BYTES_PER_PARAM) -> float:
    """Estimate model memory usage with 20% margin."""
    return round(params_b * bytes_per_param * 1.2, 2)


def _summarize_hardware(hardware_data: dict[str, Any]) -> dict[str, Any]:
    """Summarize hardware data for model recommendations."""
    memory_info = hardware_data.get("memory", {}) or {}
    gpus = hardware_data.get("gpus", []) or []
    return {
        "ram_total": memory_info.get("total_memory", 0),
        "ram_free": memory_info.get("free_memory", 0),
        "gpus": gpus,
        "has_gpu": bool(gpus),
        "max_vram": max([g.get("total_vram", 0) for g in gpus], default=0),
        "gpu_names": ", ".join([g.get("name", "GPU") for g in gpus]) if gpus else None,
    }


def _enumerate_model_configs(hw: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate models with their configurations based on hardware."""
    results: list[dict[str, Any]] = []

    for model in MODELS:
        mem_gb = _estimate_model_memory(model["params_b"], DEFAULT_BYTES_PER_PARAM)

        if hw["has_gpu"] and hw["max_vram"] >= mem_gb:
            device, mem_type = "GPU", "VRAM"
        elif hw["ram_total"] >= mem_gb:
            device, mem_type = "CPU", "RAM"
        else:
            device, mem_type = "-", "-"

        results.append(
            {
                **model,
                "precision": DEFAULT_PRECISION,
                "bytes_per_param": DEFAULT_BYTES_PER_PARAM,
                "device": device,
                "mem_type": mem_type,
                "mem_gb": mem_gb,
                "enough": device != "-",
            }
        )
    return results


def _render_config_list(configs: list[dict[str, Any]], show_device: bool = True):
    """Render a list of model configurations."""
    if not configs:
        return

    rows_html = ""
    last_index = len(configs) - 1
    for idx, cfg in enumerate(configs):
        name = cfg.get("name", "Modelo")
        repo = cfg.get("repo_id", "")
        precision = cfg.get("precision", "-")
        device = cfg.get("device", "-")
        mem_type = cfg.get("mem_type", "-")
        mem_gb = cfg.get("mem_gb")

        status_cls = "ok" if cfg.get("enough") else "bad"
        chips = [
            f'<span class="chip {status_cls}">{precision}</span>',
        ]
        if show_device and device and device != "-":
            chips.append(f'<span class="chip">{device}</span>')
        chips.append(f'<span class="chip">{mem_gb:.2f} GB {mem_type}</span>')

        rows_html += (
            '<div class="storage-row" style="padding:8px 0;">'
            f'  <div class="storage-name" style="width:303px;">{name}'
            f'    <div class="storage-sub" style="font-size:0.85rem; opacity:.85; white-space:normal; overflow:visible; text-overflow:unset;">{repo}</div>'
            f"  </div>"
            '  <div class="storage-badges">' + "".join(chips) + "</div>"
            "</div>"
        )
        if idx != last_index:
            rows_html += '<div style="height:1px; background: rgba(255,255,255,0.12); margin: 4px 0;"></div>'

    block_html = '<div class="storage-card"><div class="storage-list">' + rows_html + "</div></div>"
    st.markdown(block_html, unsafe_allow_html=True)


# ============================================================================
# DRY RUN HELPERS
# ============================================================================


def _list_local_models(base_dir: str | None = None) -> list[str]:
    """List locally available models."""
    try:
        if base_dir is None:
            base_dir = get_models_dir()
        if not os.path.isdir(base_dir):
            return []
        return [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception:
        return []


def _get_device_info() -> dict[str, Any]:
    """Get device information for PyTorch."""
    info: dict[str, Any] = {"has_torch": False, "has_cuda": False, "device": "cpu", "cuda_name": None}
    try:
        import torch  # type: ignore

        info["has_torch"] = True
        info["has_cuda"] = bool(torch.cuda.is_available())
        if info["has_cuda"]:
            info["device"] = "cuda"
            try:
                info["cuda_name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["cuda_name"] = "CUDA GPU"
    except Exception:
        pass
    return info


def _bytes_to_mb_gb(num_bytes: float) -> str:
    """Convert bytes to MB or GB string."""
    gb = num_bytes / (1024**3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = num_bytes / (1024**2)
    return f"{mb:.0f} MB"


def _load_text_from_dataset(dataset_path: str, max_samples: int = 100) -> tuple[str, int, str]:
    """Load text from a processed JSONL dataset."""
    try:
        import json

        parts: list[str] = []
        total_bytes = 0
        sample_count = 0

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if sample_count >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    # Extract text from different possible fields
                    text = data.get("text") or data.get("content") or data.get("chunk") or str(data)
                    parts.append(text)
                    total_bytes += len(text.encode("utf-8", errors="ignore"))
                    sample_count += 1
                except Exception:
                    continue

        combined_text = "\n\n".join(parts)
        return (combined_text, total_bytes, f"Dataset processado ({sample_count} amostras)")
    except Exception as ex:
        return ("", 0, f"Erro ao carregar dataset: {ex}")


def _load_builtin_text(dataset_key: str, target_bytes: int = 1_000_000) -> tuple[str, int, str]:
    """Load known text dataset."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return ("", 0, "Datasets library não instalada. Instale: pip install datasets")

    ds_name, subset, split = ("wikitext", "wikitext-2-raw-v1", "train")
    if dataset_key == "wikitext-103":
        subset = "wikitext-103-raw-v1"
    elif dataset_key == "ag_news":
        ds_name, subset, split = ("ag_news", None, "train")

    try:
        if subset:
            ds = load_dataset(ds_name, subset, split=split)
            desc = f"{ds_name}/{subset} ({split})"
        else:
            ds = load_dataset(ds_name, split=split)
            desc = f"{ds_name} ({split})"
        parts: list[str] = []
        total = 0
        text_field = "text" if "text" in ds.features else ("content" if "content" in ds.features else None)
        title_field = "title" if "title" in ds.features else None
        for ex in ds:
            segs: list[str] = []
            if title_field and ex.get(title_field):
                segs.append(str(ex[title_field]))
            if text_field and ex.get(text_field):
                segs.append(str(ex[text_field]))
            elif ex and not text_field:
                segs.append(" ".join(str(v) for v in ex.values()))
            chunk = "\n\n".join(segs)
            b = len(chunk.encode("utf-8", errors="ignore"))
            parts.append(chunk)
            total += b
            if total >= target_bytes:
                break
        text = "\n\n".join(parts)
        return (text, len(text.encode("utf-8", errors="ignore")), desc)
    except Exception as ex:
        return ("", 0, f"Falha ao carregar dataset: {ex}")


def _load_model_tokenizer(model_dir: str, device_info: dict[str, Any], force_gpu: bool = False):
    """Load model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        use_cuda = bool(device_info.get("has_cuda"))
        dtype = torch.float16 if use_cuda else torch.float32
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        if use_cuda and force_gpu:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=None)
            try:
                model.to("cuda")
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map="auto" if use_cuda else None,
            )
        return model, tok
    except Exception as ex:
        raise RuntimeError(str(ex))


def _measure_ttft(model, tokenizer, device_info: dict[str, Any]) -> float:
    """Measure time to first token."""
    try:
        import torch  # type: ignore

        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt")
        if device_info.get("has_cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.empty_cache() if device_info.get("has_cuda") else None
        start = time.perf_counter()
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        end = time.perf_counter()
        return max(0.0, end - start)
    except Exception:
        return float("nan")


def _resolve_max_len(model, tokenizer) -> int:
    """Resolve maximum context length for model."""
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < 100000:
        return max_len
    cfg_len = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    if isinstance(cfg_len, int) and cfg_len > 0:
        return cfg_len
    return 2048


def _tokenize_with_truncation(tokenizer, text: str, device_info: dict[str, Any], max_len: int):
    """Tokenize text with truncation."""
    batch_tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    if device_info.get("has_cuda"):
        batch_tokens = {k: v.to("cuda") for k, v in batch_tokens.items()}
    seq_len = int(batch_tokens["input_ids"].shape[-1])
    return batch_tokens, seq_len


def _measure_forward(model, batch_tokens, device_info: dict[str, Any]) -> float:
    """Measure forward pass time."""
    try:
        import torch  # type: ignore

        logger.info("Starting forward pass measurement")
        
        with torch.no_grad():
            # Clear cache before measurement
            if device_info.get("has_cuda"):
                try:
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache")
                except Exception:
                    pass
            
            start = time.perf_counter()
            logger.info("Executing model forward pass...")
            outputs = model(**batch_tokens)
            logger.info("Forward pass completed")
            end = time.perf_counter()
            
            elapsed = max(0.0, end - start)
            logger.info(f"Forward pass took {elapsed:.2f} seconds")
            return elapsed
    except Exception as e:
        logger.error(f"Error during forward pass: {e}", exc_info=True)
        return float("nan")


def _measure_memory_usage(device_info: dict[str, Any]) -> dict[str, Any]:
    """Measure current memory usage."""
    mem: dict[str, Any] = {}
    proc = psutil.Process()
    mem["rss"] = proc.memory_info().rss
    if device_info.get("has_torch") and device_info.get("has_cuda"):
        try:
            import torch  # type: ignore

            torch.cuda.synchronize()
            mem["cuda_alloc"] = torch.cuda.memory_allocated(0)
            mem["cuda_reserved"] = torch.cuda.memory_reserved(0)
        except Exception:
            mem["cuda_alloc"] = None
            mem["cuda_reserved"] = None
    else:
        mem["cuda_alloc"] = None
        mem["cuda_reserved"] = None
    return mem


def _render_kv_row(label: str, value: str) -> str:
    """Render a key-value row."""
    return '<div class="spec-row">' f'  <div class="spec-label">{label}</div>' f'  <div class="spec-value">{value}</div>' "</div>"


def _infer_actual_device(model, batch_tokens, device_info: dict[str, Any]) -> str:
    """Infer actual device being used for inference."""
    # 1) If any input tensor is on CUDA, consider the run as GPU
    try:
        for t in batch_tokens.values():
            try:
                if hasattr(t, "is_cuda") and bool(t.is_cuda):
                    return "cuda"
                if hasattr(t, "device") and getattr(t.device, "type", "") == "cuda":
                    return "cuda"
            except Exception:
                continue
    except Exception:
        pass
    # 2) Check hf_device_map if present
    try:
        device_map = getattr(model, "hf_device_map", None)
        if isinstance(device_map, dict):
            if any("cuda" in str(dev).lower() for dev in device_map.values()):
                return "cuda"
    except Exception:
        pass
    # 3) Fallback: scan parameters/buffers
    try:
        for p in model.parameters():
            if getattr(getattr(p, "device", None), "type", "") == "cuda" or getattr(p, "is_cuda", False):
                return "cuda"
        for b in model.buffers():
            if getattr(getattr(b, "device", None), "type", "") == "cuda" or getattr(b, "is_cuda", False):
                return "cuda"
    except Exception:
        pass
    # 4) Last resort: torch.cuda.is_available
    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and device_info.get("has_cuda"):
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class PipelineStepConfig:
    """Configuration for a pipeline step."""

    key: str
    label: str
    icon: str
    description: str


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self, config: PipelineStepConfig):
        self.config = config
        self.logger = logging.getLogger(f"app.pipeline.{config.key}")

    @abstractmethod
    def render_ui(self) -> None:
        """Render the UI for this pipeline step."""
        pass

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the backend logic for this step."""
        pass

    @abstractmethod
    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this step can be executed with the given context."""
        pass

    def can_complete(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if this step can be marked as complete."""
        return self.validate(context)


class ModelRecommendationStep(PipelineStep):
    """Model recommendation step with hardware-aware model selection."""

    def render_ui(self) -> None:
        """Render UI for model recommendation step."""
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 1 of 5</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Get hardware data from session state or run scan
        hardware_data = st.session_state.get("tp_context", {}).get("hardware_data", {})
        if not hardware_data:
            with st.spinner("Analyzing system configuration..."):
                from lite_llm_studio.core.orchestration import Orchestrator

                orchestrator = Orchestrator()
                hardware_data = orchestrator.execute_hardware_scan() or {}
                # Save to session state
                if "tp_context" not in st.session_state:
                    st.session_state.tp_context = {}
                st.session_state.tp_context["hardware_data"] = hardware_data

        # Render the recommendations UI inline
        self._render_recommendations_ui(hardware_data)

        # Handle button actions
        self._handle_model_selection()

    def _render_recommendations_ui(self, hardware_data: dict[str, Any]) -> None:
        """Render the recommendations UI inline."""
        if not hardware_data:
            st.error("Error retrieving system information.")
            return

        hw = _summarize_hardware(hardware_data)

        # Build model configurations
        configs = _enumerate_model_configs(hw)
        df = pd.DataFrame(configs)
        if df.empty:
            st.warning("Sem dados de configuração.")
            return

        runnable = df[df["enough"]].copy().sort_values(by=["params_b"], ascending=False)
        not_runnable = df[~df["enough"]].copy().sort_values(by=["params_b"], ascending=False)

        # Recommended card first
        if not runnable.empty:
            best = runnable.iloc[0].to_dict()

            tags_html = (
                f'<span class="chip ok">{best.get("precision")}</span>'
                f'<span class="chip">{best.get("device")}</span>'
                f'<span class="chip">{best.get("mem_gb"):.2f} GB {best.get("mem_type")}</span>'
            )

            card_html = f"""
            <div class="syscpu-card">
              <div class="syscpu-header">
                <div class="syscpu-left">
                  <div>
                    <div class="syscpu-title">Recomendado</div>
                    <div class="syscpu-sub">{best.get("name")} • {best.get("repo_id")}</div>
                  </div>
                </div>
                <div class="syscpu-tags">
                  {tags_html}
                </div>
              </div>
              <div class="spec-grid">
                <div class="spec-row">
                  <div class="spec-label">Parâmetros</div>
                  <div class="spec-value">{best.get("params_b")} B</div>
                </div>
                <div class="spec-row">
                  <div class="spec-label">Memória estimada ({DEFAULT_PRECISION})</div>
                  <div class="spec-value">{best.get("mem_gb"):.2f} GB {best.get("mem_type")}</div>
                </div>
                <div class="spec-row">
                  <div class="spec-label">Tam. Contexto</div>
                  <div class="spec-value">{best.get("context_length", "N/A")}</div>
                </div>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Prepare model data
            repo_id = best.get("repo_id") or ""
            folder_name = repo_id.replace("/", "-")
            is_gpu_recommended = best.get("device") == "GPU"
            recommended_gpu_layers = 999 if is_gpu_recommended else 0
            recommended_context_length = best.get("context_length", 2048)

            generation_params = {
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "top_p": DEFAULT_TOP_P,
                "context_length": recommended_context_length,
                "use_gpu": is_gpu_recommended,
                "gpu_layers": recommended_gpu_layers,
            }

            model_data_for_pipeline = {
                "name": best.get("name"),
                "repo_id": repo_id,
                "folder_name": folder_name,
                "params_b": best.get("params_b"),
                "precision": best.get("precision"),
                "device": best.get("device"),
                "mem_gb": best.get("mem_gb"),
                "generation_params": generation_params,
            }

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Baixar modelo recomendado", type="primary", use_container_width=True, key="mr_download_btn"):
                    st.session_state.mr_download_clicked = True
                    st.session_state.mr_model_data = model_data_for_pipeline
                    st.rerun()

            with col2:
                if st.button("Pular etapa", use_container_width=True, key="mr_skip_btn"):
                    st.session_state.mr_skip_clicked = True
                    st.session_state.mr_model_data = model_data_for_pipeline
                    st.rerun()

        else:
            st.warning("Nenhuma configuração atende aos requisitos de memória.")

            if st.button("Pular etapa", key="mr_skip_no_model_btn"):
                st.session_state.mr_skip_clicked = True
                st.session_state.mr_model_data = None
                st.rerun()

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        # Runnable list (dropdown)
        with st.expander("Executáveis", expanded=False):
            if not runnable.empty:
                _render_config_list(runnable.to_dict(orient="records"))
            else:
                st.info("Nenhuma configuração executável encontrada.")

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        # Not runnable list (dropdown)
        with st.expander("Não executáveis", expanded=False):
            if not not_runnable.empty:
                _render_config_list(not_runnable.to_dict(orient="records"), show_device=False)
            else:
                st.info("Todas as configurações listadas são executáveis.")

    def _handle_model_selection(self) -> None:
        """Handle model download or skip actions."""
        # Check if download button was clicked (triggered in recommendations_page)
        if st.session_state.get("mr_download_clicked"):
            model_data = st.session_state.get("mr_model_data")
            if model_data:
                repo_id = model_data.get("repo_id", "")
                folder_name = model_data.get("folder_name", "")
                
                # Use centralized models directory
                models_dir = get_models_dir()
                os.makedirs(models_dir, exist_ok=True)
                download_path = os.path.join(models_dir, folder_name)
                
                with st.spinner("Baixando modelo do Hugging Face…"):
                    try:
                        from huggingface_hub import snapshot_download
                        
                        snapshot_download(repo_id, local_dir=download_path, local_dir_use_symlinks=False)
                        st.success(f"✅ Download concluído em {download_path}")
                        
                        # Save to context
                        st.session_state.tp_context["selected_model"] = model_data
                        st.session_state.tp_context["recommended_model"] = model_data
                        
                        # Clear flags
                        st.session_state.mr_download_clicked = False
                        st.session_state.mr_model_data = None
                        
                    except ModuleNotFoundError:
                        st.error("Instale o pacote necessário: pip install -U huggingface_hub")
                    except Exception as ex:
                        st.error(f"Falha no download: {ex}")
        
        # Check if skip button was clicked
        if st.session_state.get("mr_skip_clicked"):
            model_data = st.session_state.get("mr_model_data")
            if model_data:
                # Save to context without downloading
                st.session_state.tp_context["selected_model"] = model_data
                st.session_state.tp_context["recommended_model"] = model_data
                st.info("Etapa de recomendação pulada. Prosseguindo com modelo recomendado (não baixado).")
            else:
                # No model recommended - skip with null
                st.session_state.tp_context["selected_model"] = None
                st.session_state.tp_context["recommended_model"] = None
                st.info("Etapa de recomendação pulada. Nenhum modelo selecionado.")
            
            # Clear flag
            st.session_state.mr_skip_clicked = False
            st.session_state.mr_model_data = None

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute model recommendation logic."""
        self.logger.info("Executing model recommendation step")
        
        # Get selected model from context (set by UI interactions)
        selected_model = context.get("selected_model")
        if selected_model:
            self.logger.info(f"Model selected: {selected_model}")
            return {"recommended_model": selected_model}
        
        # If no model selected, log that step was skipped
        self.logger.info("Model recommendation step completed (skipped or no model)")
        return {"recommended_model": None}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if a model has been selected or step was explicitly skipped."""
        # Allow validation to pass if either a model is selected OR if the step was explicitly completed
        selected_model = context.get("selected_model")
        recommended_model = context.get("recommended_model")
        
        # If either is set (even to None), the step was completed
        if "selected_model" in context or "recommended_model" in context:
            if selected_model:
                return True, f"Model selected: {selected_model.get('name', 'Unknown')}"
            else:
                return True, "Step skipped - proceeding without model selection"
        
        return False, "Please select a model or skip this step"


class DataPreparationStep(PipelineStep):
    """Data preparation step with PDF upload and processing."""

    def render_ui(self) -> None:
        """Render UI for data preparation step."""
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 2 of 5</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # File upload section
        st.markdown("#### Upload Documents")

        # Hide uploaded files area using custom CSS
        st.markdown(
            """
            <style>
            [data-testid="stFileUploader"] section:not([data-testid="stFileUploaderDropzone"]) {
                display: none !important;
            }
            [data-testid="stFileUploader"] > section + section {
                display: none !important;
            }
            [data-testid="stFileUploader"] ul {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files with domain-specific content",
            key="dp_pdf_uploader",
        )

        if uploaded_files:
            # Display uploaded files
            with st.expander("View uploaded files", expanded=False):
                for idx, file in enumerate(uploaded_files, 1):
                    file_size_mb = file.size / (1024 * 1024)
                    st.write(f"{idx}. **{file.name}** ({file_size_mb:.2f} MB)")

        # Processing configuration
        st.markdown("#### Processing Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Docling native chunking strategies
            strategy_options = {
                "hybrid": "Hybrid (Recommended)",
                "hierarchical": "Hierarchical",
            }

            chunking_strategy_display = st.selectbox(
                "Chunking Strategy",
                options=list(strategy_options.values()),
                index=0,  # Default to "Hybrid"
                help=(
                    "Hybrid: Advanced tokenization-aware chunking that preserves document structure "
                    "and respects token limits. Hierarchical: One chunk per document element, "
                    "preserving document hierarchy."
                ),
                key="dp_chunking_strategy_display",
            )

            # Convert back to enum value
            chunking_strategy = [k for k, v in strategy_options.items() if v == chunking_strategy_display][0]

        with col2:
            # Add vertical spacing to align with the selectbox
            st.markdown('<div style="height: 34px;"></div>', unsafe_allow_html=True)
            ocr_enabled = st.checkbox("Enable OCR", value=True, help="Enable OCR for scanned documents", key="dp_ocr_enabled")

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            st.markdown("**Tokenization Settings**")

            col_adv1, col_adv2 = st.columns(2)

            with col_adv1:
                max_tokens = st.slider(
                    "Max Tokens per Chunk",
                    min_value=64,
                    max_value=8192,
                    value=512,
                    step=64,
                    help="Maximum tokens per chunk for Docling chunkers",
                    key="dp_max_tokens",
                )

            with col_adv2:
                merge_peers = st.checkbox(
                    "Merge Small Chunks",
                    value=True,
                    help="Merge undersized chunks with same headings (Hybrid strategy only)",
                    key="dp_merge_peers",
                )

            st.markdown("**Document Processing**")
            extract_tables = st.checkbox("Extract Tables", value=True, help="Extract and format tables from documents", key="dp_extract_tables")

        dataset_name = st.text_input(
            "Dataset Name",
            help="Name for the generated dataset",
            key="dp_dataset_name",
        )

        dataset_description = st.text_area(
            "Dataset Description",
            value="",
            help="Optional description of the dataset",
            key="dp_dataset_description",
            height=100,
        )

        # Add skip button
        st.markdown("---")
        if st.button("Pular Preparação de Dados", use_container_width=True, key="dp_skip_btn"):
            # Mark as completed even though skipped
            st.session_state.tp_context["data_prep_skipped"] = True
            st.session_state.tp_context["dataset_config"] = None
            st.info("Preparação de dados pulada. Prosseguindo sem dataset processado.")
            st.rerun()

        # Store configuration in session state
        if uploaded_files:
            st.session_state.dp_uploaded_files = uploaded_files
            st.session_state.dp_config = {
                "chunking_strategy": chunking_strategy,
                "extract_tables": extract_tables,
                "ocr_enabled": ocr_enabled,
                "max_tokens": max_tokens,
                "merge_peers": merge_peers,
                "dataset_name": dataset_name,
                "dataset_description": dataset_description,
            }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute backend logic for data preparation."""
        self.logger.info("Executing data preparation step")

        # Check if step was skipped
        if context.get("data_prep_skipped"):
            self.logger.info("Data preparation step skipped")
            return {
                "processing_job": None,
                "dataset_config": None,
                "chunks_files": [],
            }

        try:
            # Get uploaded files from session state
            uploaded_files = st.session_state.get("dp_uploaded_files", [])
            config = st.session_state.get("dp_config", {})

            if not uploaded_files:
                raise ValueError("No files uploaded")

            # Create processing directory and save uploaded files directly there
            user_data_dir = get_user_data_directory()
            processed_dir = user_data_dir / "processed_documents"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files directly in processed_documents directory
            saved_files: list[str] = []
            for file in uploaded_files:
                file_path = processed_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_files.append(str(file_path))
                self.logger.info(f"Saved uploaded file: {file.name}")

            # Format is always JSONL for causal language modeling
            # Using Docling native chunking strategies only (Hybrid and Hierarchical)
            processing_config = DataProcessingConfig(
                input_files=saved_files,
                output_dir=str(processed_dir),
                extract_tables=config.get("extract_tables", True),
                ocr_enabled=config.get("ocr_enabled", True),
                chunking_strategy=ChunkingStrategy(config.get("chunking_strategy", "hybrid")),
                max_tokens=config.get("max_tokens", 512),
                merge_peers=config.get("merge_peers", True),
            )

            # Get orchestrator and process documents
            from lite_llm_studio.app.app import get_orchestrator

            orchestrator = get_orchestrator()

            # Process documents
            self.logger.info("Starting document processing...")
            job_result = orchestrator.execute_document_processing(processing_config)

            if not job_result:
                raise Exception("Document processing failed")

            self.logger.info("Document processing completed")

            # Collect chunk files
            chunks_files: list[str] = []
            for doc in job_result.get("processed_documents", []):
                chunks_file = doc.get("metadata", {}).get("chunks_file")
                if chunks_file:
                    chunks_files.append(chunks_file)

            if not chunks_files:
                raise Exception("No chunks were generated from the documents")

            # Create dataset
            datasets_dir = user_data_dir / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = config.get("dataset_name", "my_training_dataset")
            dataset_description = config.get("dataset_description", "")

            self.logger.info(f"Creating dataset: {dataset_name}")
            dataset_result = orchestrator.create_dataset(chunks_files, str(datasets_dir), dataset_name, dataset_description)

            if not dataset_result:
                raise Exception("Dataset creation failed")

            # Store results in context
            result = {
                "processing_job": job_result,
                "dataset_config": dataset_result,
                "chunks_files": chunks_files,
            }

            self.logger.info("Data preparation completed successfully")
            self.logger.info(f"Dataset created: {dataset_name}")

            return result

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this step can be executed."""
        # Check if step was skipped
        if context.get("data_prep_skipped"):
            return True, "Data preparation skipped - proceeding without processed dataset"
        
        uploaded_files = st.session_state.get("dp_uploaded_files", [])

        if not uploaded_files:
            return False, "Please upload at least one PDF file or skip this step"

        dataset_name = st.session_state.get("dp_config", {}).get("dataset_name", "").strip()
        if not dataset_name:
            return False, "Please provide a dataset name"

        return True, "Ready to process documents"


class DryRunStep(PipelineStep):
    """Dry run step for benchmarking and validating model configuration."""

    def render_ui(self) -> None:
        """Render UI for dry run step."""
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 3 of 5</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render the dry run UI inline
        self._render_dry_run_ui()

        # Add skip button for dry run
        st.markdown("---")
        if st.button("Pular Dry Run", use_container_width=True, key="dr_skip_btn"):
            # Mark as completed even though skipped
            st.session_state.tp_context["dry_run_completed"] = True
            st.session_state.tp_context["dry_run_results"] = None
            st.info("Dry run pulado. Prosseguindo para o treinamento.")
            st.rerun()

    def _render_dry_run_ui(self) -> None:
        """Render the dry run UI inline."""
        st.markdown('<div class="section-title">Dry Run</div>', unsafe_allow_html=True)

        # Ensure high-contrast for uploader button text
        st.markdown(
            """
            <style>
            [data-testid="stFileUploader"] button { color: #ffffff !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        models = _list_local_models()
        if not models:
            st.info("Nenhum modelo encontrado. Baixe um modelo na página de Recommendations.")
            return

        # Custom label (white) + collapse default label
        st.markdown('<div style="color:#ffffff; margin-bottom:4px;">Selecione o modelo local</div>', unsafe_allow_html=True)
        selected = st.selectbox("Selecione o modelo local:", models, label_visibility="collapsed")

        # Check if there's a dataset from data preparation step
        dataset_config = st.session_state.get("tp_context", {}).get("dataset_config")
        dataset_path = dataset_config.get("dataset_path") if dataset_config else None

        # Data source selection
        st.markdown("---")
        st.markdown('<div style="color:#ffffff; margin-bottom:8px;">Fonte de dados para o teste</div>', unsafe_allow_html=True)

        data_source_options = ["Usar dataset conhecido"]
        if dataset_path and os.path.exists(dataset_path):
            data_source_options.insert(0, "Usar dataset da etapa de preparação")

        data_source = st.radio(
            "Fonte de dados",
            data_source_options,
            index=0,
            horizontal=False,
            label_visibility="collapsed",
        )

        # Initialize variables
        use_builtin = False
        use_prepared_dataset = False
        precision_to_mb = {"Baixa": 1, "Média": 2, "Alta": 4}
        target_mb = precision_to_mb["Média"]
        builtin_name = None
        max_samples = 100

        if data_source == "Usar dataset da etapa de preparação":
            use_prepared_dataset = True
            st.info(f"📄 Dataset: `{os.path.basename(dataset_path)}`")
            max_samples = st.slider(
                "Número máximo de amostras para teste",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Quantidade de amostras do dataset a usar no teste",
            )

        else:  # "Usar dataset conhecido"
            use_builtin = True
            prec = st.radio("Precisão (tamanho do texto)", list(precision_to_mb.keys()), index=1, horizontal=True)
            target_mb = precision_to_mb.get(prec, 2)
            ds_keys = ["wikitext-2", "wikitext-103", "ag_news"]
            ds_labels = [f"{k} (~{target_mb} MB)" for k in ds_keys]
            choice = st.selectbox("Dataset", ds_labels, index=0)
            builtin_name = ds_keys[ds_labels.index(choice)]

        device_info = _get_device_info()

        if st.button("Executar Dry Run", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()

            try:
                progress.progress(5)
                status.info("Preparando texto…")
                
                if use_prepared_dataset:
                    # Load from prepared dataset (limit samples for safety)
                    safe_max_samples = min(max_samples, 50)  # Limit to 50 samples for dry run
                    logger.info(f"Loading prepared dataset, max samples: {safe_max_samples}")
                    text, total_bytes, desc = _load_text_from_dataset(dataset_path, safe_max_samples)
                    if total_bytes == 0:
                        st.error(desc or "Falha ao carregar dataset preparado.")
                        return
                    dataset_desc = desc
                    logger.info(f"Loaded {total_bytes} bytes from prepared dataset")
                    
                else:
                    # Load from built-in dataset (limit to 500KB for safety)
                    safe_target_bytes = min(int(target_mb * 1_000_000), 500_000)
                    logger.info(f"Loading built-in dataset, target bytes: {safe_target_bytes}")
                    text, total_bytes, desc = _load_builtin_text(
                        "wikitext-2" if builtin_name == "wikitext-2" else ("wikitext-103" if builtin_name == "wikitext-103" else "ag_news"),
                        target_bytes=safe_target_bytes,
                    )
                    if total_bytes == 0:
                        st.error(desc or "Falha ao carregar dataset.")
                        return
                    dataset_desc = f"{desc} ~{total_bytes / 1_000_000:.1f} MB"
                    logger.info(f"Loaded {total_bytes} bytes from built-in dataset")

                if not device_info.get("has_torch"):
                    st.error("PyTorch/Transformers não instalado. Instale: pip install torch transformers datasets")
                    return

                progress.progress(15)
                status.info("Coletando memória inicial…")
                mem_before = _measure_memory_usage(device_info)

                progress.progress(30)
                status.info("Carregando modelo…")
                model_dir = os.path.join(get_models_dir(), selected)
                try:
                    model, tok = _load_model_tokenizer(model_dir, device_info, force_gpu=bool(device_info.get("has_cuda")))
                except RuntimeError as ex:
                    st.error(f"Falha ao carregar modelo: {ex}")
                    return

                progress.progress(45)
                status.info("Medindo TTFT…")
                ttft_s = _measure_ttft(model, tok, device_info)

                progress.progress(60)
                status.info("Tokenizando com truncamento…")
                max_len = _resolve_max_len(model, tok)
                
                # Limit max_len to prevent memory issues (use smaller context for dry run)
                safe_max_len = min(max_len, 512)  # Limit to 512 tokens for dry run
                logger.info(f"Using max_len={safe_max_len} for dry run (model supports {max_len})")
                
                batch_tokens, seq_len = _tokenize_with_truncation(tok, text, device_info, safe_max_len)
                truncated_note = " (truncado)" if seq_len == safe_max_len else ""
                
                logger.info(f"Tokenization complete: {seq_len} tokens")

                progress.progress(80)
                status.info(f"Executando forward… ({seq_len} tokens)")
                logger.info(f"About to measure forward pass with {seq_len} tokens")
                proc_s = _measure_forward(model, batch_tokens, device_info)
                logger.info(f"Forward measurement returned: {proc_s}")

                progress.progress(90)
                status.info("Coletando memória pós-inferência…")
                mem_after_infer = _measure_memory_usage(device_info)

                progress.progress(100)
                status.success("Dry Run concluído.")
                
                # Check if forward pass failed
                if proc_s != proc_s:  # Check for NaN
                    st.warning("⚠️ Forward pass falhou. Verifique os logs para mais detalhes. Isso pode indicar problemas de memória.")
                    
            except Exception as dry_run_error:
                logger.error(f"Dry run failed: {dry_run_error}", exc_info=True)
                progress.progress(100)
                status.error(f"Erro durante o Dry Run: {str(dry_run_error)}")
                st.error(f"""
                    **Erro durante o Dry Run:**
                    
                    {str(dry_run_error)}
                    
                    **Possíveis causas:**
                    - Memória insuficiente (RAM ou VRAM)
                    - Modelo muito grande para o hardware
                    - Problemas com PyTorch/CUDA
                    
                    **Sugestões:**
                    - Tente um modelo menor
                    - Feche outros programas para liberar memória
                    - Use o botão "Pular Dry Run" para prosseguir
                """)
                
                # Try to clean up
                try:
                    import torch
                    if device_info.get("has_cuda"):
                        torch.cuda.empty_cache()
                    del model, tok
                except Exception:
                    pass
                return
            finally:
                time.sleep(0.2)

            # Determine actual device robustly
            try:
                actual_device = _infer_actual_device(model, batch_tokens, device_info)
            except Exception:
                actual_device = "cpu"

            # Render results card
            rows = []
            rows.append(_render_kv_row("Modelo", selected))
            if actual_device == "cuda":
                rows.append(_render_kv_row("Dispositivo", f"GPU ({device_info.get('cuda_name')})"))
            else:
                rows.append(_render_kv_row("Dispositivo", "CPU"))
            rows.append(_render_kv_row("Dataset usado", f"{dataset_desc} • {seq_len} tokens{truncated_note}"))
            rows.append(_render_kv_row("TTFT", f"{ttft_s:.2f} s" if ttft_s == ttft_s else "—"))
            rows.append(_render_kv_row("Tempo p/ processar dataset", f"{proc_s:.2f} s" if proc_s == proc_s else "—"))
            rows.append(_render_kv_row("Memória RAM antes", _bytes_to_mb_gb(mem_before.get("rss", 0))))
            rows.append(_render_kv_row("Memória RAM após inferência", _bytes_to_mb_gb(mem_after_infer.get("rss", 0))))
            if actual_device == "cuda":
                rows.append(_render_kv_row("VRAM alocada", _bytes_to_mb_gb(mem_after_infer.get("cuda_alloc") or 0)))
                rows.append(_render_kv_row("VRAM reservada", _bytes_to_mb_gb(mem_after_infer.get("cuda_reserved") or 0)))

            html = (
                '<div class="syscpu-card">'
                '  <div class="syscpu-header">'
                '    <div class="syscpu-left">'
                '      <div>'
                '        <div class="syscpu-title">Resultados do Dry Run</div>'
                f'        <div class="syscpu-sub">{time.strftime("%Y-%m-%d %H:%M:%S")}</div>'
                '      </div>'
                '    </div>'
                '    <div class="syscpu-tags">'
                f'      <span class="chip" style="color:black;">Tokens: {seq_len}</span>'
                f'      <span class="chip" style="color:black;">Entrada: {dataset_desc}</span>'
                '    </div>'
                '  </div>'
                f'  <div class="spec-grid">{"".join(rows)}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

            # Save dry run results to pipeline context
            if "tp_context" in st.session_state:
                dry_run_results = {
                    "model": selected,
                    "device": actual_device,
                    "dataset_desc": dataset_desc,
                    "seq_len": seq_len,
                    "ttft_s": ttft_s,
                    "proc_s": proc_s,
                    "mem_before": mem_before,
                    "mem_after_infer": mem_after_infer,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.tp_context["dry_run_results"] = dry_run_results
                st.session_state.tp_context["dry_run_completed"] = True

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute dry run logic."""
        self.logger.info("Executing dry run step")
        
        # Get dry run results from context (set by dry run page)
        dry_run_results = context.get("dry_run_results")
        dry_run_completed = context.get("dry_run_completed", False)
        
        if dry_run_completed:
            if dry_run_results:
                self.logger.info("Dry run completed successfully with results")
                return {"dry_run_completed": True, "dry_run_results": dry_run_results}
            else:
                self.logger.info("Dry run skipped")
                return {"dry_run_completed": True, "dry_run_results": None}
        
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if dry run has been completed or skipped."""
        dry_run_completed = context.get("dry_run_completed", False)
        if not dry_run_completed:
            return False, "Please complete or skip the dry run before proceeding"
        
        dry_run_results = context.get("dry_run_results")
        if dry_run_results:
            return True, "Dry run completed successfully"
        else:
            return True, "Dry run skipped - proceeding to training"


class TrainingStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """Render the UI for the training step."""
        # Step header
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 4 of 5</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Select Dataset and Base Model")

        # Locate datasets directory under the user data folder. Datasets are
        # saved during the data preparation step as ``user_data_dir/datasets/<name>``.
        user_data_dir = get_user_data_directory()
        datasets_root = user_data_dir / "datasets"
        dataset_options: list[str] = []
        dataset_paths: dict[str, str] = {}
        if datasets_root.exists():
            for p in datasets_root.iterdir():
                if p.is_dir() and (p / "train.jsonl").exists():
                    dataset_options.append(p.name)
                    dataset_paths[p.name] = str(p)
        # If no datasets are found, inform the user
        if not dataset_options:
            st.warning("No datasets were found. Please complete the Data Preparation step and create a dataset before training.")
        selected_dataset = None
        if dataset_options:
            selected_dataset = st.selectbox(
                "Dataset",
                options=dataset_options,
                help="Select a dataset directory containing train/validation/test JSONL files",
                key="tp_dataset_select",
            )
        # Store selection in session state for execution
        if selected_dataset:
            st.session_state.tp_selected_dataset = dataset_paths[selected_dataset]

        # Locate base models. Use the default models directory; if it doesn't
        # exist, fallback to a ``models`` directory in the project root.
        models_root = get_default_models_directory()
        if not models_root.exists():
            models_root = Path("models")
        model_options: list[str] = []
        model_paths: dict[str, str] = {}
        gguf_models: list[str] = []  # Track GGUF models separately
        meta_native_models: list[str] = []  # Track Meta native format models

        if models_root.exists():
            for p in models_root.iterdir():
                if p.is_dir():
                    # Check for PyTorch/Transformers format (compatible)
                    has_config = (p / "config.json").exists()

                    # Check for weights in various formats
                    has_single_weights = any((p / fname).exists() for fname in ["pytorch_model.bin", "model.safetensors"])
                    has_index = (p / "pytorch_model.bin.index.json").exists() or (p / "model.safetensors.index.json").exists()
                    has_sharded_weights = any(f.name.startswith("model-") and f.suffix == ".safetensors" for f in p.glob("*.safetensors"))

                    has_weights = has_single_weights or has_index or has_sharded_weights

                    if has_config and has_weights:
                        model_options.append(p.name)
                        model_paths[p.name] = str(p)
                    # Check for GGUF format (incompatible with training)
                    elif any(f.suffix == ".gguf" for f in p.glob("*.gguf")):
                        gguf_models.append(p.name)
                    # Check for Meta native format (consolidated.pth + params.json)
                    elif (p / "params.json").exists() and any(f.name.startswith("consolidated") and f.suffix == ".pth" for f in p.glob("*.pth")):
                        meta_native_models.append(p.name)
        selected_model = None
        if model_options:
            selected_model = st.selectbox(
                "Base Model",
                options=model_options,
                help="Select the base model to fine‑tune",
                key="tp_model_select",
            )
        else:
            st.error(f"**No compatible models found in**: `{models_root}`")

            # Show GGUF models if found, but explain they're incompatible
            if gguf_models:
                st.warning(
                    """
                    **Modelos GGUF detectados não compatíveis com fine-tuning:**
                    """
                )

            # Show Meta native format models if found
            if meta_native_models:
                st.warning(
                    """
                    **Modelos no formato Meta nativo detectados (não compatíveis):**
                    """
                )

            # Button to open models directory in Windows Explorer
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("Abrir Pasta de Modelos", key="open_models_dir"):
                    import subprocess

                    # Create directory if it doesn't exist
                    models_root.mkdir(parents=True, exist_ok=True)
                    # Open in Windows Explorer
                    subprocess.Popen(f'explorer "{models_root}"')
                    st.success(f"Abrindo: {models_root}")
            with col_btn2:
                if st.button("Recarregar Lista de Modelos", key="reload_models"):
                    st.rerun()

        if selected_model:
            st.session_state.tp_selected_model = model_paths[selected_model]

        st.markdown("---")
        st.markdown("#### Training Hyperparameters")
        # Hyperparameter inputs. Use reasonable defaults for LoRA fine‑tuning.
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Number of epochs to train",
                key="tp_epochs",
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
                help="Per‑device batch size",
                key="tp_batch_size",
            )
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-3,
                value=2e-4,
                step=1e-5,
                format="%f",
                help="Initial learning rate for the optimiser",
                key="tp_learning_rate",
            )
            max_seq_length = st.number_input(
                "Max Sequence Length",
                min_value=128,
                max_value=4096,
                value=1024,
                step=128,
                help="Maximum sequence length (context window)",
                key="tp_max_seq_length",
            )
        with col3:
            lora_r = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                max_value=64,
                value=8,
                step=1,
                help="Rank of LoRA adapters (controls adapter size)",
                key="tp_lora_r",
            )
            lora_alpha = st.number_input(
                "LoRA Alpha",
                min_value=1,
                max_value=256,
                value=16,
                step=1,
                help="LoRA alpha scaling factor",
                key="tp_lora_alpha",
            )

        # Early Stopping Configuration
        st.markdown("---")
        st.markdown("#### Early Stopping (Optional)")

        enable_early_stopping = st.checkbox(
            "Enable Early Stopping",
            value=False,
            help="Stop training early if validation loss stops improving. **Requires a validation dataset to work.**",
            key="tp_enable_early_stopping",
        )

        early_stopping_patience = None
        early_stopping_threshold = None

        if enable_early_stopping:
            col_es1, col_es2 = st.columns(2)
            with col_es1:
                early_stopping_patience = st.number_input(
                    "Patience (epochs)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="Number of epochs with no improvement after which training will be stopped",
                    key="tp_early_stopping_patience",
                )
            with col_es2:
                early_stopping_threshold = st.number_input(
                    "Min Delta",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.001,
                    step=0.001,
                    format="%.4f",
                    help="Minimum change in validation loss to qualify as an improvement",
                    key="tp_early_stopping_threshold",
                )

        # Store hyperparameters in session state for later retrieval
        st.session_state.tp_training_params = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "max_seq_length": int(max_seq_length),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "enable_early_stopping": enable_early_stopping,
            "early_stopping_patience": int(early_stopping_patience) if early_stopping_patience else None,
            "early_stopping_threshold": float(early_stopping_threshold) if early_stopping_threshold else None,
        }

        st.markdown("---")
        st.markdown("#### Output Configuration")
        output_name = st.text_input(
            "Output Model Name",
            help="Name of the directory to save the fine‑tuned model",
            key="tp_output_name",
        )
        if output_name:
            st.session_state.tp_output_model_name = output_name.strip()

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run the fine‑tuning procedure on the selected dataset and model."""
        self.logger.info("Executing training step")
        # Retrieve selections from session state
        dataset_dir = st.session_state.get("tp_selected_dataset")
        base_model_path = st.session_state.get("tp_selected_model")
        params = st.session_state.get("tp_training_params", {})
        output_name = st.session_state.get("tp_output_model_name")
        if not dataset_dir:
            raise ValueError("No dataset selected. Please choose a dataset before training.")
        if not base_model_path:
            raise ValueError("No base model selected. Please choose a model before training.")
        if not output_name:
            raise ValueError("No output model name provided.")

        # Construct absolute dataset and output paths
        # Ensure dataset directory exists
        ds_path = Path(dataset_dir)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_path}")
        # Output directory will reside under models directory
        models_root = get_default_models_directory()
        if not models_root.exists():
            models_root = Path("models")
        output_dir = models_root / output_name
        # Build training configuration dictionary
        training_cfg: dict[str, Any] = {
            "dataset_dir": str(ds_path),
            "base_model_path": str(base_model_path),
            "output_dir": str(output_dir),
            "epochs": params.get("epochs", 1),
            "batch_size": params.get("batch_size", 4),
            "learning_rate": params.get("learning_rate", 2e-4),
            "max_seq_length": params.get("max_seq_length", 1024),
            "lora_r": params.get("lora_r", 8),
            "lora_alpha": params.get("lora_alpha", 16),
            "enable_early_stopping": params.get("enable_early_stopping", False),
            "early_stopping_patience": params.get("early_stopping_patience"),
            "early_stopping_threshold": params.get("early_stopping_threshold"),
        }
        # Log configuration
        self.logger.info(f"Training configuration: {training_cfg}")
        # Obtain orchestrator and run training
        from lite_llm_studio.app.app import get_orchestrator

        orchestrator = get_orchestrator()

        # Training will be executed (Streamlit will show processing state automatically)
        self.logger.info("Starting model training...")
        result = orchestrator.execute_training(training_cfg)

        if not result or not result.get("trained_model_path"):
            raise RuntimeError("Model training failed or returned no output.")

        trained_model_path = result["trained_model_path"]
        self.logger.info(f"Training completed successfully: {trained_model_path}")

        # Update context
        return {"trained_model": trained_model_path}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this training step is ready to be executed.

        Checks for the presence of a selected dataset, selected base model
        and a non‑empty output model name. Also ensures that training
        hyperparameters have been configured.

        Args:
            context: Shared pipeline context (unused here).

        Returns:
            Tuple[bool, str]: A boolean indicating readiness, and a
            message explaining the issue if not ready.
        """
        if not st.session_state.get("tp_selected_dataset"):
            return False, "Please select a dataset to use for training"
        if not st.session_state.get("tp_selected_model"):
            return False, "Please select a base model"
        if not st.session_state.get("tp_output_model_name"):
            return False, "Please provide an output model name"
        # Check hyperparameters exist
        if not st.session_state.get("tp_training_params"):
            return False, "Please configure training hyperparameters"
        return True, "Ready to train"


class ModelExportStep(PipelineStep):
    """Model export step with GGUF conversion and evaluation."""

    def render_ui(self) -> None:
        """Render UI for model export step."""
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 5 of 5</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Check if we have a trained model
        trained_model = st.session_state.get("tp_context", {}).get("trained_model")
        if not trained_model:
            st.warning("No trained model found. Please complete the Training step first.")
            return

        # Check if evaluation has been completed
        eval_results = st.session_state.get("tp_eval_results")
        gguf_path = st.session_state.get("tp_gguf_model_path")

        if not eval_results or not gguf_path:
            # Show simple interface for starting the process
            st.markdown("#### Model Export & Evaluation")
            st.markdown(
                """
                This will automatically:
                - Convert your fine-tuned model to **GGUF format** with your chosen quantization
                - Evaluate the model using **perplexity** on validation data
                - Compare performance against the base model
                """
            )

            # Quantization selector
            col1, col2 = st.columns([2, 3])
            with col1:
                quantization = st.selectbox(
                    "Quantization Format",
                    options=["f16", "bf16", "q8_0", "f32"],
                    index=0,
                    help=(
                        "• f16: Float16 - Recommended (good quality, reasonable size)\n"
                        "• bf16: BFloat16 - Better numerical stability\n"
                        "• q8_0: 8-bit quantization - Smaller size, slight quality loss\n"
                        "• f32: Float32 - No compression (largest, highest precision)"
                    ),
                    key="tp_quantization_format",
                )
                # Store selection
                st.session_state.tp_quantization_selected = quantization

            with col2:
                st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

            st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)

            # Check if validation dataset exists
            dataset_dir = st.session_state.get("tp_selected_dataset")
            has_validation = False
            if dataset_dir:
                validation_file = Path(dataset_dir) / "validation.jsonl"
                has_validation = validation_file.exists()

            if not has_validation:
                st.warning(
                    "**No validation dataset found.** Model will be converted to GGUF, "
                    "but perplexity evaluation will be skipped.\n\n"
                    "To enable evaluation, ensure your dataset includes a `validation.jsonl` file."
                )

            st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

        else:
            # Show results after execution
            st.markdown("#### Model Export Completed")
            st.success(f"**GGUF Model saved:** `{Path(gguf_path).name}`")
            st.caption("This is a complete standalone model - ready to use!")

            # Show evaluation results
            st.markdown("---")
            st.markdown("#### Evaluation Results")

            # Metrics in columns
            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.metric("Base Model Perplexity", f"{eval_results['base_perplexity']:.2f}", help="Lower is better")

            with col_m2:
                st.metric(
                    "Fine-tuned Model Perplexity",
                    f"{eval_results['finetuned_perplexity']:.2f}",
                    delta=f"{-eval_results['improvement']:.2f}",
                    delta_color="inverse",
                    help="Lower is better. Delta shows improvement.",
                )

            with col_m3:
                improvement_pct = eval_results["improvement_pct"]
                st.metric(
                    "Improvement",
                    f"{abs(improvement_pct):.1f}%",
                    delta="Better" if improvement_pct > 0 else "Worse",
                    delta_color="normal" if improvement_pct > 0 else "inverse",
                    help="Percentage improvement in perplexity",
                )

            # Visual comparison

            base_ppl = eval_results["base_perplexity"]
            finetuned_ppl = eval_results["finetuned_perplexity"]
            max_ppl = max(base_ppl, finetuned_ppl)

            base_width = (base_ppl / max_ppl) * 100
            finetuned_width = (finetuned_ppl / max_ppl) * 100

            st.markdown(
                f"""
                <div style="margin: 20px 0;">
                    <div style="margin-bottom: 16px;">
                        <div style="font-weight: 600; margin-bottom: 6px; color: var(--text);">
                            Base Model ({eval_results['quantization']})
                        </div>
                        <div style="background: #f1f5f9; border-radius: 8px; overflow: hidden; height: 32px; position: relative;">
                            <div style="
                                background: linear-gradient(90deg, #ef4444, #dc2626);
                                height: 100%;
                                width: {base_width}%;
                                display: flex;
                                align-items: center;
                                justify-content: flex-end;
                                padding-right: 12px;
                                color: white;
                                font-weight: 700;
                                font-size: 0.9rem;
                            ">
                                {base_ppl:.2f}
                            </div>
                        </div>
                    </div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 6px; color: var(--text);">
                            Fine-tuned Model ({eval_results['quantization']})
                        </div>
                        <div style="background: #f1f5f9; border-radius: 8px; overflow: hidden; height: 32px; position: relative;">
                            <div style="
                                background: linear-gradient(90deg, #10b981, #059669);
                                height: 100%;
                                width: {finetuned_width}%;
                                display: flex;
                                align-items: center;
                                justify-content: flex-end;
                                padding-right: 12px;
                                color: white;
                                font-weight: 700;
                                font-size: 0.9rem;
                            ">
                                {finetuned_ppl:.2f}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interpretation
            st.markdown("**Interpretation**")
            if improvement_pct > 15:
                st.success(
                    f"**Excellent!** Your fine-tuned model shows significant improvement "
                    f"({improvement_pct:.1f}% better perplexity). The model has successfully "
                    f"adapted to your domain-specific data."
                )
            elif improvement_pct > 0:
                st.info(
                    f"**Good!** Your fine-tuned model shows improvement ({improvement_pct:.1f}% better). "
                    f"Consider training for more epochs or with more data for further gains."
                )
            else:
                st.warning(
                    f"The fine-tuned model shows {abs(improvement_pct):.1f}% worse perplexity. "
                    f"This could indicate overfitting, insufficient training data, or that the base "
                    f"model was already well-suited for this task. "
                )

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute backend logic for model export (conversion and evaluation)."""
        self.logger.info("Executing model export step")

        try:
            from lite_llm_studio.core.configuration.desktop_app_config import get_default_models_directory
            from lite_llm_studio.core.ml.evaluation import evaluate_model_perplexity
            from lite_llm_studio.core.ml.model_converter import convert_finetuned_model_to_gguf, convert_hf_to_gguf

            # Get trained model path
            trained_model = context.get("trained_model")
            if not trained_model:
                raise ValueError("No trained model found in context")

            # Get base model and output name
            base_model = st.session_state.get("tp_base_model_path") or st.session_state.get("tp_selected_model")
            output_name = st.session_state.get("tp_output_model_name", "finetuned_model")
            quantization = st.session_state.get("tp_quantization_selected", "f16")

            if not base_model:
                raise ValueError("Base model path not found")

            # Convert to GGUF
            self.logger.info(f"Converting fine-tuned model to GGUF ({quantization})...")
            models_dir = str(get_default_models_directory())
            result = convert_finetuned_model_to_gguf(
                adapter_path=str(trained_model),
                base_model_path=str(base_model),
                output_name=output_name,
                quantization=quantization,
                models_dir=models_dir,
            )
            gguf_path = result["gguf_model"]
            st.session_state.tp_gguf_model_path = gguf_path
            st.session_state.tp_quantization_used = quantization
            st.session_state.tp_base_model_path = str(base_model)
            self.logger.info(f"GGUF model created: {gguf_path}")

            # Evaluate if validation dataset exists
            eval_results = None
            dataset_dir = st.session_state.get("tp_selected_dataset")
            if dataset_dir:
                validation_file = Path(dataset_dir) / "validation.jsonl"
                if validation_file.exists():
                    max_samples = 100  # Fixed to 100 samples

                    self.logger.info(f"Starting model evaluation with {max_samples} samples...")
                    self.logger.info(f"Both models will use {quantization} quantization for fair comparison")

                    # Convert base model to GGUF with same quantization
                    self.logger.info(f"Converting base model to GGUF ({quantization})...")
                    models_dir_path = Path(get_default_models_directory())
                    base_gguf_dir = models_dir_path / f"_eval_base_{Path(base_model).name}"
                    base_gguf_dir.mkdir(parents=True, exist_ok=True)

                    base_gguf_path = convert_hf_to_gguf(
                        model_path=str(base_model), output_path=str(base_gguf_dir), quantization=quantization, output_name="base_model_eval"
                    )
                    self.logger.info(f"Base model GGUF created: {base_gguf_path}")

                    # Evaluate base model
                    self.logger.info("=" * 80)
                    self.logger.info("EVALUATING BASE MODEL (Step 1/2)")
                    self.logger.info("=" * 80)
                    base_perplexity = evaluate_model_perplexity(
                        model_path=base_gguf_path, validation_file=str(validation_file), max_samples=int(max_samples)
                    )
                    self.logger.info(f"Base model perplexity: {base_perplexity:.2f}")

                    # Evaluate fine-tuned model
                    self.logger.info("=" * 80)
                    self.logger.info("EVALUATING FINE-TUNED MODEL (Step 2/2)")
                    self.logger.info("=" * 80)
                    finetuned_perplexity = evaluate_model_perplexity(
                        model_path=gguf_path, validation_file=str(validation_file), max_samples=int(max_samples)
                    )
                    self.logger.info(f"Fine-tuned model perplexity: {finetuned_perplexity:.2f}")

                    # Calculate improvement
                    improvement = base_perplexity - finetuned_perplexity
                    improvement_pct = ((base_perplexity - finetuned_perplexity) / base_perplexity) * 100

                    # Store results
                    eval_results = {
                        "base_perplexity": base_perplexity,
                        "finetuned_perplexity": finetuned_perplexity,
                        "improvement": improvement,
                        "improvement_pct": improvement_pct,
                        "quantization": quantization,
                        "samples_evaluated": max_samples,
                    }
                    st.session_state.tp_eval_results = eval_results

                    # Log summary
                    self.logger.info("=" * 80)
                    self.logger.info("EVALUATION SUMMARY")
                    self.logger.info("=" * 80)
                    self.logger.info(f"Quantization used: {quantization} (same for both models)")
                    self.logger.info(f"Samples evaluated: {max_samples}")
                    self.logger.info(f"Base model perplexity: {base_perplexity:.2f}")
                    self.logger.info(f"Fine-tuned model perplexity: {finetuned_perplexity:.2f}")
                    self.logger.info(f"Improvement: {improvement:.2f} ({improvement_pct:+.1f}%)")
                    self.logger.info("=" * 80)

            return {"gguf_model": gguf_path, "evaluation_results": eval_results}

        except Exception as e:
            self.logger.error(f"Model export failed: {e}", exc_info=True)
            raise

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if model export step can be executed."""
        trained_model = context.get("trained_model")
        if not trained_model:
            return False, "Please complete the Training step first"

        return True, "Ready to export and evaluate model"


# Pipeline registry
PIPELINE_REGISTRY: list[tuple[PipelineStepConfig, type[PipelineStep]]] = [
    (
        PipelineStepConfig(
            key="model_reco",
            label="Model Recommendation",
            icon=ICONS.get("model", ""),
            description="Select and configure the base model for fine-tuning based on your requirements.",
        ),
        ModelRecommendationStep,
    ),
    (
        PipelineStepConfig(
            key="data_prep",
            label="Data Preparation",
            icon=ICONS.get("folder", ""),
            description="Upload and preprocess your training data for optimal results.",
        ),
        DataPreparationStep,
    ),
    (
        PipelineStepConfig(
            key="dry_run",
            label="Dry Run",
            icon=ICONS.get("refresh", ""),
            description="Validate your configuration and estimate training time and resources.",
        ),
        DryRunStep,
    ),
    (
        PipelineStepConfig(
            key="training", label="Training", icon=ICONS.get("play", ""), description="Execute the fine-tuning process and monitor training progress."
        ),
        TrainingStep,
    ),
    (
        PipelineStepConfig(
            key="model_export",
            label="Model Export",
            icon=ICONS.get("save", ""),
            description="Convert your model to GGUF format and evaluate its performance.",
        ),
        ModelExportStep,
    ),
]


class TrainingPipeline:
    """Manages the training pipeline state and execution."""

    def __init__(self) -> None:
        self.steps: list[PipelineStep] = []
        self.configs: list[PipelineStepConfig] = []

        # Initialize pipeline steps from registry
        for config, step_class in PIPELINE_REGISTRY:
            self.configs.append(config)
            self.steps.append(step_class(config))

        self.logger = logging.getLogger("app.pipeline.manager")

    def get_step_count(self) -> int:
        """Get total number of steps in the pipeline."""
        return len(self.steps)

    def get_step(self, index: int) -> PipelineStep:
        """Get pipeline step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        raise IndexError(f"Step index {index} out of range")

    def get_config(self, index: int) -> PipelineStepConfig:
        """Get pipeline step config by index."""
        if 0 <= index < len(self.configs):
            return self.configs[index]
        raise IndexError(f"Step index {index} out of range")

    def execute_step(self, index: int, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a specific pipeline step."""
        step = self.get_step(index)
        self.logger.info(f"Executing step {index}: {step.config.label}")
        return step.execute(context)

    def validate_step(self, index: int, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate a specific pipeline step."""
        step = self.get_step(index)
        return step.validate(context)


def _get_pipeline() -> TrainingPipeline:
    """Get or create the training pipeline instance."""
    if "tp_pipeline" not in st.session_state:
        st.session_state.tp_pipeline = TrainingPipeline()
    return st.session_state.tp_pipeline


def _init_pipeline_state() -> None:
    """Initialize pipeline state in session."""
    pipeline = _get_pipeline()

    if "tp_current_step" not in st.session_state:
        st.session_state.tp_current_step = 0
    if "tp_completed" not in st.session_state:
        st.session_state.tp_completed = [False] * pipeline.get_step_count()
    if "tp_unlocked" not in st.session_state:
        # Only the first step is unlocked initially
        st.session_state.tp_unlocked = [i == 0 for i in range(pipeline.get_step_count())]
    if "tp_context" not in st.session_state:
        st.session_state.tp_context = {}  # Shared context between steps
    if "tp_result_model" not in st.session_state:
        st.session_state.tp_result_model = None
    if "tp_processing" not in st.session_state:
        st.session_state.tp_processing = False  # Track if a step is currently processing
    if "tp_pipeline_finished" not in st.session_state:
        st.session_state.tp_pipeline_finished = False  # Track if pipeline has been finished


def _render_stepper_header() -> None:
    """Render the visual stepper header showing pipeline progress."""
    pipeline = _get_pipeline()
    current = st.session_state.tp_current_step
    completed = st.session_state.tp_completed

    # Inline CSS just for the stepper
    st.markdown(
        """
        <style>
        .tp-stepper { display:flex; align-items:center; gap:14px; margin: 10px 0 24px; }
        .tp-step { display:flex; align-items:center; gap:10px; }
        .tp-node {
            width: 34px; height: 34px; border-radius: 999px;
            display:flex; align-items:center; justify-content:center;
            font-weight: 800; font-size: .95rem;
            border: 2px solid var(--border);
            background: var(--panel); color: var(--muted);
            box-shadow: var(--shadow-1);
        }
        .tp-node.active { border-color: var(--primary); color: var(--primary); background: var(--primary-weak); }
        .tp-node.done { border-color: #10b981; color: #10b981; background: color-mix(in oklab, #10b981 12%, white); }
        .tp-label { font-weight: 700; font-size: .92rem; color: var(--text); white-space: nowrap; }
        .tp-label.locked { color: #9ca3af; }
        .tp-connector { flex:1; height: 2px; background: var(--border); }
        .tp-connector.active { background: var(--primary); }
        .tp-connector.done { background: #10b981; }
        .tp-icon { width:16px; height:16px; color: currentColor; }
        @media (max-width: 1000px) {
            .tp-label { display: none; } /* keep it compact on small widths */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build the visual stepper
    html = ['<div class="tp-stepper">']
    for i in range(pipeline.get_step_count()):
        config = pipeline.get_config(i)
        status = "locked"
        if completed[i]:
            status = "done"
        elif i == current:
            status = "active"

        node_class = f"tp-node {'active' if status=='active' else ''} {'done' if status=='done' else ''}".strip()
        label_class = f"tp-label {'locked' if status=='locked' else ''}".strip()

        icon_html = config.icon.replace("<svg", '<svg class="tp-icon"')
        html.append(
            f"""
            <div class="tp-step">
              <div class="{node_class}">{i+1}</div>
              <div class="{label_class}">{icon_html} <span style="margin-left:6px;">{config.label}</span></div>
            </div>
            """
        )
        if i < pipeline.get_step_count() - 1:
            # connector between steps
            conn_cls = "tp-connector"
            if completed[i]:
                conn_cls += " done"
            elif i < current:
                conn_cls += " active"
            html.append(f'<div class="{conn_cls}"></div>')
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


def _render_step_selector() -> None:
    """Render the row of step buttons with locked/active states."""
    pipeline = _get_pipeline()
    cols = st.columns(pipeline.get_step_count(), gap="small")

    for i in range(pipeline.get_step_count()):
        config = pipeline.get_config(i)
        with cols[i]:
            unlocked = bool(st.session_state.tp_unlocked[i])
            is_active = i == st.session_state.tp_current_step
            btn_label = f"{i+1}. {config.label}"
            if st.button(btn_label, key=f"tp_btn_{i}", use_container_width=True, disabled=not unlocked, type="primary" if is_active else "secondary"):
                st.session_state.tp_current_step = i
                st.rerun()


def _render_step_panel() -> None:
    """Render the current step panel with its specific UI and controls."""
    pipeline = _get_pipeline()
    i = st.session_state.tp_current_step
    step = pipeline.get_step(i)
    step_key = step.config.key

    # chaves usadas pelo progresso do treinamento
    if "tp_training_progress" not in st.session_state:
        st.session_state.tp_training_progress = 0.0
    if "tp_training_progress_text" not in st.session_state:
        st.session_state.tp_training_progress_text = ""

    # Render do conteúdo do passo
    step.render_ui()
    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Estado de processamento
    is_processing = st.session_state.get("tp_processing", False)

    # SLOT para CSS do footer (permite esconder/mostrar sem quebrar o layout)
    footer_css_slot = st.empty()

    # SLOT para barra de progresso (atualizada ao vivo via callback)
    progress_slot = st.empty()
    progress_widget = None  # será criado quando necessário

    # Aviso de processamento e controles visuais
    if is_processing:
        # Esconde o footer APENAS enquanto processa (sem reescrever o layout original)
        if step_key in ("data_prep", "training", "model_export"):
            footer_css_slot.markdown(
                """
                <style>
                    [data-testid="stFooter"], footer, .app-footer { display: none !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Mensagem do passo
        step_messages = {
            "model_reco": "",
            "data_prep": "Processing documents and creating dataset...",
            "dry_run": "",
            "training": "Training model...",
            "model_export": "Converting model to GGUF and evaluating performance...",
        }
        processing_msg = step_messages.get(step_key, "Processing...")

        # Banner “loading”
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, rgba(37,99,235,.10) 0%, rgba(37,99,235,.20) 50%, rgba(37,99,235,.10) 100%);
                border: 1px solid rgba(37,99,235,.30);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
                text-align: center;
            ">
                <div style="
                    display:inline-block;width:16px;height:16px;border:2px solid rgba(37,99,235,.30);
                    border-top:2px solid #2563eb;border-radius:50%;animation:spin 1s linear infinite;
                    margin-right:12px;vertical-align:middle;
                "></div>
                <span style="font-weight:600;color:#2563eb;vertical-align:middle;">{processing_msg}</span>
            </div>
            <style>
                @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Barra de progresso VIVA apenas no passo de treinamento
        if step_key == "training":
            current_val = float(st.session_state.get("tp_training_progress", 0.0))
            current_val = max(0.0, min(1.0, current_val))
            current_text = st.session_state.get("tp_training_progress_text") or f"{int(current_val * 100)}%"
            progress_widget = progress_slot.progress(current_val, text=current_text)
            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    else:
        # Remove qualquer CSS injetado para o footer (restaura o layout original automaticamente)
        footer_css_slot.empty()
        # Garante que o slot de progresso esteja limpo quando não estiver processando
        progress_slot.empty()

    # Navegação
    left, mid, right = st.columns([1, 2, 1], vertical_alignment="center")
    with left:
        if st.button("🡠 Back", key="tp_nav_back", disabled=(i == 0 or is_processing), use_container_width=True):
            st.session_state.tp_current_step = max(0, i - 1)
            st.rerun()

    with mid:
        is_completed = st.session_state.tp_completed[i]
        can_complete, validation_msg = step.can_complete(st.session_state.tp_context)
        completed_label = "✓ Completed" if is_completed else "Complete Step"
        button_disabled = is_completed or not can_complete or is_processing
        complete_clicked = st.button(
            completed_label,
            key=f"tp_complete_{i}",
            type="secondary" if is_completed else "primary",
            use_container_width=True,
            disabled=button_disabled,
            help=validation_msg if not can_complete else None,
        )
        if complete_clicked and not is_processing:
            st.session_state.tp_processing = True
            st.rerun()

    with right:
        can_go_next = (i < pipeline.get_step_count() - 1) and st.session_state.tp_completed[i] and st.session_state.tp_unlocked[i + 1]
        if st.button("Next 🡢", key="tp_nav_next", disabled=(not can_go_next or is_processing), use_container_width=True):
            st.session_state.tp_current_step = min(pipeline.get_step_count() - 1, i + 1)
            st.rerun()

    # Execução do passo (acontece após o clique em "Complete Step")
    if is_processing and not st.session_state.tp_completed[i]:
        try:
            # Passo de treinamento: registra callback que atualiza a barra em tempo real
            if step_key == "training":
                from lite_llm_studio.app.app import get_orchestrator

                orchestrator = get_orchestrator()

                def _progress_cb(msg: str, percent: float | None) -> None:
                    # normaliza (aceita 0–1 ou 0–100)
                    if percent is None:
                        p = st.session_state.get("tp_training_progress", 0.0)
                    else:
                        p = float(percent)
                        if p > 1.0:
                            p = p / 100.0
                        p = max(0.0, min(1.0, p))

                    text = msg or f"{int(p * 100)}%"
                    # mantém no estado (para o próximo rerender)…
                    st.session_state.tp_training_progress = p
                    st.session_state.tp_training_progress_text = text
                    # …e atualiza o widget imediatamente
                    if progress_widget is not None:
                        progress_widget.progress(p, text=text)

                orchestrator.set_training_progress_callback(_progress_cb)

            # Executa o backend do passo
            result = pipeline.execute_step(i, st.session_state.tp_context)
            st.session_state.tp_context.update(result)
            st.session_state.tp_completed[i] = True

            # Desbloqueia próximo passo
            if i + 1 < pipeline.get_step_count():
                st.session_state.tp_unlocked[i + 1] = True
                st.session_state.tp_current_step = i + 1

            st.success(f"{step.config.label} completed successfully!")
        except Exception as e:
            st.error(f"Error completing step: {str(e)}")
        finally:
            st.session_state.tp_processing = False
            if step_key == "training":
                st.session_state.tp_training_progress = 1.0
                if not st.session_state.get("tp_training_progress_text"):
                    st.session_state.tp_training_progress_text = "Training finished."
                if progress_widget is not None:
                    progress_widget.progress(1.0, text=st.session_state.tp_training_progress_text)
            st.rerun()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Finish Pipeline button (only on last step)
    if i == pipeline.get_step_count() - 1:
        finished = all(st.session_state.tp_completed)
        if st.button("Finish Pipeline & Return to Home", key="tp_finish", disabled=not finished, use_container_width=True, type="primary"):
            try:
                # Clear pipeline state
                st.session_state.tp_pipeline_finished = True
                st.session_state.current_page = "Home"
                st.success("Pipeline completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error finishing pipeline: {str(e)}")


def _render_pipeline_summary() -> None:
    """Render a summary of the pipeline progress."""
    pipeline = _get_pipeline()
    completed_count = sum(st.session_state.tp_completed)
    total_count = pipeline.get_step_count()

    progress = completed_count / total_count if total_count > 0 else 0

    st.progress(progress, text=f"Pipeline Progress: {completed_count}/{total_count} steps completed")


def render_training_page() -> None:
    """Main function to render the training page with scalable pipeline architecture."""
    logger.info("Rendering training page")

    # Initialize pipeline state
    _init_pipeline_state()

    # Render main UI components
    _render_stepper_header()
    _render_pipeline_summary()
    _render_step_selector()
    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    _render_step_panel()
