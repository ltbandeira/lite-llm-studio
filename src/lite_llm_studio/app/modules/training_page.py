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
from datetime import datetime
from typing import Any

import pandas as pd
import psutil
import streamlit as st

from lite_llm_studio.core.configuration.data_schema import ChunkingStrategy, DataProcessingConfig
from lite_llm_studio.core.configuration.desktop_app_config import get_user_data_directory, get_default_models_directory

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

        with torch.no_grad():
            start = time.perf_counter()
            _ = model(**batch_tokens)
            end = time.perf_counter()
            return max(0.0, end - start)
    except Exception:
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
                        <div class="kpi-label">Step 1 of 4</div>
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
                        <div class="kpi-label">Step 2 of 4</div>
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
            # Create capitalized display names for chunking strategies
            strategy_options = {
                "hybrid": "Hybrid (Recommended)",
                "hierarchical": "Hierarchical",
                "paragraph": "Paragraph",
                "fixed_size": "Fixed Size",
            }

            chunking_strategy_display = st.selectbox(
                "Chunking Strategy",
                options=list(strategy_options.values()),
                index=0,  # Default to "Hybrid"
                help="Hybrid: Advanced tokenization-aware chunking that preserves document structure and respects token limits.Best for fine-tuning.",
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
                    max_value=2048,
                    value=512,
                    step=64,
                    help="Maximum tokens per chunk (for Hybrid/Hierarchical strategies)",
                    key="dp_max_tokens",
                )

            with col_adv2:
                merge_peers = st.checkbox(
                    "Merge Small Chunks",
                    value=True,
                    help="Merge undersized chunks with same headings (Hybrid strategy only)",
                    key="dp_merge_peers",
                )

            st.markdown("**Legacy Chunking Parameters** (for Paragraph/Fixed Size strategies)")

            col_leg1, col_leg2 = st.columns(2)

            with col_leg1:
                chunk_size = st.slider(
                    "Chunk Size (words)",
                    min_value=128,
                    max_value=4096,
                    value=512,
                    step=128,
                    help="Size of chunks in words (Fixed Size strategy only)",
                    key="dp_chunk_size",
                )

            with col_leg2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (words)",
                    min_value=0,
                    max_value=512,
                    value=50,
                    step=10,
                    help="Overlap between chunks in words (Fixed Size strategy only)",
                    key="dp_chunk_overlap",
                )

            st.markdown("**Document Processing**")
            extract_tables = st.checkbox("Extract Tables", value=True, help="Extract and format tables from documents", key="dp_extract_tables")

        dataset_name = st.text_input(
            "Dataset Name",
            help="Name for the generated dataset",
            key="dp_dataset_name",
        )

        dataset_description = st.text_area(
            "Dataset Description (optional)",
            value="",
            help="Optional description of the dataset",
            key="dp_dataset_description",
            height=100,
        )

        # Store configuration in session state
        if uploaded_files:
            st.session_state.dp_uploaded_files = uploaded_files
            st.session_state.dp_config = {
                "chunking_strategy": chunking_strategy,
                "extract_tables": extract_tables,
                "ocr_enabled": ocr_enabled,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_tokens": max_tokens,
                "merge_peers": merge_peers,
                "dataset_name": dataset_name,
                "dataset_description": dataset_description,
            }

        # Create a placeholder container for processing status
        # This reserves space and prevents UI misalignment during processing
        st.markdown("---")
        st.session_state.dp_status_container = st.empty()

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute backend logic for data preparation."""
        self.logger.info("Executing data preparation step")

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
            processing_config = DataProcessingConfig(
                input_files=saved_files,
                output_dir=str(processed_dir),
                extract_tables=config.get("extract_tables", True),
                ocr_enabled=config.get("ocr_enabled", True),
                chunking_strategy=ChunkingStrategy(config.get("chunking_strategy", "hybrid")),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                max_tokens=config.get("max_tokens", 512),
                merge_peers=config.get("merge_peers", True),
            )

            # Get orchestrator and process documents
            from lite_llm_studio.app.app import get_orchestrator

            orchestrator = get_orchestrator()

            # Use status container if available, otherwise fallback to spinner
            status_container = st.session_state.get("dp_status_container")

            if status_container:
                # Clear the container and show processing status
                status_container.empty()

                # Step 1: Process documents
                with status_container.container():
                    st.info("Processing documents...")
                    job_result = orchestrator.execute_document_processing(processing_config)

                # Update to show success
                status_container.empty()
                with status_container.container():
                    st.success("Documents processed!")
            else:
                with st.spinner("Processing documents..."):
                    job_result = orchestrator.execute_document_processing(processing_config)

            if not job_result:
                raise Exception("Document processing failed")

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

            if status_container:
                # Step 2: Create dataset
                status_container.empty()
                with status_container.container():
                    st.info("Creating dataset...")
                    dataset_result = orchestrator.create_dataset(chunks_files, str(datasets_dir), dataset_name, dataset_description)

                # Update to show final success
                status_container.empty()
                with status_container.container():
                    st.success("Dataset created successfully!")
            else:
                with st.spinner("Creating dataset..."):
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
        uploaded_files = st.session_state.get("dp_uploaded_files", [])

        if not uploaded_files:
            return False, "Please upload at least one PDF file"

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
                        <div class="kpi-label">Step 3 of 4</div>
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
                    # Load from prepared dataset
                    text, total_bytes, desc = _load_text_from_dataset(dataset_path, max_samples)
                    if total_bytes == 0:
                        st.error(desc or "Falha ao carregar dataset preparado.")
                        return
                    dataset_desc = desc
                    
                else:
                    # Load from built-in dataset
                    text, total_bytes, desc = _load_builtin_text(
                        "wikitext-2" if builtin_name == "wikitext-2" else ("wikitext-103" if builtin_name == "wikitext-103" else "ag_news"),
                        target_bytes=int(target_mb * 1_000_000),
                    )
                    if total_bytes == 0:
                        st.error(desc or "Falha ao carregar dataset.")
                        return
                    dataset_desc = f"{desc} ~{target_mb} MB"

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
                batch_tokens, seq_len = _tokenize_with_truncation(tok, text, device_info, max_len)
                truncated_note = " (truncado)" if seq_len == max_len else ""

                progress.progress(80)
                status.info("Executando forward…")
                proc_s = _measure_forward(model, batch_tokens, device_info)

                progress.progress(90)
                status.info("Coletando memória pós-inferência…")
                mem_after_infer = _measure_memory_usage(device_info)

                progress.progress(100)
                status.success("Dry Run concluído.")
            finally:
                time.sleep(0.2)

            # Determine actual device robustly
            actual_device = _infer_actual_device(model, batch_tokens, device_info)

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
        """ """
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 4 of 4</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TODO: Implement training UI

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """ """
        # TODO: Implement backend logic for training
        self.logger.info("Executing training step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """ """
        # TODO: Add proper validation logic
        return True, "True"


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
]


class TrainingPipeline:
    """Manages the training pipeline state and execution."""

    def __init__(self):
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

    # Render the step-specific UI
    step.render_ui()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Action row: Complete / Back / Next
    left, mid, right = st.columns([1, 2, 1], vertical_alignment="center")

    # Check if currently processing
    is_processing = st.session_state.get("tp_processing", False)

    with left:
        # Disable Back button if at first step or processing
        if st.button("🡠 Back", key="tp_nav_back", disabled=(i == 0 or is_processing), use_container_width=True):
            st.session_state.tp_current_step = max(0, i - 1)
            st.rerun()

    with mid:
        is_completed = st.session_state.tp_completed[i]

        # Check if step can be completed
        can_complete, validation_msg = step.can_complete(st.session_state.tp_context)

        completed_label = "✓ Completed" if is_completed else "Complete Step"
        # Disable Complete button if already completed, can't complete, or processing
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
            # Set processing state immediately and rerun to disable buttons
            st.session_state.tp_processing = True
            st.rerun()

        # Execute processing if state is set (on next render)
        if is_processing and not is_completed:
            # Execute step backend logic
            try:
                result = pipeline.execute_step(i, st.session_state.tp_context)
                # Update context with step results
                st.session_state.tp_context.update(result)
                st.session_state.tp_completed[i] = True

                # Unlock next step if any
                if i + 1 < pipeline.get_step_count():
                    st.session_state.tp_unlocked[i + 1] = True
                    st.session_state.tp_current_step = i + 1

                st.success(f"{step.config.label} completed successfully!")
            except Exception as e:
                st.error(f"Error completing step: {str(e)}")
            finally:
                # Reset processing state
                st.session_state.tp_processing = False
                st.rerun()

    with right:
        can_go_next = (i < pipeline.get_step_count() - 1) and st.session_state.tp_completed[i] and st.session_state.tp_unlocked[i + 1]
        # Disable Next button if can't go next or processing
        if st.button("Next 🡢", key="tp_nav_next", disabled=(not can_go_next or is_processing), use_container_width=True):
            st.session_state.tp_current_step = min(pipeline.get_step_count() - 1, i + 1)
            st.rerun()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Finalization
    if i == pipeline.get_step_count() - 1:
        finished = all(st.session_state.tp_completed)
        if st.button("Finish Pipeline", key="tp_finish", disabled=not finished, use_container_width=True):
            try:
                # Save final results
                final_result = st.session_state.tp_context.get("trained_model")
                if final_result:
                    st.session_state.tp_result_model = final_result
                    st.success("Pipeline completed successfully! The trained model is now available.")
                    # TODO: Implement model persistence and indexing
                else:
                    st.warning("Pipeline completed but no trained model was found in context.")
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
