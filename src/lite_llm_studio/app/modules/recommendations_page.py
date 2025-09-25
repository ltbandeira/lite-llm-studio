"""
Module app.modules.recommendations_page
--------------------------------------

This module renders a hardware-aware recommendations page for models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Bytes per parameter approximation for each quantization
BYTES_PER_PARAM: Dict[str, float] = {"fp16": 2.0, "int8": 1.0, "q4": 0.5, "q5": 0.625}

# Supported models catalog (minimal examples; can be extended)
MODELS: List[Dict[str, Any]] = [
    {"name": "Qwen2-0.5B", "params_b": 0.5, "repo_id": "Qwen/Qwen2-0.5B"},
    {"name": "TinyLlama-1.1B-Chat", "params_b": 1.1, "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    {"name": "Llama-3.1-70B", "params_b": 70.0, "repo_id": "meta-llama/Llama-3.1-70B"},
]


def estimate_model_memory(params_b: float, bytes_per_param: float) -> float:
    # Adds a 20% margin
    return round(params_b * bytes_per_param * 1.2, 2)


def summarize_hardware(hardware_data: Dict[str, Any]) -> Dict[str, Any]:
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


# Enumerate all model/quantization configurations

def enumerate_model_configs(hw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enumerate models across quantizations, evaluating fit on CPU RAM or GPU VRAM."""
    results: List[Dict[str, Any]] = []
    for model in MODELS:
        for precision, bpp in BYTES_PER_PARAM.items():
            mem_gb = estimate_model_memory(model["params_b"], bpp)
            if hw["has_gpu"] and hw["max_vram"] >= mem_gb:
                device, mem_type = "GPU", "VRAM"
            elif hw["ram_total"] >= mem_gb:
                device, mem_type = "CPU", "RAM"
            else:
                device, mem_type = "-", "-"
            results.append({
                **model,
                "precision": precision,
                "bytes_per_param": bpp,
                "device": device,
                "mem_type": mem_type,
                "mem_gb": mem_gb,
                "enough": device != "-",
            })
    return results


# Report helper (kept for future use / parity)

def generate_report(model: Dict[str, Any], hw: Dict[str, Any]) -> Dict[str, Any]:
    params_b = model["params_b"]
    checkpoints_gb = round(params_b * 2, 1)
    q4_mem = estimate_model_memory(params_b, BYTES_PER_PARAM["q4"]) 
    fp16_mem = estimate_model_memory(params_b, BYTES_PER_PARAM["fp16"]) 

    def recipe(name: str, precision: str, mem_gb: float, extra_disk_gb: float, seq_len: int, batch_size: int) -> Dict[str, Any]:
        fits_gpu = hw["has_gpu"] and hw["max_vram"] >= mem_gb
        return {
            "name": name,
            "precision": precision,
            "min_vram_gb": mem_gb if hw["has_gpu"] else None,
            "min_ram_gb": mem_gb,
            "min_disk_free_gb": checkpoints_gb + extra_disk_gb,
            "suggested_seq_len": seq_len,
            "suggested_batch_size": batch_size,
            "fits_gpu": fits_gpu,
        }

    return {
        "model_id": model.get("repo_id", model["name"]),
        "display_name": model["name"],
        "params_b": params_b,
        "checkpoints_gb": checkpoints_gb,
        "generated_at": datetime.now().isoformat(),
        "chosen_device": model.get("device"),
        "usage_mem_type": model.get("mem_type"),
        "estimated_usage_gb": model.get("mem_gb"),
        "recipes": [
            recipe("q4-quantized", "q4", q4_mem, 9, 4096, 1),
            recipe("normal", "fp16", fp16_mem, 24, 2048, 2 if hw["max_vram"] > 16 else 1),
        ],
    }


# Small helpers to render list blocks using existing CSS classes

def _render_config_list(configs: List[Dict[str, Any]], show_device: bool = True):
    if not configs:
        return

    rows_html = ''
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
            f'  <div class="storage-name" style="width303px;">{name}'
            f'    <div class="storage-sub" style="font-size:0.85rem; opacity:.85; white-space:normal; overflow:visible; text-overflow:unset;">{repo}</div>'
            f'  </div>'
            '  <div class="storage-badges">' + "".join(chips) + "</div>"
            "</div>"
        )
        if idx != last_index:
            rows_html += '<div style="height:1px; background: rgba(255,255,255,0.12); margin: 4px 0;"></div>'

    block_html = '<div class="storage-card"><div class="storage-list">' + rows_html + "</div></div>"
    st.markdown(block_html, unsafe_allow_html=True)


# Page renderer

def render_recommendations_page(hardware_data: Dict[str, Any]):
    if not hardware_data:
        st.error("Error retrieving system information.")
        return

    hw = summarize_hardware(hardware_data)

    # Build model configurations across quantizations
    configs = enumerate_model_configs(hw)
    df = pd.DataFrame(configs)
    if df.empty:
        st.warning("Sem dados de configuração.")
        return

    runnable = df[df["enough"]].copy().sort_values(by=["params_b", "bytes_per_param"], ascending=[False, True])
    not_runnable = df[~df["enough"]].copy().sort_values(by=["params_b", "bytes_per_param"], ascending=[False, True])

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
              <div class="spec-label">Memória estimada</div>
              <div class="spec-value">{best.get("mem_gb"):.2f} GB {best.get("mem_type")}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Simple download button using huggingface_hub
        repo_id = best.get("repo_id") or ""
        folder_name = repo_id.replace("/", "-")
        if st.button("Baixar modelo recomendado", type="primary"):
            with st.spinner("Baixando modelo do Hugging Face…"):
                try:
                    from huggingface_hub import snapshot_download

                    snapshot_download(repo_id, local_dir=f"models/{folder_name}", local_dir_use_symlinks=False)
                    st.success(f"Download concluído em models/{folder_name}")
                except ModuleNotFoundError:
                    st.error("Instale o pacote necessário: pip install -U huggingface_hub")
                except Exception as ex:
                    st.error(f"Falha no download: {ex}")
    else:
        st.warning("Nenhuma configuração atende aos requisitos de memória.")

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


