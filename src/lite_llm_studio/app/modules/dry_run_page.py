"""
Module app.modules.dry_run_page
------------------------------

Benchmarks a locally downloaded model to estimate TTFT, processing time for ~1MB,
and memory usage on this machine.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any, Dict, List, Tuple

import psutil
import streamlit as st


def _list_local_models(base_dir: str = "models") -> List[str]:
    try:
        if not os.path.isdir(base_dir):
            return []
        return [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    except Exception:
        return []


def _get_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"has_torch": False, "has_cuda": False, "device": "cpu", "cuda_name": None}
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
    gb = num_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = num_bytes / (1024 ** 2)
    return f"{mb:.0f} MB"


def _read_text_from_uploads(uploads: List[io.BytesIO]) -> Tuple[str, int]:
    total_bytes = 0
    parts: List[str] = []
    for uploaded in uploads:
        try:
            raw = uploaded.read()
            total_bytes += len(raw)
            parts.append(raw.decode("utf-8", errors="ignore"))
        except Exception:
            continue
    return ("\n".join(parts), total_bytes)


def _load_builtin_text(dataset_key: str, target_bytes: int = 1_000_000) -> Tuple[str, int, str]:
    """Load known text dataset and return concatenated text up to ~target_bytes.
    Returns (text, byte_len, description).
    """
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
        parts: List[str] = []
        total = 0
        text_field = "text" if "text" in ds.features else ("content" if "content" in ds.features else None)
        title_field = "title" if "title" in ds.features else None
        for ex in ds:
            segs: List[str] = []
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


def _load_model_tokenizer(model_dir: str, device_info: Dict[str, Any], force_gpu: bool = False):
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


def _measure_ttft(model, tokenizer, device_info: Dict[str, Any]) -> float:
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
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and 0 < max_len < 100000:
        return max_len
    cfg_len = getattr(getattr(model, "config", object()), "max_position_embeddings", None)
    if isinstance(cfg_len, int) and cfg_len > 0:
        return cfg_len
    return 2048


def _tokenize_with_truncation(tokenizer, text: str, device_info: Dict[str, Any], max_len: int):
    batch_tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    if device_info.get("has_cuda"):
        batch_tokens = {k: v.to("cuda") for k, v in batch_tokens.items()}
    seq_len = int(batch_tokens["input_ids"].shape[-1])
    return batch_tokens, seq_len


def _measure_forward(model, batch_tokens, device_info: Dict[str, Any]) -> float:
    try:
        import torch  # type: ignore

        with torch.no_grad():
            start = time.perf_counter()
            _ = model(**batch_tokens)
            end = time.perf_counter()
            return max(0.0, end - start)
    except Exception:
        return float("nan")


def _measure_memory_usage(device_info: Dict[str, Any]) -> Dict[str, Any]:
    mem: Dict[str, Any] = {}
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
    return (
        '<div class="spec-row">'
        f'  <div class="spec-label">{label}</div>'
        f'  <div class="spec-value">{value}</div>'
        "</div>"
    )


def _infer_actual_device(model, batch_tokens, device_info: Dict[str, Any]) -> str:
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


def render_dry_run_page():
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

    st.markdown('<div style="color:#ffffff; margin-top:8px; margin-bottom:4px;">Envie arquivo(s) de texto para o teste</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Envie arquivo(s) de texto para o teste",
        type=["txt", "log", "json", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    use_builtin = st.checkbox("Usar dataset conhecido (sem enviar arquivos)", value=not bool(uploads))

    precision_to_mb = {"Baixa": 1, "Média": 2, "Alta": 4}
    target_mb = precision_to_mb["Média"]
    builtin_name = None
    if use_builtin:
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
            if use_builtin:
                text, total_bytes, desc = _load_builtin_text(
                    "wikitext-2" if builtin_name == "wikitext-2" else ("wikitext-103" if builtin_name == "wikitext-103" else "ag_news"),
                    target_bytes=int(target_mb * 1_000_000),
                )
                if total_bytes == 0:
                    st.error(desc or "Falha ao carregar dataset.")
                    return
                dataset_desc = f"{desc} ~{target_mb} MB"
            else:
                if not uploads:
                    st.error("Envie arquivos ou selecione um dataset conhecido.")
                    return
                text, total_bytes = _read_text_from_uploads(uploads)
                dataset_desc = f"Uploads ({_bytes_to_mb_gb(total_bytes)})"

            if not device_info.get("has_torch"):
                st.error("PyTorch/Transformers não instalado. Instale: pip install torch transformers datasets")
                return

            progress.progress(15)
            status.info("Coletando memória inicial…")
            mem_before = _measure_memory_usage(device_info)

            progress.progress(30)
            status.info("Carregando modelo…")
            model_dir = os.path.join("models", selected)
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
