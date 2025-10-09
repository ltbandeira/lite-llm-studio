"""
Module app.components.ui_components
-----------------------------------

This module contains functions to generate HTML for various UI components.
"""

import logging
from typing import Any

from ..icons import ICONS

# Get logger for UI components
logger = logging.getLogger("app.components")


def format_gb(value: Any) -> str:
    try:
        gb_value = float(value or 0)
    except (ValueError, TypeError):
        return "—"

    if gb_value >= 1024:
        return f"{gb_value/1024:.1f} TB"
    else:
        return f"{gb_value:.0f} GB"


def create_kpi_cards_html(os_info: dict[str, Any], cpu_info: dict[str, Any], mem_info: dict[str, Any], gpus: list[dict[str, Any]]) -> str:
    logger.debug("Creating KPI cards HTML")
    total_mem = float(mem_info.get("total_memory") or 0)
    free_mem = float(mem_info.get("free_memory") or 0)
    mem_used_pct = ((total_mem - free_mem) / total_mem * 100) if total_mem else None
    first_gpu_name = (gpus[0] or {}).get("name") if gpus else "—"

    return f"""
    <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-icon">{ICONS['system']}</div>
      <div class="kpi-body">
      <div class="kpi-label">System</div>
      <div class="kpi-value">{os_info.get('system','—')}</div>
      <div class="kpi-help">{os_info.get('version','')}</div>
      </div>
    </div>

    <div class="kpi-card">
      <div class="kpi-icon">{ICONS['cpu']}</div>
      <div class="kpi-body">
      <div class="kpi-label">CPU</div>
      <div class="kpi-value">{cpu_info.get('brand','—')}</div>
      <div class="kpi-help">Cores: {cpu_info.get('cores','—')} · Threads: {cpu_info.get('threads','—')}</div>
      </div>
    </div>

    <div class="kpi-card">
      <div class="kpi-icon">{ICONS['ram']}</div>
      <div class="kpi-body">
      <div class="kpi-label">RAM</div>
      <div class="kpi-value">{mem_info.get('total_memory','—')} GB</div>
      <div class="kpi-help">{'' if mem_used_pct is None else f'Usage: {mem_used_pct:.0f}% · Free: {free_mem:.0f} GB'}</div>
      </div>
    </div>

    <div class="kpi-card">
      <div class="kpi-icon">{ICONS['gpu']}</div>
      <div class="kpi-body">
      <div class="kpi-label">GPUs</div>
      <div class="kpi-value">{len(gpus)}</div>
      <div class="kpi-help">{first_gpu_name}</div>
      </div>
    </div>
    </div>
    """


def create_system_cpu_card_html(os_info: dict[str, Any], cpu_info: dict[str, Any]) -> str:
    logger.debug("Creating system/CPU card HTML")
    return f"""
    <div class="syscpu-card">
    <div class="syscpu-header">
    <div class="syscpu-left">
    <div class="syscpu-icon">{ICONS["cpu"]}</div>
    <div>
    <div class="syscpu-title">System &amp; CPU</div>
    <div class="syscpu-sub">{cpu_info.get("brand","—")}</div>
    </div>
    </div>
    <div class="syscpu-tags">
    <span class="chip sm">{os_info.get("system","—")}</span>
    <span class="chip sm">{os_info.get("version","—")}</span>
    <span class="chip sm">{os_info.get("arch","—")}</span>
    </div>
    </div>
    <div class="spec-grid">
    <div class="spec-row">
        <div class="spec-label">CPU Model</div>
        <div class="spec-value" title="{cpu_info.get("brand","—")}">{cpu_info.get("brand","—")}</div>
    </div>
    <div class="spec-row">
        <div class="spec-label">Cores / Threads</div>
        <div class="spec-value">{cpu_info.get("cores","—")} / {cpu_info.get("threads","—")}</div>
    </div>
    <div class="spec-row">
        <div class="spec-label">Base Frequency</div>
        <div class="spec-value">{cpu_info.get("frequency","—")} GHz</div>
    </div>
    <div class="spec-row">
        <div class="spec-label">Architecture</div>
        <div class="spec-value">{os_info.get("arch","—")}</div>
    </div>
    </div>
    """


def _get_gpu_brand_class(gpu_name: str) -> str:
    name_lower = (gpu_name or "").lower()
    if "nvidia" in name_lower:
        return "brand-nvidia"
    elif "amd" in name_lower or "radeon" in name_lower:
        return "brand-amd"
    elif "intel" in name_lower:
        return "brand-intel"
    return ""


def create_gpu_cards_html(gpus: list[dict[str, Any]]) -> str:
    logger.debug(f"Creating GPU cards HTML for {len(gpus)} GPU(s)")
    if not gpus:
        return ""

    items_html = '<div class="gpu-list">'

    for idx, gpu in enumerate(gpus):
        name = gpu.get("name", "GPU")
        vram = gpu.get("total_vram")
        driver = gpu.get("driver")

        cuda_chip = (
            '<span class="chip ok" title="CUDA disponível">CUDA</span>'
            if gpu.get("cuda")
            else '<span class="chip" title="Sem suporte CUDA">No CUDA</span>'
        )

        # VRAM usage visualization
        used_vram = gpu.get("used_vram")
        if isinstance(used_vram, int | float) and isinstance(vram, int | float) and vram:
            pct = max(0, min(100, (used_vram / float(vram)) * 100))
            vram_html = (
                f'<div class="gpu-vram">'
                f'<div class="gpu-vrambar"><span style="width:{pct:.0f}%"></span></div>'
                f'<div class="gpu-vramtxt">{used_vram:.1f} / {float(vram):.1f} GB</div>'
                f"</div>"
            )
        else:
            v = "—" if vram in (None, "") else f"{vram} GB"
            vram_html = f'<span class="chip" title="Memória de vídeo total">VRAM: {v}</span>'

        brand_cls = _get_gpu_brand_class(name)

        items_html += (
            f'<div class="gpu-item {brand_cls}">'
            f'<div class="gpu-row">'
            f'<div class="gpu-left">'
            f'<div class="gpu-ico">{ICONS["gpu"]}</div>'
            f"<div>"
            f'<div class="gpu-title">GPU #{idx+1}: {name}</div>'
            f"</div>"
            f"</div>"
            f'<div class="gpu-right">'
            f"{cuda_chip}"
            f"{vram_html}"
            f'<span class="chip" title="Versão do driver">Driver: {driver or "—"}</span>'
            f"</div>"
            f"</div>"
            f"</div>"
        )

    items_html += "</div>"
    return items_html


def create_storage_card_html(disks: list[dict[str, Any]]) -> str:
    logger.debug(f"Creating storage card HTML for {len(disks)} disk(s)")
    if not disks:
        return ""

    total_all = sum(float(d.get("total_space") or 0) for d in disks)
    used_all = sum(float(d.get("used_space") or 0) for d in disks)
    free_all = max(0.0, total_all - used_all)
    pct_all = (used_all / total_all * 100) if total_all else 0.0

    html = (
        '<div class="storage-card">'
        '  <div class="storage-header">'
        '    <div class="storage-left">'
        f'      <div class="storage-ico">{ICONS["drive"]}</div>'
        "      <div>"
        '        <div class="storage-title">Disks &amp; Volumes</div>'
        f'        <div class="storage-sub">Total {format_gb(total_all)} \
                    • Used {format_gb(used_all)} ({pct_all:.0f}%) • Free {format_gb(free_all)}</div>'
        "      </div>"
        "    </div>"
        '    <div class="storage-tags">'
        f'      <span class="chip">Total: {format_gb(total_all)}</span>'
        f'      <span class="chip">Free: {format_gb(free_all)}</span>'
        "    </div>"
        "  </div>"
        '  <div class="storage-list">'
    )

    for disk in disks:
        name = disk.get("name", "Disk")
        total = float(disk.get("total_space") or 0)
        used = float(disk.get("used_space") or 0)
        pct = (used / total * 100) if total else 0.0

        if pct < 70:
            cls, label = "ok", "Healthy"
        elif pct < 90:
            cls, label = "warn", "Warning"
        else:
            cls, label = "bad", "Critical"

        bar_class = f"storage-bar {'warn' if cls=='warn' else ('bad' if cls=='bad' else '')}"

        html += (
            '<div class="storage-row">'
            f'  <div class="storage-name">{name}</div>'
            f'  <div class="{bar_class}" title="{format_gb(used)} / {format_gb(total)}">'
            f'    <span style="width:{pct:.0f}%"></span>'
            "  </div>"
            '  <div class="storage-badges">'
            f'    <span class="chip {"ok" if cls=="ok" else ("warn" if cls=="warn" else "bad")}">{label}</span>'
            f'    <span class="chip">{format_gb(used)} / {format_gb(total)} ({pct:.0f}%)</span>'
            "  </div>"
            "</div>"
        )

    html += "  </div></div>"
    return html


def create_directory_cards_html(
    dir_path: str,
    model_count: int,
    last_indexed: str | None = None,
    path_exists: bool | None = None,
) -> str:
    logger.debug(f"Creating directory cards HTML for path: {dir_path}")
    path_text = dir_path if dir_path else "Models directory"
    help_text = "Automatically managed models directory"

    if model_count > 0 and (path_exists is True):
        chip_cls = "ok"
        chip_text = "Ready"
        sub_text = f"Last indexed: {last_indexed}" if last_indexed else "Models ready to use"
    elif path_exists is False:
        chip_cls = "warn"
        chip_text = "Creating..."
        sub_text = "Directory will be created automatically"
    else:
        chip_cls = "warn"
        chip_text = "Empty"
        sub_text = "Train a new model and click 'Index Models'"

    # Truncate very long paths for display
    display_path = path_text
    if len(display_path) > 60:
        display_path = "..." + display_path[-57:]

    return f"""
    <div class="kpi-grid dir-summary-grid">
      <div class="kpi-card">
        <div class="kpi-icon">{ICONS['folder']}</div>
        <div class="kpi-body">
          <div class="kpi-label">Models Directory</div>
          <div class="kpi-value mono" title="{path_text}">{display_path}</div>
          <div class="kpi-help">{help_text}</div>
        </div>
      </div>

      <div class="kpi-card">
        <div class="kpi-icon">{ICONS['model']}</div>
        <div class="kpi-body">
          <div class="kpi-label">Status</div>
          <div class="kpi-value">{model_count} indexed <span class="chip {chip_cls} sm">{chip_text}</span></div>
          <div class="kpi-help">{sub_text}</div>
        </div>
      </div>
    </div>
    """
