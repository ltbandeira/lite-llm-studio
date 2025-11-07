"""
Module app.modules.hardware_page
--------------------------------

This module contains the hardware monitoring page content with real-time monitoring.
"""

import logging
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import psutil
import streamlit as st

from ..components import create_gpu_cards_html, create_kpi_cards_html, create_storage_card_html, create_system_cpu_card_html

# Get logger for hardware page
logger = logging.getLogger("app.pages.hardware")

# Constants for real-time monitoring
REFRESH_INTERVAL = 3  # seconds
MAX_DATA_POINTS = 20  # Keep last 20 data points (1 minute of data)


def create_fixed_axis_chart(df: pd.DataFrame, x_col: str, y_col: str, y_min: float = 0, y_max: float = 100) -> alt.Chart:
    """Create a line chart with fixed Y-axis scale."""
    chart = alt.Chart(df).mark_line(
        color='#1f77b4',
        strokeWidth=2,
        point=alt.OverlayMarkDef(color='#1f77b4', size=30)
    ).encode(
        x=alt.X(x_col, axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y(y_col, scale=alt.Scale(domain=[y_min, y_max]), axis=alt.Axis(title=f'{y_col}')),
        tooltip=[x_col, alt.Tooltip(y_col, format='.1f')]
    ).properties(
        height=200
    ).configure_axis(
        gridColor='#444444',
        gridOpacity=0.3
    ).configure_view(
        strokeWidth=0
    )
    return chart


def get_realtime_metrics():
    """Get current system metrics for real-time monitoring."""
    metrics = {
        "timestamp": datetime.now(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / (1024**3),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    # Try to get GPU metrics (NVIDIA only)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            gpu_metrics = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpu_metrics.append({
                        "utilization": float(parts[0]),
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                        "memory_percent": (float(parts[1]) / float(parts[2]) * 100) if float(parts[2]) > 0 else 0
                    })
            metrics["gpus"] = gpu_metrics
    except Exception as e:
        logger.debug(f"GPU metrics not available: {e}")
        metrics["gpus"] = []
    
    return metrics


@st.fragment(run_every=REFRESH_INTERVAL)
def render_realtime_charts():
    """Render real-time monitoring charts that update every 3 seconds using fragments."""
    
    # Initialize session state for historical data (only once)
    if "monitoring_data" not in st.session_state:
        st.session_state.monitoring_data = {
            "timestamps": deque(maxlen=MAX_DATA_POINTS),
            "cpu": deque(maxlen=MAX_DATA_POINTS),
            "ram": deque(maxlen=MAX_DATA_POINTS),
            "gpu_util": deque(maxlen=MAX_DATA_POINTS),
            "gpu_mem": deque(maxlen=MAX_DATA_POINTS),
        }
    
    # Collect new metrics (decoupled from display)
    metrics = get_realtime_metrics()
    
    # Update historical data in session state
    st.session_state.monitoring_data["timestamps"].append(metrics["timestamp"].strftime("%H:%M:%S"))
    st.session_state.monitoring_data["cpu"].append(metrics["cpu_percent"])
    st.session_state.monitoring_data["ram"].append(metrics["ram_percent"])
    
    # Update GPU data if available
    has_gpu = bool(metrics["gpus"])
    if has_gpu:
        gpu_data = metrics["gpus"][0]
        st.session_state.monitoring_data["gpu_util"].append(gpu_data["utilization"])
        st.session_state.monitoring_data["gpu_mem"].append(gpu_data["memory_percent"])
    else:
        gpu_data = None
    
    # Display update timestamp
    st.markdown(
        f'<div style="color: #888; font-size: 0.9rem; margin-bottom: 16px;">'
        f'Last update: {metrics["timestamp"].strftime("%H:%M:%S")} • Auto-refresh every {REFRESH_INTERVAL}s</div>',
        unsafe_allow_html=True
    )
    
    # Prepare data for charts (only if we have data)
    if len(st.session_state.monitoring_data["timestamps"]) > 0:
        # CPU & RAM Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### RAM Usage - {metrics['ram_percent']:.1f}%")
            #st.caption(f"{metrics['ram_used_gb']:.2f} / {metrics['ram_total_gb']:.2f} GB")
            df_ram = pd.DataFrame({
                "Time": list(st.session_state.monitoring_data["timestamps"]),
                "RAM %": list(st.session_state.monitoring_data["ram"])
            })
            chart_ram = create_fixed_axis_chart(df_ram, "Time", "RAM %")
            st.altair_chart(chart_ram, use_container_width=True)
        
        with col2:
            st.markdown(f"#### CPU Usage - {metrics['cpu_percent']:.1f}%")
            df_cpu = pd.DataFrame({
                "Time": list(st.session_state.monitoring_data["timestamps"]),
                "CPU %": list(st.session_state.monitoring_data["cpu"])
            })
            chart_cpu = create_fixed_axis_chart(df_cpu, "Time", "CPU %")
            st.altair_chart(chart_cpu, use_container_width=True)
        
        # GPU Charts (only if GPU is available)
        if has_gpu:
            gpu_col1, gpu_col2 = st.columns(2)
            
            with gpu_col1:
                st.markdown(f"#### GPU Utilization - {gpu_data['utilization']:.1f}%")
                df_gpu_util = pd.DataFrame({
                    "Time": list(st.session_state.monitoring_data["timestamps"]),
                    "GPU %": list(st.session_state.monitoring_data["gpu_util"])
                })
                chart_gpu_util = create_fixed_axis_chart(df_gpu_util, "Time", "GPU %")
                st.altair_chart(chart_gpu_util, use_container_width=True)
            
            with gpu_col2:
                st.markdown(f"#### GPU Memory (VRAM) - {gpu_data['memory_percent']:.1f}%")
                #st.caption(f"{gpu_data['memory_used_mb'] / 1024:.2f} / {gpu_data['memory_total_mb'] / 1024:.2f} GB")
                df_gpu_mem = pd.DataFrame({
                    "Time": list(st.session_state.monitoring_data["timestamps"]),
                    "VRAM %": list(st.session_state.monitoring_data["gpu_mem"])
                })
                chart_gpu_mem = create_fixed_axis_chart(df_gpu_mem, "Time", "VRAM %")
                st.altair_chart(chart_gpu_mem, use_container_width=True)
                #st.line_chart(df_gpu_mem.set_index("Time"), height=200, use_container_width=True)
        else:
            st.info("💡 No NVIDIA GPU detected. GPU monitoring is only available for NVIDIA GPUs.")
    else:
        st.info("⏳ Collecting initial data...")


def render_hardware_page(hardware_data: dict[str, Any]):
    logger.info("Rendering hardware page")

    if not hardware_data:
        logger.error("No hardware data available")
        st.error("Error retrieving system information.")
        return

    # Extract data sections
    os_info = hardware_data.get("os") or {}
    cpu_info = hardware_data.get("cpu") or {}
    mem_info = hardware_data.get("memory") or {}
    gpus = hardware_data.get("gpus") or []
    disks = hardware_data.get("disks") or []

    logger.debug(f"Hardware data: OS={bool(os_info)}, CPU={bool(cpu_info)}, Memory={bool(mem_info)}, GPUs={len(gpus)}, Disks={len(disks)}")

    # Render KPI cards
    logger.debug("Rendering KPI cards")
    kpi_html = create_kpi_cards_html(os_info, cpu_info, mem_info, gpus)
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)

    # Real-time monitoring section (using fragment for auto-refresh)
    st.markdown('<div class="section-title">📊 Real-Time Monitoring</div>', unsafe_allow_html=True)
    render_realtime_charts()
    
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    # System and CPU information
    logger.debug("Rendering system and CPU information")
    st.markdown('<div class="section-title">System & CPU</div>', unsafe_allow_html=True)
    syscpu_html = create_system_cpu_card_html(os_info, cpu_info)
    st.markdown(syscpu_html, unsafe_allow_html=True)

    # GPU information
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GPUs</div>', unsafe_allow_html=True)

    if gpus:
        logger.info(f"Displaying {len(gpus)} GPU(s)")
        gpu_html = create_gpu_cards_html(gpus)
        st.markdown(gpu_html, unsafe_allow_html=True)
    else:
        logger.info("No dedicated GPUs detected")
        st.info("No dedicated GPUs detected.")

    # Storage information
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Storage</div>', unsafe_allow_html=True)

    if disks:
        storage_html = create_storage_card_html(disks)
        st.markdown(storage_html, unsafe_allow_html=True)
    else:
        st.info("No storage devices detected.")
