"""
Module app.modules.hardware_page
--------------------------------

This module contains the hardware monitoring page content.
"""

from typing import Any

import streamlit as st

from ..components import create_gpu_cards_html, create_kpi_cards_html, create_storage_card_html, create_system_cpu_card_html


def render_hardware_page(hardware_data: dict[str, Any]):
    if not hardware_data:
        st.error("Error retrieving system information.")
        return

    # Extract data sections
    os_info = hardware_data.get("os") or {}
    cpu_info = hardware_data.get("cpu") or {}
    mem_info = hardware_data.get("memory") or {}
    gpus = hardware_data.get("gpus") or []
    disks = hardware_data.get("disks") or []

    # Render KPI cards
    kpi_html = create_kpi_cards_html(os_info, cpu_info, mem_info, gpus)
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown('<div class="section"></div>', unsafe_allow_html=True)

    # System and CPU information
    st.markdown('<div class="section-title">System & CPU</div>', unsafe_allow_html=True)
    syscpu_html = create_system_cpu_card_html(os_info, cpu_info)
    st.markdown(syscpu_html, unsafe_allow_html=True)

    # GPU information
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GPUs</div>', unsafe_allow_html=True)

    if gpus:
        gpu_html = create_gpu_cards_html(gpus)
        st.markdown(gpu_html, unsafe_allow_html=True)
    else:
        st.info("No dedicated GPUs detected.")

    # Storage information
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Storage</div>', unsafe_allow_html=True)

    if disks:
        storage_html = create_storage_card_html(disks)
        st.markdown(storage_html, unsafe_allow_html=True)
    else:
        st.info("No storage devices detected.")
