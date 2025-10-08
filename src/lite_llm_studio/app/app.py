"""
Module app.app
--------------

This is the main entry point for the Streamlit application.
It orchestrates all the modular components and manages the application flow.
"""

from typing import Any

import streamlit as st

from lite_llm_studio.app.modules import (
    render_hardware_page,
    render_home_page,
    render_training_page,
    render_recommendations_page,
    render_dry_run_page,
)
from lite_llm_studio.app.navigation import create_sidebar_navigation, render_bottom_bar, render_top_bar

# Import modular components
from lite_llm_studio.app.styles import load_fonts_and_styles
from lite_llm_studio.core.orchestration import Orchestrator

# ------------------------------
# Application Configuration
# ------------------------------
st.set_page_config(page_title="LiteLLM Studio", layout="wide", initial_sidebar_state="expanded")

# Load fonts and styles
load_fonts_and_styles()


# ------------------------------
# Session State Management
# ------------------------------
def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"


# ------------------------------
# Data Loading Functions
# ------------------------------
@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_data(show_spinner=False)
def run_hardware_scan() -> dict[str, Any]:
    orchestrator = get_orchestrator()
    report_data = orchestrator.execute_hardware_scan()
    return report_data or {}


# ------------------------------
# Page Configuration
# ------------------------------
PAGE_TITLES = {
    "Home": "LiteLLM Studio",
    "Hardware": "Hardware Overview",
    "Recommendations": "Model Recommendations",
    "Dry Run": "Dry Run",
    "Training": "Model Training",
}


# ------------------------------
# Pipeline Configuration
# ------------------------------
PIPELINE_STAGES = [
    "Hardware",
    "Recommendations",
    "Dry Run",
    "Training",
]

# ------------------------------
# Função para renderizar o onboarding/pipeline no topo
# ------------------------------
def render_pipeline_onboarding(stages, current_stage, completed_stages):
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    cols = st.columns(len(stages))
    for idx, (col, stage) in enumerate(zip(cols, stages)):
        is_current = stage == current_stage
        is_completed = stage in completed_stages
        color = "#6366f1" if is_current else ("#22c55e" if is_completed else "#334155")
        border = "3px solid #6366f1" if is_current else ("2px solid #22c55e" if is_completed else "2px solid #334155")
        style = f"background:{color}20;border-radius:8px;padding:12px 0;margin:0 4px;border:{border};color:#fff;font-weight:700;text-align:center;"
        label = f"{idx+1}. {stage}"
        col.markdown(f"<div style='{style}'>{label}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ------------------------------
# Main Application Logic
# ------------------------------
def main():
    # Initialize session state
    init_session_state()
    if "completed_stages" not in st.session_state:
        st.session_state.completed_stages = set(["Hardware"])
    if "current_stage" not in st.session_state:
        st.session_state.current_stage = PIPELINE_STAGES[0]
    current_stage = st.session_state.current_stage
    completed_stages = st.session_state.completed_stages
    # Render top bar
    render_top_bar(PAGE_TITLES.get(current_stage, current_stage))
    # Render onboarding pipeline no topo
    render_pipeline_onboarding(PIPELINE_STAGES, current_stage, completed_stages)
    # Render etapa atual
    stage_idx = PIPELINE_STAGES.index(current_stage)
    def go_next():
        if stage_idx + 1 < len(PIPELINE_STAGES):
            st.session_state.current_stage = PIPELINE_STAGES[stage_idx + 1]
            st.rerun()
    if current_stage == "Hardware":
        with st.spinner("Analyzing system configuration..."):
            hardware_data = run_hardware_scan()
        render_hardware_page(hardware_data)
        st.session_state["hardware_data"] = hardware_data
        st.session_state.completed_stages.add("Hardware")
        if st.button("Próxima etapa: Recomendações", type="primary"):
            go_next()
    elif current_stage == "Recommendations":
        hardware_data = st.session_state.get("hardware_data")
        if not hardware_data:
            st.warning("Complete a etapa de Hardware primeiro.")
        else:
            render_recommendations_page(hardware_data)
            st.session_state.completed_stages.add("Recommendations")
            if st.button("Próxima etapa: Dry Run", type="primary"):
                go_next()
    elif current_stage == "Dry Run":
        render_dry_run_page()
        st.session_state.completed_stages.add("Dry Run")
        if st.button("Próxima etapa: Treinamento", type="primary"):
            go_next()
    elif current_stage == "Training":
        render_training_page()
        st.session_state.completed_stages.add("Training")
        st.success("Treinamento")
    # Render bottom bar
    render_bottom_bar()


# ------------------------------
# Application Entry Point
# ------------------------------
if __name__ == "__main__":
    main()
