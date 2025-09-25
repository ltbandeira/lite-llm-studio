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
# Main Application Logic
# ------------------------------
def main():
    # Initialize session state
    init_session_state()

    # Render sidebar navigation
    with st.sidebar:
        selected_page = create_sidebar_navigation()

    # Render top bar with dynamic title
    render_top_bar(PAGE_TITLES.get(selected_page, selected_page))

    # Route to appropriate page
    if selected_page == "Home":
        render_home_page()

    elif selected_page == "Hardware":
        # Load hardware data and render hardware page
        with st.spinner("Analyzing system configuration..."):
            hardware_data = run_hardware_scan()
        render_hardware_page(hardware_data)

    elif selected_page == "Training":
        render_training_page()

    elif selected_page == "Recommendations":
        # Reuse hardware scan used by Hardware page
        with st.spinner("Analisando hardware…"):
            hardware_data = run_hardware_scan()
        render_recommendations_page(hardware_data)

    elif selected_page == "Dry Run":
        render_dry_run_page()

    # Render bottom bar
    render_bottom_bar()


# ------------------------------
# Application Entry Point
# ------------------------------
if __name__ == "__main__":
    main()
