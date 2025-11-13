"""
Module app.app
--------------

Main entry point for the Streamlit application.

This module wires up the UI chrome (top/bottom bars, sidebar navigation),
sets up logging and app directories, and routes to the appropriate pages
(Home, Hardware, Training). It also exposes cached helpers for creating the
`Orchestrator` and running a hardware scan.
"""

import logging
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
from lite_llm_studio.app.utils import get_app_logger, setup_app_logging
from lite_llm_studio.core.orchestration import Orchestrator

# ------------------------------
# Application Configuration
# ------------------------------
st.set_page_config(page_title="LiteLLM Studio", layout="wide", initial_sidebar_state="expanded")

# Load fonts and styles
load_fonts_and_styles()

# Setup logging for Streamlit app
try:
    setup_app_logging()
    logger = get_app_logger("app.streamlit")
    logger.info("Streamlit application started")
except Exception as e:
    print(f"Warning: Could not setup application logging: {e}")
    logger = logging.getLogger(__name__)

# Setup application directories
try:
    from lite_llm_studio.core.configuration import setup_application_directories

    directories = setup_application_directories()
    logger.info(f"Application directories initialized: {list(directories.keys())}")
except Exception as e:
    logger.warning(f"Could not setup application directories: {e}")


# ------------------------------
# Session State Management
# ------------------------------
def init_session_state() -> None:
    """
    Initialize Streamlit session state with defaults.
    """
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
        logger.debug("Session state initialized with default page: Home")


# ------------------------------
# Data Loading Functions
# ------------------------------
@st.cache_resource
def get_orchestrator() -> Orchestrator:
    """
    Create (and cache) a single Orchestrator instance for the app lifetime.

    Returns:
        Orchestrator: A singleton-like orchestrator for the UI session.
    """
    logger.debug("Creating Orchestrator instance")
    return Orchestrator()


@st.cache_data(show_spinner=False)
def run_hardware_scan() -> dict[str, Any]:
    """
    Run a hardware scan via the orchestrator and cache the result.

    Returns:
        dict[str, Any]: Hardware report as a JSON-serializable dict.
                        Returns an empty dict on failure.
    """
    logger.info("Starting hardware scan")
    orchestrator = get_orchestrator()
    report_data = orchestrator.execute_hardware_scan()
    logger.info(f"Hardware scan completed, data keys: {list(report_data.keys()) if report_data else 'None'}")
    return report_data or {}


# ------------------------------
# Page Configuration
# ------------------------------
PAGE_TITLES: dict[str, str] = {
    "Home": "LiteLLM Studio",
    "Hardware": "Hardware Overview",
    "Recommendations": "Model Recommendations",
    "Dry Run": "Dry Run",
    "Training": "Model Training",
}


# ------------------------------
# Main Application Logic
# ------------------------------
def main() -> None:
    """
    Render the Streamlit application.

    Flow:
        1) Initialize session state.
        2) Render the sidebar navigation and determine the active page.
        3) Render the top bar with a dynamic title.
        4) Route to the appropriate page renderer:
           - Home: general landing content.
           - Hardware: run hardware scan and show results.
           - Training: model training workflow UI.
        5) Render the bottom bar.
    """
    # Initialize session state
    init_session_state()

    # Render sidebar navigation
    with st.sidebar:
        selected_page = create_sidebar_navigation()

    logger.info(f"Navigating to page: {selected_page}")

    # Render top bar with dynamic title
    render_top_bar(PAGE_TITLES.get(selected_page, selected_page))

    # Route to appropriate page
    try:
        if selected_page == "Home":
            logger.debug("Rendering Home page")
            render_home_page()

        elif selected_page == "Hardware":
            logger.debug("Rendering Hardware page")
            # Load hardware data and render hardware page
            with st.spinner("Analyzing system configuration..."):
                hardware_data = run_hardware_scan()
            render_hardware_page(hardware_data)

        elif selected_page == "Training":
            logger.debug("Rendering Training page")
            render_training_page()
        else:
            logger.warning(f"Unknown page requested: {selected_page}")

    except Exception as e:
        logger.error(f"Error rendering page {selected_page}: {e}", exc_info=True)
        st.error("An error occurred while loading the page. Please try again.")

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
