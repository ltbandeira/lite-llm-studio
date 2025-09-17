"""
Module app.modules.training_page
--------------------------------

This module contains the model training page content.
"""

import logging
import streamlit as st

# Get logger for training page
logger = logging.getLogger("app.pages.training")


def render_training_page():
    logger.info("Rendering training page")
    st.markdown("---")
