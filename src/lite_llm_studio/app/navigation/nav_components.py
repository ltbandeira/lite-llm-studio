"""
Module app.navigation.nav_components
------------------------------------

This module contains navigation-related components and utilities.
"""

from datetime import datetime

import streamlit as st


def load_sidebar_logo_b64() -> str:
    import base64

    try:
        from importlib import resources

        with resources.files("lite_llm_studio.app.resources").joinpath("lateral_bar_icon.png").open("rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def create_sidebar_navigation() -> str:
    # Load logo
    b64 = load_sidebar_logo_b64()
    if b64:
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{b64}" alt="LiteLLM Studio" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div style="font-size: 2.5rem; margin-bottom: .5rem; color: #ffffff;">LS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Navigation menu
    menu_options = {"Home": "Home", "Hardware": "Hardware", "Recommendations": "Recommendations", "Dry Run": "Dry Run", "Training": "Training"}

    for display_name, page_name in menu_options.items():
        if st.sidebar.button(
            display_name,
            key=f"btn_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.current_page == page_name else "secondary",
        ):
            st.session_state.current_page = page_name
            st.rerun()

    return st.session_state.current_page


def create_pipeline_navigation(stages, completed_stages=None):
    if completed_stages is None:
        completed_stages = set()
    st.markdown("<div class='nav-caption'>Pipeline</div>", unsafe_allow_html=True)
    current_stage = st.session_state.get("current_stage", stages[0])
    stage_idx = stages.index(current_stage)
    for idx, stage in enumerate(stages):
        is_completed = stage in completed_stages
        is_current = idx == stage_idx
        disabled = idx > stage_idx
        btn_label = f"{idx+1}. {stage}"
        btn_type = "primary" if is_current else ("secondary" if not is_completed else "success")
        if st.sidebar.button(
            btn_label,
            key=f"pipeline_{stage}",
            use_container_width=True,
            type=btn_type,
            disabled=disabled,
        ):
            st.session_state.current_stage = stage
            st.session_state.current_page = stage
            st.rerun()
    # Visual indicator
    st.markdown(f"<div style='margin-top:1rem; color:#aaa;'>Etapa atual: <b>{current_stage}</b></div>", unsafe_allow_html=True)
    return st.session_state.get("current_stage", stages[0])


def render_top_bar(title: str):
    st.markdown(
        f"""
        <div class="app-topbar">
            <div class="title">{title}</div>
            <div class="actions">
                <!-- Actions can be added here if needed -->
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bottom_bar():
    year = datetime.now().year
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="left">{year} © LiteLLM Studio</div>
            <div class="right">
                <a href="#" target="_blank" rel="noreferrer">About</a>
                <a href="#" target="_blank" rel="noreferrer">Contact</a>
                <a href="#" target="_blank" rel="noreferrer">Docs</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
