"""
Module app.modules.training_page
--------------------------------

This module contains the model training page content.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import streamlit as st

from ..icons import ICONS

# Get logger for training page
logger = logging.getLogger("app.pages.training")


@dataclass
class PipelineStepConfig:
    """Configuration for a pipeline step."""

    key: str
    label: str
    icon: str
    description: str


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(self, config: PipelineStepConfig):
        self.config = config
        self.logger = logging.getLogger(f"app.pipeline.{config.key}")

    @abstractmethod
    def render_ui(self) -> None:
        """Render the UI for this pipeline step."""
        pass

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the backend logic for this step."""
        pass

    @abstractmethod
    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this step can be executed with the given context."""
        pass

    def can_complete(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if this step can be marked as complete."""
        return self.validate(context)


class ModelRecommendationStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """ """
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 1 of 4</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TODO: Implement model recommendation UI

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """ """
        # TODO: Implement backend logic for model recommendation
        self.logger.info("Executing model recommendation step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """ """
        # TODO: Add proper validation logic
        return True, "True"


class DataPreparationStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """ """
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 2 of 4</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TODO: Implement data preparation UI

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """ """
        # TODO: Implement backend logic for data preparation
        self.logger.info("Executing data preparation step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """ """
        # TODO: Add proper validation logic
        return True, "True"


class DryRunStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """Render UI for dry run step."""
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 3 of 4</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TODO: Implement dry run UI

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """ """
        # TODO: Implement backend logic for dry run
        self.logger.info("Executing dry run step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """ """
        # TODO: Add proper validation logic
        return True, "True"


class TrainingStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """ """
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card" style="grid-column: span 4;">
                    <div class="kpi-icon">{self.config.icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">Step 4 of 4</div>
                        <div class="kpi-value">{self.config.label}</div>
                        <div class="kpi-help">{self.config.description}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # TODO: Implement training UI

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """ """
        # TODO: Implement backend logic for training
        self.logger.info("Executing training step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """ """
        # TODO: Add proper validation logic
        return True, "True"


# Pipeline registry
PIPELINE_REGISTRY: list[tuple[PipelineStepConfig, type[PipelineStep]]] = [
    (
        PipelineStepConfig(
            key="model_reco",
            label="Model Recommendation",
            icon=ICONS.get("model", ""),
            description="Select and configure the base model for fine-tuning based on your requirements.",
        ),
        ModelRecommendationStep,
    ),
    (
        PipelineStepConfig(
            key="data_prep",
            label="Data Preparation",
            icon=ICONS.get("folder", ""),
            description="Upload, validate, and preprocess your training data for optimal results.",
        ),
        DataPreparationStep,
    ),
    (
        PipelineStepConfig(
            key="dry_run",
            label="Dry Run",
            icon=ICONS.get("refresh", ""),
            description="Validate your configuration and estimate training time and resources.",
        ),
        DryRunStep,
    ),
    (
        PipelineStepConfig(
            key="training", label="Training", icon=ICONS.get("play", ""), description="Execute the fine-tuning process and monitor training progress."
        ),
        TrainingStep,
    ),
]


class TrainingPipeline:
    """Manages the training pipeline state and execution."""

    def __init__(self):
        self.steps: list[PipelineStep] = []
        self.configs: list[PipelineStepConfig] = []

        # Initialize pipeline steps from registry
        for config, step_class in PIPELINE_REGISTRY:
            self.configs.append(config)
            self.steps.append(step_class(config))

        self.logger = logging.getLogger("app.pipeline.manager")

    def get_step_count(self) -> int:
        """Get total number of steps in the pipeline."""
        return len(self.steps)

    def get_step(self, index: int) -> PipelineStep:
        """Get pipeline step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        raise IndexError(f"Step index {index} out of range")

    def get_config(self, index: int) -> PipelineStepConfig:
        """Get pipeline step config by index."""
        if 0 <= index < len(self.configs):
            return self.configs[index]
        raise IndexError(f"Step index {index} out of range")

    def execute_step(self, index: int, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a specific pipeline step."""
        step = self.get_step(index)
        self.logger.info(f"Executing step {index}: {step.config.label}")
        return step.execute(context)

    def validate_step(self, index: int, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate a specific pipeline step."""
        step = self.get_step(index)
        return step.validate(context)


def _get_pipeline() -> TrainingPipeline:
    """Get or create the training pipeline instance."""
    if "tp_pipeline" not in st.session_state:
        st.session_state.tp_pipeline = TrainingPipeline()
    return st.session_state.tp_pipeline


def _init_pipeline_state() -> None:
    """Initialize pipeline state in session."""
    pipeline = _get_pipeline()

    if "tp_current_step" not in st.session_state:
        st.session_state.tp_current_step = 0
    if "tp_completed" not in st.session_state:
        st.session_state.tp_completed = [False] * pipeline.get_step_count()
    if "tp_unlocked" not in st.session_state:
        # Only the first step is unlocked initially
        st.session_state.tp_unlocked = [i == 0 for i in range(pipeline.get_step_count())]
    if "tp_context" not in st.session_state:
        st.session_state.tp_context = {}  # Shared context between steps
    if "tp_result_model" not in st.session_state:
        st.session_state.tp_result_model = None


def _render_stepper_header() -> None:
    """Render the visual stepper header showing pipeline progress."""
    pipeline = _get_pipeline()
    current = st.session_state.tp_current_step
    completed = st.session_state.tp_completed

    # Inline CSS just for the stepper
    st.markdown(
        """
        <style>
        .tp-stepper { display:flex; align-items:center; gap:14px; margin: 10px 0 24px; }
        .tp-step { display:flex; align-items:center; gap:10px; }
        .tp-node {
            width: 34px; height: 34px; border-radius: 999px;
            display:flex; align-items:center; justify-content:center;
            font-weight: 800; font-size: .95rem;
            border: 2px solid var(--border);
            background: var(--panel); color: var(--muted);
            box-shadow: var(--shadow-1);
        }
        .tp-node.active { border-color: var(--primary); color: var(--primary); background: var(--primary-weak); }
        .tp-node.done { border-color: #10b981; color: #10b981; background: color-mix(in oklab, #10b981 12%, white); }
        .tp-label { font-weight: 700; font-size: .92rem; color: var(--text); white-space: nowrap; }
        .tp-label.locked { color: #9ca3af; }
        .tp-connector { flex:1; height: 2px; background: var(--border); }
        .tp-connector.active { background: var(--primary); }
        .tp-connector.done { background: #10b981; }
        .tp-icon { width:16px; height:16px; color: currentColor; }
        @media (max-width: 1000px) {
            .tp-label { display: none; } /* keep it compact on small widths */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build the visual stepper
    html = ['<div class="tp-stepper">']
    for i in range(pipeline.get_step_count()):
        config = pipeline.get_config(i)
        status = "locked"
        if completed[i]:
            status = "done"
        elif i == current:
            status = "active"

        node_class = f"tp-node {'active' if status=='active' else ''} {'done' if status=='done' else ''}".strip()
        label_class = f"tp-label {'locked' if status=='locked' else ''}".strip()

        icon_html = config.icon.replace("<svg", '<svg class="tp-icon"')
        html.append(
            f"""
            <div class="tp-step">
              <div class="{node_class}">{i+1}</div>
              <div class="{label_class}">{icon_html} <span style="margin-left:6px;">{config.label}</span></div>
            </div>
            """
        )
        if i < pipeline.get_step_count() - 1:
            # connector between steps
            conn_cls = "tp-connector"
            if completed[i]:
                conn_cls += " done"
            elif i < current:
                conn_cls += " active"
            html.append(f'<div class="{conn_cls}"></div>')
    html.append("</div>")

    st.markdown("".join(html), unsafe_allow_html=True)


def _render_step_selector() -> None:
    """Render the row of step buttons with locked/active states."""
    pipeline = _get_pipeline()
    cols = st.columns(pipeline.get_step_count(), gap="small")

    for i in range(pipeline.get_step_count()):
        config = pipeline.get_config(i)
        with cols[i]:
            unlocked = bool(st.session_state.tp_unlocked[i])
            is_active = i == st.session_state.tp_current_step
            btn_label = f"{i+1}. {config.label}"
            if st.button(btn_label, key=f"tp_btn_{i}", use_container_width=True, disabled=not unlocked, type="primary" if is_active else "secondary"):
                st.session_state.tp_current_step = i
                st.rerun()


def _render_step_panel() -> None:
    """Render the current step panel with its specific UI and controls."""
    pipeline = _get_pipeline()
    i = st.session_state.tp_current_step
    step = pipeline.get_step(i)

    # Render the step-specific UI
    step.render_ui()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Action row: Complete / Back / Next
    left, mid, right = st.columns([1, 2, 1], vertical_alignment="center")

    with left:
        if st.button("🡠 Back", key="tp_nav_back", disabled=(i == 0), use_container_width=True):
            st.session_state.tp_current_step = max(0, i - 1)
            st.rerun()

    with mid:
        is_completed = st.session_state.tp_completed[i]

        # Check if step can be completed
        can_complete, validation_msg = step.can_complete(st.session_state.tp_context)

        completed_label = "✓ Completed" if is_completed else "Complete Step"
        button_disabled = is_completed or not can_complete

        if st.button(
            completed_label,
            key=f"tp_complete_{i}",
            type="secondary" if is_completed else "primary",
            use_container_width=True,
            disabled=button_disabled,
            help=validation_msg if not can_complete else None,
        ):
            # Execute step backend logic
            try:
                result = pipeline.execute_step(i, st.session_state.tp_context)
                # Update context with step results
                st.session_state.tp_context.update(result)
                st.session_state.tp_completed[i] = True

                # Unlock next step if any
                if i + 1 < pipeline.get_step_count():
                    st.session_state.tp_unlocked[i + 1] = True
                    st.session_state.tp_current_step = i + 1

                st.success(f"{step.config.label} completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error completing step: {str(e)}")

    with right:
        can_go_next = (i < pipeline.get_step_count() - 1) and st.session_state.tp_completed[i] and st.session_state.tp_unlocked[i + 1]
        if st.button("Next 🡢", key="tp_nav_next", disabled=not can_go_next, use_container_width=True):
            st.session_state.tp_current_step = min(pipeline.get_step_count() - 1, i + 1)
            st.rerun()

    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)

    # Finalization
    if i == pipeline.get_step_count() - 1:
        finished = all(st.session_state.tp_completed)
        if st.button("Finish Pipeline", key="tp_finish", disabled=not finished, use_container_width=True):
            try:
                # Save final results
                final_result = st.session_state.tp_context.get("trained_model")
                if final_result:
                    st.session_state.tp_result_model = final_result
                    st.success("Pipeline completed successfully! The trained model is now available.")
                    # TODO: Implement model persistence and indexing
                else:
                    st.warning("Pipeline completed but no trained model was found in context.")
            except Exception as e:
                st.error(f"Error finishing pipeline: {str(e)}")


def _render_pipeline_summary() -> None:
    """Render a summary of the pipeline progress."""
    pipeline = _get_pipeline()
    completed_count = sum(st.session_state.tp_completed)
    total_count = pipeline.get_step_count()

    progress = completed_count / total_count if total_count > 0 else 0

    st.progress(progress, text=f"Pipeline Progress: {completed_count}/{total_count} steps completed")


def render_training_page() -> None:
    """Main function to render the training page with scalable pipeline architecture."""
    logger.info("Rendering training page")

    # Initialize pipeline state
    _init_pipeline_state()

    # Render main UI components
    _render_stepper_header()
    _render_pipeline_summary()
    _render_step_selector()
    st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
    _render_step_panel()
