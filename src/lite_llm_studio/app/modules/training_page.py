"""
Module app.modules.training_page
--------------------------------

This module contains the model training page content.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from lite_llm_studio.core.configuration.data_schema import ChunkingStrategy, DataProcessingConfig
from lite_llm_studio.core.configuration.desktop_app_config import get_default_models_directory, get_user_data_directory

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
        """Execute backend logic for model recommendation."""
        # For now, this step is automatically completed as model selection
        # happens in the training step. This could be expanded to include
        # model recommendation logic based on hardware capabilities.
        self.logger.info("Executing model recommendation step")
        return {}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if model recommendation step can be completed.

        For now, this step is always valid as it's informational.
        Can be expanded to check hardware requirements.
        """
        # This step is always valid for now
        return True, "Ready to proceed to data preparation"


class DataPreparationStep(PipelineStep):
    """Data preparation step with PDF upload and processing."""

    def render_ui(self) -> None:
        """Render UI for data preparation step."""
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

        st.markdown("---")

        # File upload section
        st.markdown("#### Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files with domain-specific content",
            key="dp_pdf_uploader",
        )

        if uploaded_files:
            # Display uploaded files
            with st.expander("View uploaded files", expanded=False):
                for idx, file in enumerate(uploaded_files, 1):
                    file_size_mb = file.size / (1024 * 1024)
                    st.write(f"{idx}. **{file.name}** ({file_size_mb:.2f} MB)")

        # Processing configuration
        st.markdown("#### Processing Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Create capitalized display names for chunking strategies
            strategy_options = {
                "hybrid": "Hybrid (Recommended)",
                "hierarchical": "Hierarchical",
                "paragraph": "Paragraph",
                "fixed_size": "Fixed Size",
            }

            chunking_strategy_display = st.selectbox(
                "Chunking Strategy",
                options=list(strategy_options.values()),
                index=0,  # Default to "Hybrid"
                help="Hybrid: Advanced tokenization-aware chunking that preserves document structure and respects token limits.Best for fine-tuning.",
                key="dp_chunking_strategy_display",
            )

            # Convert back to enum value
            chunking_strategy = [k for k, v in strategy_options.items() if v == chunking_strategy_display][0]

        with col2:
            # Add vertical spacing to align with the selectbox
            st.markdown('<div style="height: 34px;"></div>', unsafe_allow_html=True)
            ocr_enabled = st.checkbox("Enable OCR", value=True, help="Enable OCR for scanned documents", key="dp_ocr_enabled")

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            st.markdown("**Tokenization Settings**")

            col_adv1, col_adv2 = st.columns(2)

            with col_adv1:
                max_tokens = st.slider(
                    "Max Tokens per Chunk",
                    min_value=64,
                    max_value=1024,
                    value=512,
                    step=64,
                    help="Maximum tokens per chunk (for Hybrid/Hierarchical strategies)",
                    key="dp_max_tokens",
                )

            with col_adv2:
                merge_peers = st.checkbox(
                    "Merge Small Chunks",
                    value=True,
                    help="Merge undersized chunks with same headings (Hybrid strategy only)",
                    key="dp_merge_peers",
                )

            st.markdown("**Legacy Chunking Parameters** (for Paragraph/Fixed Size strategies)")

            col_leg1, col_leg2 = st.columns(2)

            with col_leg1:
                chunk_size = st.slider(
                    "Chunk Size (words)",
                    min_value=128,
                    max_value=4096,
                    value=512,
                    step=128,
                    help="Size of chunks in words (Fixed Size strategy only)",
                    key="dp_chunk_size",
                )

            with col_leg2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (words)",
                    min_value=0,
                    max_value=512,
                    value=50,
                    step=10,
                    help="Overlap between chunks in words (Fixed Size strategy only)",
                    key="dp_chunk_overlap",
                )

            st.markdown("**Document Processing**")
            extract_tables = st.checkbox("Extract Tables", value=True, help="Extract and format tables from documents", key="dp_extract_tables")

        dataset_name = st.text_input(
            "Dataset Name",
            help="Name for the generated dataset",
            key="dp_dataset_name",
        )

        dataset_description = st.text_area(
            "Dataset Description (optional)",
            value="",
            help="Optional description of the dataset",
            key="dp_dataset_description",
            height=100,
        )

        # Store configuration in session state
        if uploaded_files:
            st.session_state.dp_uploaded_files = uploaded_files
            st.session_state.dp_config = {
                "chunking_strategy": chunking_strategy,
                "extract_tables": extract_tables,
                "ocr_enabled": ocr_enabled,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "max_tokens": max_tokens,
                "merge_peers": merge_peers,
                "dataset_name": dataset_name,
                "dataset_description": dataset_description,
            }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute backend logic for data preparation."""
        self.logger.info("Executing data preparation step")

        try:
            # Get uploaded files from session state
            uploaded_files = st.session_state.get("dp_uploaded_files", [])
            config = st.session_state.get("dp_config", {})

            if not uploaded_files:
                raise ValueError("No files uploaded")

            # Create processing directory and save uploaded files directly there
            user_data_dir = get_user_data_directory()
            processed_dir = user_data_dir / "processed_documents"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded files directly in processed_documents directory
            saved_files: list[str] = []
            for file in uploaded_files:
                file_path = processed_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_files.append(str(file_path))
                self.logger.info(f"Saved uploaded file: {file.name}")

            # Format is always JSONL for causal language modeling
            processing_config = DataProcessingConfig(
                input_files=saved_files,
                output_dir=str(processed_dir),
                extract_tables=config.get("extract_tables", True),
                ocr_enabled=config.get("ocr_enabled", True),
                chunking_strategy=ChunkingStrategy(config.get("chunking_strategy", "hybrid")),
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                max_tokens=config.get("max_tokens", 512),
                merge_peers=config.get("merge_peers", True),
            )

            # Get orchestrator and process documents
            from lite_llm_studio.app.app import get_orchestrator

            orchestrator = get_orchestrator()

            # Process documents
            self.logger.info("Starting document processing...")
            job_result = orchestrator.execute_document_processing(processing_config)

            if not job_result:
                raise Exception("Document processing failed")

            self.logger.info("Document processing completed")

            # Collect chunk files
            chunks_files: list[str] = []
            for doc in job_result.get("processed_documents", []):
                chunks_file = doc.get("metadata", {}).get("chunks_file")
                if chunks_file:
                    chunks_files.append(chunks_file)

            if not chunks_files:
                raise Exception("No chunks were generated from the documents")

            # Create dataset
            datasets_dir = user_data_dir / "datasets"
            datasets_dir.mkdir(parents=True, exist_ok=True)

            dataset_name = config.get("dataset_name", "my_training_dataset")
            dataset_description = config.get("dataset_description", "")

            self.logger.info(f"Creating dataset: {dataset_name}")
            dataset_result = orchestrator.create_dataset(chunks_files, str(datasets_dir), dataset_name, dataset_description)

            if not dataset_result:
                raise Exception("Dataset creation failed")

            # Store results in context
            result = {
                "processing_job": job_result,
                "dataset_config": dataset_result,
                "chunks_files": chunks_files,
            }

            self.logger.info("Data preparation completed successfully")
            self.logger.info(f"Dataset created: {dataset_name}")

            return result

        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this step can be executed."""
        uploaded_files = st.session_state.get("dp_uploaded_files", [])

        if not uploaded_files:
            return False, "Please upload at least one PDF file"

        dataset_name = st.session_state.get("dp_config", {}).get("dataset_name", "").strip()
        if not dataset_name:
            return False, "Please provide a dataset name"

        return True, "Ready to process documents"


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
        """Execute backend logic for dry run step.

        This could validate the configuration, estimate training time,
        and check resource availability. For now, it's a placeholder.
        """
        self.logger.info("Executing dry run step")

        # Check if we have the necessary session state from previous steps
        dataset_name = context.get("dataset_name")
        if dataset_name:
            self.logger.info(f"Dry run for dataset: {dataset_name}")

        return {"dry_run_completed": True}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if dry run step is ready.

        For MVP, this is optional and can always be skipped.
        """
        # Dry run is always valid (optional step)
        return True, "Dry run validation passed"


class TrainingStep(PipelineStep):
    """ """

    def render_ui(self) -> None:
        """Render the UI for the training step."""
        # Step header
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

        st.markdown("---")
        st.markdown("#### Select Dataset and Base Model")

        # Locate datasets directory under the user data folder. Datasets are
        # saved during the data preparation step as ``user_data_dir/datasets/<name>``.
        user_data_dir = get_user_data_directory()
        datasets_root = user_data_dir / "datasets"
        dataset_options: list[str] = []
        dataset_paths: dict[str, str] = {}
        if datasets_root.exists():
            for p in datasets_root.iterdir():
                if p.is_dir() and (p / "train.jsonl").exists():
                    dataset_options.append(p.name)
                    dataset_paths[p.name] = str(p)
        # If no datasets are found, inform the user
        if not dataset_options:
            st.warning("No datasets were found. Please complete the Data Preparation step and create a dataset before training.")
        selected_dataset = None
        if dataset_options:
            selected_dataset = st.selectbox(
                "Dataset",
                options=dataset_options,
                help="Select a dataset directory containing train/validation/test JSONL files",
                key="tp_dataset_select",
            )
        # Store selection in session state for execution
        if selected_dataset:
            st.session_state.tp_selected_dataset = dataset_paths[selected_dataset]

        # Locate base models. Use the default models directory; if it doesn't
        # exist, fallback to a ``models`` directory in the project root.
        models_root = get_default_models_directory()
        if not models_root.exists():
            models_root = Path("models")
        model_options: list[str] = []
        model_paths: dict[str, str] = {}
        gguf_models: list[str] = []  # Track GGUF models separately
        meta_native_models: list[str] = []  # Track Meta native format models

        if models_root.exists():
            for p in models_root.iterdir():
                if p.is_dir():
                    # Check for PyTorch/Transformers format (compatible)
                    has_config = (p / "config.json").exists()

                    # Check for weights in various formats
                    has_single_weights = any((p / fname).exists() for fname in ["pytorch_model.bin", "model.safetensors"])
                    has_index = (p / "pytorch_model.bin.index.json").exists() or (p / "model.safetensors.index.json").exists()
                    has_sharded_weights = any(f.name.startswith("model-") and f.suffix == ".safetensors" for f in p.glob("*.safetensors"))

                    has_weights = has_single_weights or has_index or has_sharded_weights

                    if has_config and has_weights:
                        model_options.append(p.name)
                        model_paths[p.name] = str(p)
                    # Check for GGUF format (incompatible with training)
                    elif any(f.suffix == ".gguf" for f in p.glob("*.gguf")):
                        gguf_models.append(p.name)
                    # Check for Meta native format (consolidated.pth + params.json)
                    elif (p / "params.json").exists() and any(f.name.startswith("consolidated") and f.suffix == ".pth" for f in p.glob("*.pth")):
                        meta_native_models.append(p.name)
        selected_model = None
        if model_options:
            selected_model = st.selectbox(
                "Base Model",
                options=model_options,
                help="Select the base model to fine‑tune",
                key="tp_model_select",
            )
        else:
            st.error(f"**No compatible models found in**: `{models_root}`")

            # Show GGUF models if found, but explain they're incompatible
            if gguf_models:
                st.warning(
                    f"""
                    **Modelos GGUF detectados não compatíveis com fine-tuning:**
                    """
                )

            # Show Meta native format models if found
            if meta_native_models:
                st.warning(
                    f"""
                    **Modelos no formato Meta nativo detectados (não compatíveis):**
                    """
                )

            # Button to open models directory in Windows Explorer
            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("Abrir Pasta de Modelos", key="open_models_dir"):
                    import subprocess

                    # Create directory if it doesn't exist
                    models_root.mkdir(parents=True, exist_ok=True)
                    # Open in Windows Explorer
                    subprocess.Popen(f'explorer "{models_root}"')
                    st.success(f"Abrindo: {models_root}")
            with col_btn2:
                if st.button("Recarregar Lista de Modelos", key="reload_models"):
                    st.rerun()

        if selected_model:
            st.session_state.tp_selected_model = model_paths[selected_model]

        st.markdown("---")
        st.markdown("#### Training Hyperparameters")
        # Hyperparameter inputs. Use reasonable defaults for LoRA fine‑tuning.
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Number of epochs to train",
                key="tp_epochs",
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=4,
                step=1,
                help="Per‑device batch size",
                key="tp_batch_size",
            )
            gradient_accumulation = st.number_input(
                "Gradient Accumulation",
                min_value=1,
                max_value=32,
                value=1,
                step=1,
                help="Number of gradient accumulation steps",
                key="tp_gradient_accumulation",
            )
        with col2:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-3,
                value=2e-4,
                step=1e-5,
                format="%f",
                help="Initial learning rate for the optimiser",
                key="tp_learning_rate",
            )
            max_seq_length = st.number_input(
                "Max Sequence Length",
                min_value=128,
                max_value=4096,
                value=1024,
                step=128,
                help="Maximum sequence length (context window)",
                key="tp_max_seq_length",
            )
        with col3:
            lora_r = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                max_value=64,
                value=8,
                step=1,
                help="Rank of LoRA adapters (controls adapter size)",
                key="tp_lora_r",
            )
            lora_alpha = st.number_input(
                "LoRA Alpha",
                min_value=1,
                max_value=256,
                value=16,
                step=1,
                help="LoRA alpha scaling factor",
                key="tp_lora_alpha",
            )
            lora_dropout = st.number_input(
                "LoRA Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.01,
                format="%.2f",
                help="Dropout probability for LoRA layers",
                key="tp_lora_dropout",
            )

        # Store hyperparameters in session state for later retrieval
        st.session_state.tp_training_params = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation),
            "learning_rate": float(learning_rate),
            "max_seq_length": int(max_seq_length),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
        }

        st.markdown("---")
        st.markdown("#### Output Configuration")
        output_name = st.text_input(
            "Output Model Name",
            value="finetuned_model",
            help="Name of the directory to save the fine‑tuned model",
            key="tp_output_name",
        )
        if output_name:
            st.session_state.tp_output_model_name = output_name.strip()

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run the fine‑tuning procedure on the selected dataset and model."""
        self.logger.info("Executing training step")
        # Retrieve selections from session state
        dataset_dir = st.session_state.get("tp_selected_dataset")
        base_model_path = st.session_state.get("tp_selected_model")
        params = st.session_state.get("tp_training_params", {})
        output_name = st.session_state.get("tp_output_model_name")
        if not dataset_dir:
            raise ValueError("No dataset selected. Please choose a dataset before training.")
        if not base_model_path:
            raise ValueError("No base model selected. Please choose a model before training.")
        if not output_name:
            raise ValueError("No output model name provided.")

        # Construct absolute dataset and output paths
        user_data_dir = get_user_data_directory()
        # Ensure dataset directory exists
        ds_path = Path(dataset_dir)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_path}")
        # Output directory will reside under models directory
        models_root = get_default_models_directory()
        if not models_root.exists():
            models_root = Path("models")
        output_dir = models_root / output_name
        # Build training configuration dictionary
        training_cfg: dict[str, Any] = {
            "dataset_dir": str(ds_path),
            "base_model_path": str(base_model_path),
            "output_dir": str(output_dir),
            "epochs": params.get("epochs", 1),
            "batch_size": params.get("batch_size", 4),
            "learning_rate": params.get("learning_rate", 2e-4),
            "max_seq_length": params.get("max_seq_length", 1024),
            "lora_r": params.get("lora_r", 8),
            "lora_alpha": params.get("lora_alpha", 16),
            "lora_dropout": params.get("lora_dropout", 0.05),
            "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1),
        }
        # Log configuration
        self.logger.info(f"Training configuration: {training_cfg}")
        # Obtain orchestrator and run training
        from lite_llm_studio.app.app import get_orchestrator

        orchestrator = get_orchestrator()

        # Training will be executed (Streamlit will show processing state automatically)
        self.logger.info("Starting model training...")
        result = orchestrator.execute_training(training_cfg)

        if not result or not result.get("trained_model_path"):
            raise RuntimeError("Model training failed or returned no output.")

        trained_model_path = result["trained_model_path"]
        self.logger.info(f"Training completed successfully: {trained_model_path}")

        # Update context
        return {"trained_model": trained_model_path}

    def validate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Validate if this training step is ready to be executed.

        Checks for the presence of a selected dataset, selected base model
        and a non‑empty output model name. Also ensures that training
        hyperparameters have been configured.

        Args:
            context: Shared pipeline context (unused here).

        Returns:
            Tuple[bool, str]: A boolean indicating readiness, and a
            message explaining the issue if not ready.
        """
        if not st.session_state.get("tp_selected_dataset"):
            return False, "Please select a dataset to use for training"
        if not st.session_state.get("tp_selected_model"):
            return False, "Please select a base model"
        if not st.session_state.get("tp_output_model_name"):
            return False, "Please provide an output model name"
        # Check hyperparameters exist
        if not st.session_state.get("tp_training_params"):
            return False, "Please configure training hyperparameters"
        return True, "Ready to train"


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
            description="Upload and preprocess your training data for optimal results.",
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
    if "tp_processing" not in st.session_state:
        st.session_state.tp_processing = False  # Track if a step is currently processing
    if "tp_pipeline_finished" not in st.session_state:
        st.session_state.tp_pipeline_finished = False  # Track if pipeline has been finished


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

    # Check if currently processing
    is_processing = st.session_state.get("tp_processing", False)

    with left:
        # Disable Back button if at first step or processing
        if st.button("🡠 Back", key="tp_nav_back", disabled=(i == 0 or is_processing), use_container_width=True):
            st.session_state.tp_current_step = max(0, i - 1)
            st.rerun()

    with mid:
        is_completed = st.session_state.tp_completed[i]

        # Check if step can be completed
        can_complete, validation_msg = step.can_complete(st.session_state.tp_context)

        completed_label = "✓ Completed" if is_completed else "Complete Step"
        # Disable Complete button if already completed, can't complete, or processing
        button_disabled = is_completed or not can_complete or is_processing

        complete_clicked = st.button(
            completed_label,
            key=f"tp_complete_{i}",
            type="secondary" if is_completed else "primary",
            use_container_width=True,
            disabled=button_disabled,
            help=validation_msg if not can_complete else None,
        )

        if complete_clicked and not is_processing:
            # Set processing state immediately and rerun to disable buttons
            st.session_state.tp_processing = True
            st.rerun()

    with right:
        can_go_next = (i < pipeline.get_step_count() - 1) and st.session_state.tp_completed[i] and st.session_state.tp_unlocked[i + 1]
        # Disable Next button if can't go next or processing
        if st.button("Next 🡢", key="tp_nav_next", disabled=(not can_go_next or is_processing), use_container_width=True):
            st.session_state.tp_current_step = min(pipeline.get_step_count() - 1, i + 1)
            st.rerun()

    # Execute processing OUTSIDE the columns to avoid breaking alignment
    # This happens on the render after the button click
    if is_processing and not st.session_state.tp_completed[i]:
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
        except Exception as e:
            st.error(f"Error completing step: {str(e)}")
        finally:
            # Reset processing state
            st.session_state.tp_processing = False
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
                    st.session_state.tp_pipeline_finished = True  # Mark pipeline as finished
                    st.success("Pipeline completed successfully! The trained model is now available.")
                    st.rerun()  # Rerun to show GGUF conversion section
                else:
                    st.warning("Pipeline completed but no trained model was found in context.")
            except Exception as e:
                st.error(f"Error finishing pipeline: {str(e)}")

        # Show GGUF conversion section if pipeline has been finished
        if st.session_state.get("tp_pipeline_finished", False) and st.session_state.get("tp_result_model"):
            st.markdown("---")
            st.markdown("#### Convert to GGUF")

            col1, col2 = st.columns(2)
            with col1:
                quantization = st.selectbox(
                    "Quantization Format",
                    options=["f16", "q8_0", "f32", "bf16", "auto"],
                    index=0,
                    help="f16: Recommended (16-bit float), q8_0: Compressed (8-bit), f32: No compression, auto: Detects automatically",
                    key="tp_gguf_quant",
                )

            if st.button("Convert to GGUF", key="tp_convert_gguf", type="primary"):
                try:
                    with st.spinner("Converting model to GGUF... This may take a few minutes."):
                        from lite_llm_studio.core.configuration import get_default_models_directory
                        from lite_llm_studio.core.ml.model_converter import convert_finetuned_model_to_gguf

                        # Get the trained model path
                        final_result = st.session_state.get("tp_result_model")

                        # Get base model path from training config
                        base_model = st.session_state.get("tp_selected_model")

                        # Get models directory to save GGUF files
                        models_dir = str(get_default_models_directory())

                        logger.info(f"Converting model to GGUF: adapter={final_result}, base={base_model}, quant={quantization}")

                        result = convert_finetuned_model_to_gguf(
                            adapter_path=final_result,
                            base_model_path=base_model,
                            quantization=quantization,
                            models_dir=models_dir,  # Save GGUF in models/ directory
                        )

                        st.success("GGUF model conversion completed successfully!")

                        # Show the paths in a more organized way
                        st.markdown("#### Generated Files")

                        # GGUF model (main output)
                        st.markdown("**GGUF Model (ready to use):**")
                        st.code(result["gguf_model"], language=None)
                        st.caption("This is a **complete standalone model** - ready to use!")

                except Exception as e:
                    st.error(f"Conversion Error: {str(e)}")
                    logger.error(f"GGUF conversion error: {e}", exc_info=True)


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
