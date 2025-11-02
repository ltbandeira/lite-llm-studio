"""
Module app.modules.home_page
----------------------------

This module contains the model home page content.
"""

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from lite_llm_studio.core.configuration.desktop_app_config import (
    ensure_directory_exists,
    get_default_models_directory,
    get_models_directory_info,
)
from lite_llm_studio.core.configuration.model_schema import DiscoveryConfig, GenParams, RuntimeSpec
from lite_llm_studio.core.instrumentation.scanner import HardwareScanner
from lite_llm_studio.core.ml.registry import ModelRegistry
from lite_llm_studio.core.ml.runtimes.llamacpp_rt import LlamaCppRuntime

from ..components import create_directory_cards_html
from ..icons import ICONS

# Get logger for home page
logger = logging.getLogger("app.pages.home")


# Create a singleton hardware scanner instance
_hardware_scanner = None


def get_hardware_scanner() -> HardwareScanner:
    """Get or create a singleton hardware scanner instance."""
    global _hardware_scanner
    if _hardware_scanner is None:
        _hardware_scanner = HardwareScanner("app.hardware")
    return _hardware_scanner


def render_home_page():
    logger.info("Rendering home page")

    # Initialize session state for model management
    if "models_directory" not in st.session_state:
        default_models_dir = get_default_models_directory()
        ensure_directory_exists(default_models_dir, "Default models directory")
        st.session_state.models_directory = str(default_models_dir)
        logger.debug(f"Initialized models directory: {default_models_dir}")
    if "indexed_models" not in st.session_state:
        st.session_state.indexed_models = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "model_config" not in st.session_state:
        st.session_state.model_config = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0
    if "model_runtime" not in st.session_state:
        st.session_state.model_runtime = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "loaded_model" not in st.session_state:
        st.session_state.loaded_model = None
    if "last_indexed_at" not in st.session_state:
        st.session_state.last_indexed_at = None
    if "is_indexing" not in st.session_state:
        st.session_state.is_indexing = False
    if "compose_widget" not in st.session_state:
        st.session_state.compose_widget = ""
    if "use_gpu" not in st.session_state:
        st.session_state.use_gpu = False
    if "gpu_layers" not in st.session_state:
        st.session_state.gpu_layers = 0

    # Section 1: Models Directory and Indexing
    render_models_directory_section()

    st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)

    # Section 2: Model Selection
    render_model_selection_section()

    st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)

    # Section 3: Model Configuration
    render_model_configuration_section()

    st.markdown('<div style="height: 32px;"></div>', unsafe_allow_html=True)

    # Section 4: Model Interaction
    render_model_interaction_section()


def render_models_directory_section():
    logger.debug("Rendering models directory section")
    st.markdown('<div class="section-title">Models Directory</div>', unsafe_allow_html=True)

    # Get models directory info
    models_info = get_models_directory_info()
    dir_path = models_info["path"]
    model_count = len(st.session_state.indexed_models or [])
    last_idx = st.session_state.last_indexed_at

    logger.debug(f"Models directory: {dir_path}, indexed models: {model_count}")

    # Update session state with the current directory
    st.session_state.models_directory = dir_path

    # Directory info cards
    cards_html = create_directory_cards_html(
        dir_path=dir_path,
        model_count=model_count,
        last_indexed=last_idx,
        path_exists=models_info["exists"],
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Directory information and controls
    info_col, action_col = st.columns([5, 2], gap="small", vertical_alignment="bottom")

    with info_col:
        st.markdown(f"**Models Directory:** `{dir_path}`")
        stats_l, stats_c, stats_r = st.columns([1, 1, 1])
        with stats_l:
            st.markdown("**Directory:** Ready" if models_info["exists"] else "**Directory:** Missing")
        with stats_c:
            st.markdown(f'**Files:** {models_info["model_count"]} models')
        with stats_r:
            if models_info["total_size"] > 0:
                size_gb = models_info["total_size"] / (1024**3)
                st.markdown(f"**Size:** {size_gb:.1f} GB")
            else:
                st.markdown("**Size:** 0 GB")

    with action_col:
        disabled = st.session_state.is_indexing or not models_info["exists"]
        button_text = "Indexing..." if st.session_state.is_indexing else "Index Models"

        # botão ocupa toda a coluna e fica alinhado pela base do bloco
        clicked = st.button(button_text, use_container_width=True, disabled=disabled)

        if clicked:
            logger.info("Starting model indexing")
            st.session_state.is_indexing = True
            try:
                ensure_directory_exists(Path(dir_path), "Models directory")
                index_models()
                st.session_state.last_indexed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
                logger.info(f"Models indexed successfully: {len(st.session_state.indexed_models)} models found")
            except Exception as e:
                logger.error(f"Error indexing models: {e}", exc_info=True)
            finally:
                st.session_state.is_indexing = False
                st.rerun()

    # Show indexed models count
    if st.session_state.indexed_models:
        st.success(f"Found {len(st.session_state.indexed_models)} indexed models ready to use")
    elif models_info["model_count"] > 0:
        st.info(f"Found {models_info['model_count']} model files - click 'Index Models' to make them available")
    else:
        st.info("No model files found")


def render_model_selection_section():
    """Render the model selection section."""
    logger.debug("Rendering model selection section")
    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)

    if not st.session_state.indexed_models:
        logger.debug("No models indexed")
        st.info("No models indexed yet. Please set a directory and index models first.")
        return

    # Create model selection card
    selection_card_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['model']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Available Models</div>
                <div class="kpi-value">{len(st.session_state.indexed_models)} models found</div>
                <div class="kpi-help">Select a model to configure and use</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['play']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Selected</div>
                <div class="kpi-value">{st.session_state.selected_model['name'] if st.session_state.selected_model else 'None'}</div>
                <div class="kpi-help">Current active model</div>
            </div>
        </div>
    </div>
    """
    st.markdown(selection_card_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Model selection dropdown
    model_options = ["Select a model..."] + [model["name"] for model in st.session_state.indexed_models]
    selected_model_name = st.selectbox(
        "Choose Model",
        options=model_options,
        index=(
            0
            if not st.session_state.selected_model
            else model_options.index(st.session_state.selected_model["name"]) if st.session_state.selected_model["name"] in model_options else 0
        ),
        key="model_selector",
        label_visibility="collapsed",
    )

    if selected_model_name != "Select a model...":
        selected_model = next((model for model in st.session_state.indexed_models if model["name"] == selected_model_name), None)
        if selected_model != st.session_state.selected_model:
            logger.info(f"Model selected: {selected_model_name}")

            # Check if we're changing from a loaded model to a different model
            if st.session_state.loaded_model and st.session_state.loaded_model != selected_model:
                logger.info("Different model selected - resetting loaded state and configuration")
                st.session_state.model_loaded = False
                st.session_state.loaded_model = None
                # Reset configuration to defaults for the new model
                st.session_state.model_config = {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9,
                    "context_length": 4096,
                    "use_gpu": False,
                    "gpu_layers": 0,
                }
                # Clear chat history and compose field when switching models
                st.session_state.chat_history = []
                st.session_state.message_counter = 0
                st.session_state.compose_widget = ""

            st.session_state.selected_model = selected_model
            st.rerun()


def render_model_configuration_section():
    """Render the model configuration section."""
    logger.debug("Rendering model configuration section")
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    if not st.session_state.selected_model:
        logger.debug("No model selected for configuration")
        st.info("Please select a model first to configure it.")
        return

    # Configuration cards with Compute Mode
    cards_col, compute_col = st.columns([10, 1], gap="small", vertical_alignment="center")

    with cards_col:
        # Configuration cards - larger and better spaced
        config_card_html = f"""
        <div class="kpi-grid" style="gap: 10px;">
            <div class="kpi-card" style="flex: 1; min-width: 280px;">
                <div class="kpi-icon">{ICONS['settings']}</div>
                <div class="kpi-body">
                    <div class="kpi-label">Model</div>
                    <div class="kpi-value" style="font-size: 1.1em;">{st.session_state.selected_model["name"]}</div>
                    <div class="kpi-help">Configure parameters for model inference</div>
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 280px;">
                <div class="kpi-icon">{ICONS['cpu']}</div>
                <div class="kpi-body">
                    <div class="kpi-label">Status</div>
                    <div class="kpi-value" style="font-size: 1.1em;">{
                        "Loaded" if (st.session_state.model_loaded and 
                                    st.session_state.loaded_model and 
                                    st.session_state.loaded_model == st.session_state.selected_model) 
                        else "Not Loaded"
                    }</div>
                    <div class="kpi-help">{
                        "Model ready for inference" if (st.session_state.model_loaded and 
                                                       st.session_state.loaded_model and 
                                                       st.session_state.loaded_model == st.session_state.selected_model)
                        else "Load model to begin"
                    }</div>
                </div>
            </div>
        </div>
        """
        st.markdown(config_card_html, unsafe_allow_html=True)

    with compute_col:
        # Hardware selection next to the cards
        hardware_scanner = get_hardware_scanner()
        cuda_available = hardware_scanner.check_cuda_support()

        if cuda_available:
            # Simple CPU/GPU selection
            use_gpu = st.radio(
                "Compute Mode",
                options=["CPU", "GPU"],
                index=1 if st.session_state.get("use_gpu", False) else 0,
                help="Choose CPU for compatibility or GPU for speed",
            )

            if use_gpu == "GPU":
                st.session_state.use_gpu = True
                st.session_state.gpu_layers = -1  # Always use all layers for GPU
            else:
                st.session_state.use_gpu = False
                st.session_state.gpu_layers = 0
        else:
            # CUDA not available - force CPU mode
            st.radio("Compute Mode", options=["CPU"], index=0, disabled=True, help="GPU not available")
            st.session_state.use_gpu = False
            st.session_state.gpu_layers = 0

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Configuration parameters
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.model_config.get("temperature", 0.7),
            step=0.1,
            help="Controls randomness in responses",
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=8192,
            value=st.session_state.model_config.get("max_tokens", 1024),
            help="Maximum number of tokens to generate",
        )

    with col2:
        top_p = st.slider(
            "Top P", min_value=0.0, max_value=1.0, value=st.session_state.model_config.get("top_p", 0.9), step=0.01, help="Nucleus sampling parameter"
        )

        context_length = st.number_input(
            "Context Length",
            min_value=1,
            max_value=32768,
            value=st.session_state.model_config.get("context_length", 4096),
            help="Maximum context window size",
        )

    # Update session state with configuration (including GPU settings)
    st.session_state.model_config.update(
        {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "context_length": context_length,
            "use_gpu": st.session_state.get("use_gpu", False),
            "gpu_layers": st.session_state.get("gpu_layers", 0),
        }
    )

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Model action - only Load Model button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if model is already loaded
        is_model_loaded = (
            st.session_state.model_loaded and st.session_state.loaded_model and st.session_state.loaded_model == st.session_state.selected_model
        )

        if st.button("Load Model", use_container_width=True, disabled=is_model_loaded):
            logger.info(f"User requested to load model: {st.session_state.selected_model.get('name', 'Unknown')}")
            if load_model():
                st.rerun()

    # Show status message
    if st.session_state.model_loaded and st.session_state.loaded_model and st.session_state.loaded_model == st.session_state.selected_model:
        st.success("Model loaded and ready for inference")
    else:
        st.info("Configure the parameters above and click 'Load Model' to begin inference.")


def render_model_interaction_section():
    """Render the model interaction section."""
    logger.debug("Rendering model interaction section")
    st.markdown('<div class="section-title">Model Interaction</div>', unsafe_allow_html=True)

    if not st.session_state.selected_model:
        logger.debug("No model selected for interaction")
        st.info("Please select and configure a model first.")
        return

    # Check if the selected model is loaded
    if not st.session_state.model_loaded or not st.session_state.loaded_model or st.session_state.loaded_model != st.session_state.selected_model:
        logger.debug("Model not loaded or different model selected")
        if st.session_state.loaded_model and st.session_state.loaded_model != st.session_state.selected_model:
            st.warning("Different model selected. Please load the selected model in the Configuration section first.")
        else:
            st.warning("Model not loaded. Please load the model in the Configuration section first.")
        return

    if st.session_state.get("_submit_compose", False):
        msg = (st.session_state.pop("_submit_compose_text", "") or "").strip()
        st.session_state._submit_compose = False
        # Ensure the compose widget is cleared
        st.session_state.compose_widget = ""
        if msg:
            send_message(msg)
            return

    # Interaction card
    interaction_card_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['chat']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Chat Interface</div>
                <div class="kpi-value">Ready to chat</div>
                <div class="kpi-help">Interact with {st.session_state.selected_model["name"]}</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['system']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Messages</div>
                <div class="kpi-value">{len(st.session_state.chat_history)}</div>
                <div class="kpi-help">Total chat messages</div>
            </div>
        </div>
    </div>
    """
    st.markdown(interaction_card_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Chat toolbar
    _, tb_right = st.columns([1, 0.22])
    with tb_right:
        if st.button("Clear history", key="clear_chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.message_counter = 0
            st.session_state.pending_user_msg = None
            st.rerun()

    # Chat history panel
    with st.container(height=380, border=True):
        if st.session_state.get("chat_history"):
            logger.debug(f"Rendering chat history with {len(st.session_state.chat_history)} messages")
            for msg in st.session_state.chat_history:
                role = "user" if msg["role"] == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(msg["content"])
        else:
            logger.debug("No chat history to render")
            st.info("No messages yet. Start the conversation below!")

    # Fixed input field at the bottom of the page
    with st.form("compose_form", border=False):
        user_input = st.text_area(
            "Message",
            value=st.session_state.get("compose_widget", ""),
            label_visibility="collapsed",
            placeholder="Type your message…",
            height=100,
        )
        bcol1, _ = st.columns([1, 7])
        send_clicked = bcol1.form_submit_button("Send", use_container_width=True)

        if send_clicked:
            # Lê o conteúdo digitado DIRETO da variável do form
            prompt = (user_input or "").strip()
            if prompt:
                # Limpa o campo e envia a mensagem
                st.session_state.compose_widget = ""
                st.session_state._submit_compose = True
                st.session_state._submit_compose_text = prompt
                st.rerun()
            else:
                st.warning("Type a message first.")


def index_models():
    """Index models from the default directory using ModelRegistry."""
    logger.info("Starting model indexing process")
    try:
        models_dir = Path(st.session_state.models_directory)
        logger.debug(f"Indexing models from directory: {models_dir}")

        # Ensure directory exists
        if not models_dir.exists():
            logger.warning(f"Models directory does not exist, creating: {models_dir}")
            ensure_directory_exists(models_dir, "Models directory")
            if not models_dir.exists():
                raise Exception("Could not create models directory")

        # Use ModelRegistry to discover models
        discovery_config = DiscoveryConfig(models_root=str(models_dir))
        registry = ModelRegistry(discovery_config)

        # Scan for models with progress indication
        logger.debug("Scanning for models...")
        model_cards = registry.scan()
        logger.debug(f"Found {len(model_cards)} model cards")

        # Convert ModelCards to the format expected by the UI
        models = []
        for card in model_cards:
            model_file = card.model_file()
            if model_file.exists():
                stat = model_file.stat()
                models.append(
                    {
                        "name": card.name,
                        "slug": card.slug,
                        "path": str(model_file),
                        "size": f"{stat.st_size / (1024**3):.2f} GB",
                        "format": model_file.suffix.upper(),
                        "modified": f"{stat.st_mtime:.0f}",
                        "card": card,  # Keep the original card for loading
                    }
                )

        st.session_state.indexed_models = models
        logger.info(f"Indexing complete: {len(models)} models indexed")

        if len(models) == 0:
            logger.warning("No models found in directory")
            st.warning("No models found. Make sure to place .gguf files in the models directory.")
        else:
            logger.info(f"Successfully indexed models: {[m['name'] for m in models]}")
            st.success(f"Successfully indexed {len(models)} models!")

    except Exception as e:
        logger.error(f"Error indexing models: {e}", exc_info=True)
        st.error(f"Error indexing models: {str(e)}")
        raise


def load_model():
    """Load the selected model using LlamaCppRuntime."""
    logger.info("Attempting to load model")
    try:
        if not st.session_state.selected_model:
            logger.error("No model selected")
            st.error("No model selected")
            return False

        model_card = st.session_state.selected_model.get("card")
        if not model_card:
            logger.error("Model card not found")
            st.error("Model card not found")
            return False

        logger.info(f"Loading model: {model_card.name}")

        # Create runtime spec from configuration
        use_gpu = st.session_state.model_config.get("use_gpu", False)
        gpu_layers = st.session_state.model_config.get("gpu_layers", -1) if use_gpu else 0
        runtime_spec = RuntimeSpec(n_ctx=st.session_state.model_config.get("context_length", 4096), n_threads=None, n_gpu_layers=gpu_layers)

        # Debug info
        logger.info("GPU Configuration Debug:")
        logger.info(f"  use_gpu from config: {use_gpu}")
        logger.info(f"  gpu_layers from config: {st.session_state.model_config.get('gpu_layers', 'NOT_SET')}")
        logger.info(f"  calculated gpu_layers: {gpu_layers}")
        logger.info(f"  runtime_spec.n_gpu_layers: {runtime_spec.n_gpu_layers}")

        # Initialize or reinitialize runtime
        if st.session_state.model_runtime is None:
            logger.debug("Initializing LlamaCppRuntime")
            st.session_state.model_runtime = LlamaCppRuntime()
        elif st.session_state.model_loaded:
            logger.debug("Unloading previous model before loading new one")
            try:
                st.session_state.model_runtime.unload()
            except Exception as unload_error:
                logger.warning(f"Error unloading previous model: {unload_error}")

        # Load model
        with st.spinner(f"Loading model {model_card.name}..."):
            st.session_state.model_runtime.load(model_card, runtime_spec)
            st.session_state.model_loaded = True
            st.session_state.loaded_model = st.session_state.selected_model

        logger.info(f"Model {model_card.name} loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        st.session_state.loaded_model = None
        return False


def send_message(message: str):
    """Send a message and get model response."""
    logger.info(f"Processing user message: {message[:50]}...")

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": message})

    try:
        # Check if model is loaded and matches selected model
        if (
            not st.session_state.model_loaded
            or not st.session_state.model_runtime
            or not st.session_state.loaded_model
            or st.session_state.loaded_model != st.session_state.selected_model
        ):
            logger.warning("Attempted to send message with no model loaded or model mismatch")
            model_response = "Model not loaded or different model selected. Please load the model first."
        else:
            # Create generation parameters
            gen_params = GenParams(
                temperature=st.session_state.model_config.get("temperature", 0.7),
                top_p=st.session_state.model_config.get("top_p", 0.9),
                max_new_tokens=st.session_state.model_config.get("max_tokens", 256),
                stop=["User:", "System:", "\n\nUser:", "\n\nSystem:"],  # Stop at conversation markers
            )
            logger.debug(f"Generation params: temp={gen_params.temperature}, " f"top_p={gen_params.top_p}, max_tokens={gen_params.max_new_tokens}")

            # Generate response using the loaded model
            logger.debug("Generating model response...")
            with st.spinner("Generating response..."):
                model_response = st.session_state.model_runtime.generate(
                    st.session_state.chat_history, gen_params  # Include all messages including current user message
                )

                # Clean up the response
                model_response = model_response.strip()
                if not model_response:
                    logger.warning("Empty response generated")
                    model_response = "I couldn't generate a response. Please try again."
                else:
                    logger.info(f"Response generated successfully (length: {len(model_response)} chars)")

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        model_response = f"Error generating response: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": model_response})
    logger.info(f"Added assistant response to chat history. Total messages: {len(st.session_state.chat_history)}")
    logger.debug(f"Last message: {st.session_state.chat_history[-1] if st.session_state.chat_history else 'None'}")

    # Increment counter to reset input field
    st.session_state.message_counter += 1

    # Clear the compose widget to ensure field is empty
    st.session_state.compose_widget = ""

    # Trigger rerun to update the interface
    st.rerun()
