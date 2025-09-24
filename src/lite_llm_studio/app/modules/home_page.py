"""
Module app.modules.home_page
----------------------------

This module contains the model home page content.
"""

import os
from pathlib import Path
from typing import Any

import streamlit as st

from ..icons import ICONS
from lite_llm_studio.core.ml.registry import ModelRegistry
from lite_llm_studio.core.ml.runtimes.llamacpp_rt import LlamaCppRuntime
from lite_llm_studio.core.configuration.model_schema import DiscoveryConfig, GenParams, RuntimeSpec


def render_home_page():
    """Render the home page with model management and interaction sections."""
    st.markdown("---")

    # Initialize session state for model management
    if "models_directory" not in st.session_state:
        st.session_state.models_directory = ""
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
    """Render the models directory and indexing section."""
    st.markdown('<div class="section-title">Models Directory</div>', unsafe_allow_html=True)

    # Create a card for directory selection and model indexing
    directory_card_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['folder']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Model Directory</div>
                <div class="kpi-value">{"Set directory path" if not st.session_state.models_directory else "Directory configured"}</div>
                <div class="kpi-help">Browse and index models from local directory</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['refresh']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Status</div>
                <div class="kpi-value">{len(st.session_state.indexed_models)} models</div>
                <div class="kpi-help">Indexed models available</div>
            </div>
        </div>
    </div>
    """
    st.markdown(directory_card_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        directory_path = st.text_input(
            "Directory Path",
            value=st.session_state.models_directory,
            placeholder="Enter path to models directory (e.g., C:\\models or /home/user/models)",
            key="models_dir_input",
            label_visibility="collapsed",
        )
        if directory_path != st.session_state.models_directory:
            st.session_state.models_directory = directory_path

    with col2:
        if st.button("🔍 Browse", use_container_width=True):
            st.info("File browser integration would be implemented here")

    with col3:
        if st.button("📋 Index Models", use_container_width=True, disabled=not st.session_state.models_directory):
            index_models()

    # Show directory status
    if st.session_state.models_directory:
        if os.path.exists(st.session_state.models_directory):
            st.success(f"✅ Directory exists: {st.session_state.models_directory}")
            st.info(f"Found {len(st.session_state.indexed_models)} indexed models")
        else:
            st.error("❌ Directory does not exist")


def render_model_selection_section():
    """Render the model selection section."""
    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)

    if not st.session_state.indexed_models:
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
            st.session_state.selected_model = selected_model
            st.rerun()


def render_model_configuration_section():
    """Render the model configuration section."""
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    if not st.session_state.selected_model:
        st.info("Please select a model first to configure it.")
        return

    # Configuration card
    config_card_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['settings']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Model Configuration</div>
                <div class="kpi-value">{st.session_state.selected_model["name"]}</div>
                <div class="kpi-help">Configure parameters for model inference</div>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">{ICONS['cpu']}</div>
            <div class="kpi-body">
                <div class="kpi-label">Performance</div>
                <div class="kpi-value">{"🟢 Loaded" if st.session_state.model_loaded else "🔴 Not Loaded"}</div>
                <div class="kpi-help">{"Model ready for inference" if st.session_state.model_loaded else "Load model to begin"}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(config_card_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)

    # Configuration parameters
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.model_config.get("temperature", 0.7),
            step=0.1,
            help="Controls randomness in responses",
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=8192,
            value=st.session_state.model_config.get("max_tokens", 2048),
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

    # Update session state with configuration
    st.session_state.model_config.update({"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "context_length": context_length})

    # Model status and actions
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Reset Config", use_container_width=True):
            st.session_state.model_config = {}
            st.rerun()

    with col2:
        if st.button("✅ Save Config", use_container_width=True):
            st.success("Configuration saved!")

    with col3:
        model_status = "🟢 Loaded" if st.session_state.model_loaded else "🔴 Not Loaded"
        if st.button(f"🚀 Load Model", use_container_width=True, disabled=st.session_state.model_loaded):
            load_model()

    with col4:
        if st.button("🛑 Unload", use_container_width=True, disabled=not st.session_state.model_loaded):
            unload_model()

    # Show model status
    if st.session_state.model_loaded:
        st.success(f"✅ Model loaded and ready for inference")
    else:
        st.info(f"ℹ️ Model not loaded. Click 'Load Model' to begin inference.")


def render_model_interaction_section():
    """Render the model interaction section."""
    st.markdown('<div class="section-title">Model Interaction</div>', unsafe_allow_html=True)

    if not st.session_state.selected_model:
        st.info("Please select and configure a model first.")
        return

    if not st.session_state.model_loaded:
        st.warning("⚠️ Model not loaded. Please load the model in the Configuration section first.")
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

    # Chat interface
    chat_container = st.container()

    # Display chat history
    if st.session_state.chat_history:
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(
                        f"""<div style="background-color: #f0f9ff; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #0ea5e9;">
                        <strong>You:</strong> {message['content']}</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""<div style="background-color: #f8fafc; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #6366f1;">
                        <strong>{st.session_state.selected_model['name']}:</strong> {message['content']}</div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("💬 Start a conversation with your model...")

    # Input area
    col1, col2 = st.columns([4, 1])

    with col1:
        # Use contador para forçar reset do campo
        user_input = st.text_area(
            "Message",
            placeholder="Type your message here...",
            key=f"user_message_input_{st.session_state.message_counter}",
            height=100,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown('<div style="height: 45px;"></div>', unsafe_allow_html=True)
        send_clicked = st.button("Send", use_container_width=True, disabled=not user_input.strip())

        if send_clicked and user_input.strip():
            send_message(user_input.strip())

        st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def index_models():
    """Index models from the specified directory using ModelRegistry."""
    try:
        models_dir = Path(st.session_state.models_directory)
        if not models_dir.exists():
            st.error("Directory does not exist")
            return

        # Use ModelRegistry to discover models
        discovery_config = DiscoveryConfig(models_root=str(models_dir))
        registry = ModelRegistry(discovery_config)

        with st.spinner("Scanning for models..."):
            model_cards = registry.scan()

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
        st.success(f"Successfully indexed {len(models)} models using ModelRegistry!")
        st.rerun()

    except Exception as e:
        st.error(f"Error indexing models: {str(e)}")


def load_model():
    """Load the selected model using LlamaCppRuntime."""
    try:
        if not st.session_state.selected_model:
            st.error("No model selected")
            return False

        model_card = st.session_state.selected_model.get("card")
        if not model_card:
            st.error("Model card not found")
            return False

        # Create runtime spec from configuration
        runtime_spec = RuntimeSpec(
            n_ctx=st.session_state.model_config.get("context_length", 4096), n_threads=None, n_gpu_layers=0  # Auto-detect  # CPU-only for now
        )

        # Initialize runtime
        if st.session_state.model_runtime is None:
            st.session_state.model_runtime = LlamaCppRuntime()

        # Load model
        with st.spinner(f"Loading model {model_card.name}..."):
            st.session_state.model_runtime.load(model_card, runtime_spec)
            st.session_state.model_loaded = True

        st.success(f"✅ Model {model_card.name} loaded successfully!")
        return True

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return False


def unload_model():
    """Unload the current model."""
    try:
        if st.session_state.model_runtime:
            st.session_state.model_runtime.unload()
        st.session_state.model_loaded = False
        st.success("Model unloaded successfully!")
    except Exception as e:
        st.error(f"Error unloading model: {str(e)}")


def send_message(message: str):
    """Send a message and get model response."""
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": message})

    try:
        # Check if model is loaded
        if not st.session_state.model_loaded or not st.session_state.model_runtime:
            model_response = "⚠️ Model not loaded. Please load a model first."
        else:
            # Create generation parameters
            gen_params = GenParams(
                temperature=st.session_state.model_config.get("temperature", 0.7),
                top_p=st.session_state.model_config.get("top_p", 0.9),
                max_new_tokens=st.session_state.model_config.get("max_tokens", 256),
                stop=["User:", "System:", "\n\nUser:", "\n\nSystem:"],  # Stop at conversation markers
            )

            # Generate response using the loaded model
            with st.spinner("Generating response..."):
                model_response = st.session_state.model_runtime.generate(
                    st.session_state.chat_history, gen_params  # Include all messages including current user message
                )

                # Clean up the response
                model_response = model_response.strip()
                if not model_response:
                    model_response = "I couldn't generate a response. Please try again."

    except Exception as e:
        model_response = f"❌ Error generating response: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": model_response})

    # Increment counter to reset input field
    st.session_state.message_counter += 1

    # Trigger rerun to update the interface
    st.rerun()
