import json
import os
from asyncio.log import logger
from datetime import datetime
from typing import Any

import streamlit as st

from lite_llm_studio.core.orchestration import Orchestrator

# ------------------ Config ------------------
st.set_page_config(page_title="LiteLLM Studio", layout="wide", initial_sidebar_state="expanded")

# ------------------ Global Styles ------------------
st.markdown(
    """
    <style>
    /* ===== Esconde cabeçalho nativo e label de widgets ===== */
    header[data-testid="stHeader"], div[data-testid="stSidebarHeader"] { display: none !important; }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none !important; }

    /* ===== Tema claro no conteúdo ===== */
    .stApp { color: #262730; background-color: #ffffff; }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: #000000;
        width: 220px !important;
        min-width: 220px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding: 1.5rem 0;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    /* ===== Logo topo ===== */
    .sidebar-logo {
        text-align: center;
        padding: 0 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin: 0 1rem 1rem 1rem;
    }
    .sidebar-logo img {
        width: 132px !important;
        height: 132px !important;
        border-radius: 8px;
        object-fit: contain;
    }

    /* ===== Botões de navegação ===== */
    .stSidebar .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 10px 12px;
        margin: 3px 0;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
    }
    
    .stSidebar .stButton > button[kind="secondary"] {
        background-color: transparent;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stSidebar .stButton > button[kind="secondary"]:hover {
        background-color: rgba(255,255,255,0.10);
        color: #ffffff;
        border-color: rgba(255,255,255,0.2);
    }
    
    .stSidebar .stButton > button[kind="primary"] {
        background-color: rgba(99,102,241,0.22);
        color: #ffffff;
        border-left: 3px solid #6366f1;
        font-weight: 700;
    }

    /* ===== Conteúdo principal ===== */
    .main .block-container { padding: 2rem 3rem; max-width: 1200px; }

    /* ===== Métricas / Cards ===== */
    .metric-card {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: all .2s ease;
    }
    .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateY(-2px); }

    h1 { color: #1f2937; font-weight: 700; margin-bottom: 2rem; }
    h2, h3 { color: #374151; font-weight: 600; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #f9fafb; border-radius: 8px; padding: .25rem; margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 6px; color: #6b7280; font-weight: 500; }
    .stTabs [aria-selected="true"] {
        background: #ffffff; color: #1f2937; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white; border: none; border-radius: 8px; font-weight: 600;
        padding: .75rem 1.5rem; transition: all .2s ease;
    }
    .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99,102,241,.4); }

    [data-testid="metric-container"] {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    [data-testid="metric-container"] [data-testid="metric-value"] { font-size: 1.5rem; font-weight: 700; color: #1f2937; }
    [data-testid="metric-container"] [data-testid="metric-label"] { font-size: .9rem; color: #6b7280; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------ Helpers ------------------
def get_image_base64(image_path: str) -> str:
    """Converte imagem para base64 para exibição inline"""
    import base64

    try:
        with open(image_path, "rb") as f:
            data = f.read()
            return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""


def _kv(label: str, value: Any):
    col1, col2 = st.columns([1, 3])
    col1.caption(label)
    col2.write(value if value not in (None, "") else "—")


def init_session_state():
    """Inicializa o estado da sessão"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"


def create_navigation_menu():
    """Cria menu de navegação nativo do Streamlit"""
    st.sidebar.markdown("")

    # Opções do menu
    menu_options = {"Dashboard": "Dashboard", "Hardware": "Hardware"}

    # Cria botões para cada opção
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


@st.cache_resource
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


@st.cache_data(show_spinner=False)
def run_hardware_scan() -> dict[str, Any]:
    orchestrator = get_orchestrator()
    report_data = orchestrator.execute_hardware_scan()
    return report_data or {}


# ------------------ App Initialization ------------------
init_session_state()

# ------------------ Sidebar ------------------
with st.sidebar:
    # Logo no topo
    icon_path = r"C:\Source\lite-llm-studio\src\lite_llm_studio\app\resources\lateral_bar_icon.png"
    if os.path.exists(icon_path):
        b64 = get_image_base64(icon_path)
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

    # Menu de navegação
    selected = create_navigation_menu()

# ------------------ Routing ------------------
page_key = selected

# ------------------ Pages ------------------
if page_key == "Dashboard":
    st.title("LiteLLM Studio")
    st.markdown("---")

elif page_key == "Hardware":
    st.title("Local Hardware Overview")

    with st.spinner("Analyzing system configuration..."):
        data = run_hardware_scan()

    if not data:
        st.error("Error retrieving system information.")
        st.stop()

    os_info = data.get("os") or {}
    cpu_info = data.get("cpu") or {}
    mem_info = data.get("memory") or {}
    gpus = data.get("gpus") or []
    disks = data.get("disks") or []

    st.markdown("### Visão Geral do Sistema")

    gcols = st.columns(4)
    with gcols[0]:
        st.metric(label="Sistema", value=f"{os_info.get('system', '—')}", help=f"Versão: {os_info.get('version', 'N/A')}")
    with gcols[1]:
        st.metric(
            label="Processador",
            value=cpu_info.get("brand", "—"),
            help=f"Cores: {cpu_info.get('cores', '—')} | Threads: {cpu_info.get('threads', '—')}",
        )
    with gcols[2]:
        st.metric(label="Memória RAM", value=f"{mem_info.get('total_memory', '—')} GB", help=f"Livre: {mem_info.get('free_memory', '—')} GB")
    with gcols[3]:
        st.metric(
            label="GPUs",
            value=str(len(gpus)),
            help=", ".join([g.get("name", "GPU") for g in gpus[:2]]) + ("..." if len(gpus) > 2 else "") if gpus else "Nenhuma detectada",
        )

    st.markdown("---")

    tab_os, tab_cpu, tab_gpu, tab_disk = st.tabs(["Sistema", "Processador", "GPUs", "Armazenamento"])

    with tab_os:
        st.markdown("#### Informações do Sistema Operacional")
        col1, col2 = st.columns(2)
        with col1:
            _kv("Sistema", os_info.get("system"))
            _kv("Versão", os_info.get("version"))
        with col2:
            _kv("Arquitetura", os_info.get("arch", "—"))

    with tab_cpu:
        st.markdown("#### Especificações do Processador")
        col1, col2 = st.columns(2)
        with col1:
            _kv("Modelo", cpu_info.get("brand"))
            _kv("Arquitetura", cpu_info.get("arch"))
            _kv("Cores Físicos", cpu_info.get("cores"))
        with col2:
            _kv("Threads", cpu_info.get("threads"))
            _kv("Frequência Base", f"{cpu_info.get('frequency', '—')} GHz")

    with tab_gpu:
        st.markdown("#### Placas Gráficas Disponíveis")
        if gpus:
            for idx, g in enumerate(gpus):
                with st.expander(f"GPU #{idx + 1}: {g.get('name', 'GPU Desconhecida')}", expanded=(len(gpus) == 1)):
                    col1, col2 = st.columns(2)
                    with col1:
                        _kv("VRAM Total", f"{g.get('total_vram', '—')} GB")
                        _kv("Suporte CUDA", "Sim" if g.get("cuda") else "Não")
                    with col2:
                        _kv("Driver", g.get("driver", "—"))
        else:
            st.info("Nenhuma GPU dedicada detectada no sistema.")

    with tab_disk:
        st.markdown("#### Dispositivos de Armazenamento")
        if disks:
            for idx, d in enumerate(disks):
                with st.expander(f"{d.get('name', f'Disco {idx + 1}')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        _kv("Capacidade Total", f"{d.get('total_space', '—')} GB")
                        _kv("Espaço Usado", f"{d.get('used_space', '—')} GB")
                    with col2:
                        _kv("Espaço Livre", f"{d.get('free_space', '—')} GB")
                        if d.get("total_space") and d.get("used_space"):
                            try:
                                used_pct = (float(d.get("used_space", 0)) / float(d.get("total_space", 1))) * 100
                                st.progress(used_pct / 100, text=f"Uso: {used_pct:.1f}%")
                            except (ValueError, TypeError) as e:
                                logger.debug("Dado inválido ao calcular progresso: %r | d=%r", e, d)
        else:
            st.info("Nenhum dispositivo de armazenamento detectado.")

    st.markdown("---")
    st.markdown("### Exportar Relatório")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Última análise: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    with col2:
        djson = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            "Baixar JSON",
            data=djson.encode("utf-8"),
            file_name=f"hardware_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
