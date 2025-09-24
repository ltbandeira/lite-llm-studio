"""
Module app.styles.app_styles
----------------------------

This module contains all CSS styles used in the application.
"""

import streamlit as st


def get_google_fonts() -> str:
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    """


def get_global_styles() -> str:
    return """
    <style>
    /* ===== Hide native components ===== */
    header[data-testid="stHeader"], div[data-testid="stSidebarHeader"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }

    /* ===== Design tokens ===== */
    :root {
        --sidebar-w: 208px;
        --topbar-h: 64px;
        --footer-h: 52px;

        --bg: #f3f4f6;
        --panel: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;

        --primary: #6366f1;
        --primary-weak: #eef2ff;

        --radius: 12px;
        --shadow-1: 0 1px 3px rgba(0,0,0,.10);
        --shadow-2: 0 4px 12px rgba(0,0,0,.12);

        --font-sans:
            ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
            Inter, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji",
            "Segoe UI Emoji";
        --ok-bg:#ecfdf5; --ok-bd:#a7f3d0; --ok-tx:#065f46;
        --warn-bg:#fffbeb; --warn-bd:#fcd34d; --warn-tx:#92400e;
        --bad-bg:#fef2f2; --bad-bd:#fecaca; --bad-tx:#991b1b;
    }

    /* ===== Base ===== */
    .stApp {
        color: var(--text);
        background-color: var(--bg);
        font-family: var(--font-sans);
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: #0a0a0a;
        width: var(--sidebar-w) !important;
        min-width: var(--sidebar-w) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 1.25rem 0 1rem;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    /* ===== Sidebar Logo ===== */
    .sidebar-logo {
        text-align: center;
        padding: 0 0 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin: 0 1rem 1rem 1rem;
    }

    .sidebar-logo img{
        width: 132px !important;
        height: 132px !important;
        border-radius: 8px;
        object-fit: contain;
    }

    /* ===== Sidebar Menu ===== */
    .nav-caption {
        padding: .25rem 1rem; margin: .25rem 0 .25rem;
        font-size: .75rem; color: rgba(229,231,235,.6);
        text-transform: uppercase; letter-spacing:.12rem; font-weight:700;
    }

    .nav-container {
        padding: 0 .75rem;
    }

    .stSidebar .stButton > button {
        position: relative;
        width:100%;
        border: none;
        border-radius: 10px;
        padding: 10px 12px 10px 16px;
        margin: 4px 0;
        background: transparent;
        color: #d1d5db;
        font-weight: 600;
        font-size: .97rem;
        letter-spacing:.15px;
        transition: background .2s ease, color .2s ease;
        outline: none;
    }
    
    .stSidebar .stButton > button:hover {
        background: rgba(255,255,255,.06); color:#fff;
    }
    
    .stSidebar .stButton > button:focus-visible {
        outline:2px solid rgba(99,102,241,.55);
        outline-offset:2px;
    }

    .stSidebar .stButton > button[kind="primary"] {
        background: rgba(99,102,241,.14);
        color:#fff;
    }

    .stSidebar .stButton > button[kind="primary"]::before {
        content:"";
        position:absolute;
        left:8px;
        top:8px;
        bottom:8px;
        width:4px;
        background: var(--primary);
        border-radius:999px;
    }

    /* ===== Topbar ===== */
    .app-topbar {
        position: fixed;
        top: 0;
        left: var(--sidebar-w);
        width: calc(100% - var(--sidebar-w));
        height: var(--topbar-h);
        background: var(--panel);
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        z-index: 101;
        box-shadow: var(--shadow-1);
    }
    
    .app-topbar .title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: .2px;
        color: var(--text);
    }
    
    .app-topbar .actions a {
        margin-left: 16px;
        color: var(--muted);
        font-weight: 600;
        text-decoration: none;
        font-size: .95rem;
    }
    
    .app-topbar .actions a:hover,
    .app-topbar .actions a:focus-visible {
        color: var(--text);
        text-decoration: underline;
        outline: none;
    }

    /* ===== Main Content ===== */
    .main .block-container {
        padding: calc(var(--topbar-h) + 16px) clamp(1.5rem, 2vw, 3rem) calc(var(--footer-h) + 28px);
        max-width: clamp(960px, 90vw, 1400px);
    }

    /* ===== Cards Hardware ===== */
    h1 {
        color: var(--text);
        font-weight: 800;
        margin-bottom: 2rem;
        letter-spacing: .2px;
    }
    
    h2, h3 {
        color: #374151;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f9fafb;
        border-radius: 8px;
        padding: .25rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #6b7280;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--panel);
        color: var(--text);
        box-shadow: var(--shadow-1);
    }
    
    /* ===== KPI Cards ===== */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: 16px;
    }

    @media (max-width: 1280px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0,1fr));
        }
    }

    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-1);
        display: flex;
        gap: 12px;
        align-items: flex-start;
        transition: box-shadow .2s ease, transform .05s ease;
    }
    
    .kpi-card:hover {
        box-shadow: var(--shadow-2);
        transform: translateY(-1px);
    }
    
    @media (prefers-reduced-motion: reduce) {
        .kpi-card:hover {
            transform: none;
        }
    }

    .kpi-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--primary-weak);
        border: 1px solid #e8eaff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
    }
    
    .kpi-icon svg {
        width: 20px;
        height: 20px;
        color: var(--primary);
    }

    .kpi-body {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .kpi-label {
        font-size: .80rem;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: .2px;
    }

    .kpi-value {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .kpi-help {
        font-size: .75rem;
        color: #9ca3af;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* ===== Footer ===== */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: var(--sidebar-w);
        width: calc(100% - var(--sidebar-w));
        height: var(--footer-h);
        background: var(--panel);
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        z-index: 100;
        box-shadow: var(--shadow-1);
    }
    
    .app-footer .left {
        color:var(--muted);
        font-size:.9rem;
    }
    
    .app-footer .right a {
        margin-left: 16px;
        color:var(--muted);
        font-weight:600;
        text-decoration:none;
    }
    
    .app-footer .right a:hover,
    .app-footer .right a:focus-visible {
        color:var(--text);
        text-decoration:underline;
        outline:none;
    }

    /* ====== Hardware page styles ====== */
    .section {
        margin: 1rem 0 1.25rem;
    }

    .card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
    }
    
    .card + .card {
        margin-top: 12px;
    }

    .two-col {
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 16px;
    }

    @media (max-width: 992px) {
        .two-col {
            grid-template-columns: 1fr;
        }
    }

    .section-title {
        font-weight: 800;
        color: var(--text);
        margin: 0 0 8px 0;
        font-size: 1rem;
    }

    .section-sub {
        color: var(--muted);
        font-size: .9rem;
        margin-bottom: 12px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 8px;
        font-size: .8rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        color: #374151;
        background: #f9fafb;
        font-weight: 600;
    }
    
    .chip.ok {
        color: var(--ok-tx);
        background: var(--ok-bg);
        border-color: var(--ok-bd);
    }
    
    .chip.warn {
        color: var(--warn-tx);
        background: var(--warn-bg);
        border-color: var(--warn-bd);
    }
    
    .chip.bad {
        color: var(--bad-tx);
        background: var(--bad-bg);
        border-color: var(--bad-bd);
    }

    .list {
        display: grid;
        gap: 12px;
    }
    
    .disk-row {
        display: grid;
        grid-template-columns: 200px 1fr 200px;
        gap: 12px;
        align-items: center;
    }
    
    .disk-name {
        font-weight: 700;
        color: var(--text);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .disk-bar {
        height: 10px;
        background: #eef2f7;
        border-radius: 999px;
        overflow: hidden;
    }
    
    .disk-bar > span {
        display: block;
        height: 100%;
        background: var(--primary);
    }

    .gpu-list {
        display: grid;
        gap: 12px;
    }

    .gpu-item {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: var(--shadow-1);
    }
    
    .gpu-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }

    .gpu-left {
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
    }

    .gpu-ico {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--primary-weak);
        border: 1px solid #e8eaff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
    }
    
    .gpu-ico svg {
        width: 20px;
        height: 20px;
        color: var(--primary);
    }
    
    .gpu-title {
        font-weight: 800;
        color: var(--text);
    }
    
    .gpu-right {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }

    .brand-nvidia .gpu-ico {
        background: #ecfdf5;
        border-color: #a7f3d0;
    }

    .brand-intel .gpu-ico {
        background: #eff6ff;
        border-color: #bfdbfe;
    }

    .brand-amd .gpu-ico {
        background: #fef2f2;
        border-color: #fecaca;
    }

    .gpu-vram {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .gpu-vrambar {
        width: 140px;
        height: 8px;
        background: #eef2f7;
        border-radius: 999px;
        overflow: hidden;
    }

    .gpu-vrambar > span {
        display: block;
        height: 100%;
        background: var(--primary);
    }
    
    .gpu-vramtxt {
        font-size: .75rem;
        color: #6b7280;
    }

    .syscpu-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-1);
    }

    .syscpu-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: .75rem;
    }

    .syscpu-left {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
    }

    .syscpu-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--primary-weak);
        border: 1px solid #e8eaff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
    }

    .syscpu-icon svg {
        width: 20px;
        height: 20px;
        color: var(--primary);
    }

    .syscpu-title {
        font-weight: 800;
        color: var(--text);
    }

    .syscpu-sub {
        color: var(--muted);
        font-size: .9rem;
    }

    .syscpu-tags {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }

    .chip.sm {
        font-size: .72rem;
        padding: 2px 6px;
    }

    .spec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px 16px;
    }

    @media (max-width: 992px) {
        .spec-grid {
            grid-template-columns: 1fr;
        }
    }

    .spec-row {
        display: grid;
        grid-template-columns: 160px 1fr;
        gap: 8px;
        padding: 8px 10px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #f9fafb;
    }
    
    .spec-label {
        font-size: .8rem;
        color: var(--muted);
        font-weight: 700;
    }
    
    .spec-value {
        color: var(--text);
        font-weight: 700;
        word-break: break-word;
    }
    
    .storage-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: var(--shadow-1);
    }

    .storage-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: .5rem;
    }

    .storage-left {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 0;
    }

    .storage-ico {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--primary-weak);
        border: 1px solid #e8eaff;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .storage-ico svg {
        width: 20px;
        height: 20px;
        color: var(--primary);
    }

    .storage-title {
        font-weight: 800;
        color: var(--text);
    }
    
    .storage-sub {
        color: var(--muted);
        font-size: .9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .storage-tags {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
    }

    .storage-list {
        display: grid;
        gap: 10px;
    }
    
    .storage-row {
        display: grid;
        grid-template-columns: 80px 1fr auto;
        gap: 12px;
        align-items: center;
    }
    
    @media (max-width: 992px) {
        .storage-row {
            grid-template-columns: 1fr;
        }
    }

    .storage-name {
        font-weight: 800;
        color: var(--text);
    }

    .storage-badges {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .storage-bar {
        height: 12px;
        background: #eef2f7;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }

    .storage-bar > span {
        display: block;
        height: 100%;
        width: 0;
        transition: width .6s ease;
        background: linear-gradient(90deg, #6366f1, #7c83ff);
    }

    .storage-bar.warn > span {
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }

    .storage-bar.bad  > span {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    
    </style>
    """


def load_fonts_and_styles():
    st.markdown(get_google_fonts(), unsafe_allow_html=True)
    st.markdown(get_global_styles(), unsafe_allow_html=True)


def get_home_styles() -> str:
    return """
    /* Layout principal da Home */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    /* Títulos e tipografia locais */
    .main h1, .main h2, .main h3 { color: var(--text); font-weight: 700; }
    .main [data-testid="stMarkdownContainer"] p { color: var(--text); }

    /* Cards (alinhados ao Hardware) */
    .list-card, .chat-card{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        box-shadow: var(--shadow-1);
    }
    .list-item{
        display:flex; align-items:center; justify-content:space-between; gap:12px;
        padding:10px 8px; border:1px solid var(--border); border-radius:10px; background: var(--panel);
    }
    .list-item + .list-item{ margin-top:8px; }
    .list-left{ display:flex; align-items:center; gap:10px; min-width:0; }
    .list-ico{ width:28px; height:28px; border-radius:8px; background: var(--primary-weak); border:1px solid #e8eaff; display:flex; align-items:center; justify-content:center; }
    .list-ico svg{ width:16px; height:16px; color: var(--primary); }
    .empty{
        display:flex; align-items:center; gap:10px; padding:10px;
        border:1px dashed var(--border); border-radius:10px; color: var(--muted); background: var(--panel);
    }

    /* Cabeçalho do card (ícone + título + subtítulo) */
    .card-head{
        display:flex; align-items:center; gap:10px;
        padding: 8px; border:1px solid var(--border); border-radius:10px; background: var(--panel);
    }
    .card-head .list-ico{ width:28px; height:28px; border-radius:8px; background: var(--primary-weak); border:1px solid #e8eaff; display:flex; align-items:center; justify-content:center; }
    .card-head .list-ico svg{ width:16px; height:16px; color: var(--primary); }
    .card-head .title{ font-weight:800; color: var(--text); line-height:1; }
    .card-head .sub{ color: var(--muted); font-size:.9rem; margin-top:2px; }
    .list-card .inner-gap{ height: 10px; }

    /* Expanders com o mesmo design dos cards */
    .main .streamlit-expanderHeader{
        background: var(--panel);
        border: 2px solid var(--border);
        border-radius: 12px !important;
        color: var(--text);
        font-weight: 700;
        padding: 16px !important;
        margin-bottom: 4px !important;
        transition: all 0.2s ease;
    }
    
    .main .streamlit-expanderHeader:hover{
        border-color: var(--primary);
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.1);
    }
    
    .main .streamlit-expanderContent{
        background: var(--panel);
        border: 2px solid var(--border);
        border-top: none;
        border-radius: 0 0 12px 12px !important;
        padding: 16px !important;
    }

    .main details[open] .streamlit-expanderHeader{
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        border-bottom: none;
    }

    /* Botões (consistentes com o resto do app) */
    .main .stButton > button{
        background: #fff;
        border: 2px solid var(--primary);
        color: var(--primary);
        font-weight: 600;
        border-radius: 10px;
        transition: all .2s ease;
        box-shadow: none;
    }
    .main .stButton > button:hover{
        background: var(--primary);
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,70,229,.24);
    }
    .main .stButton > button[kind="primary"]{
        background: var(--primary);
        color: #fff;
        border: 2px solid var(--primary);
    }
    .main .stButton > button[kind="primary"]:hover{
        filter: brightness(0.95);
    }

    /* Botões dentro dos cards: reforça especificidade */
    .list-card .stButton > button{
        background: #fff !important;
        border: 2px solid var(--primary) !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all .2s ease !important;
        box-shadow: none !important;
    }
    .list-card .stButton > button:hover{
        background: var(--primary) !important;
        color: #fff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79,70,229,.24) !important;
    }
    .list-card .stButton > button[kind="primary"]{
        background: var(--primary) !important;
        color: #fff !important;
        border: 2px solid var(--primary) !important;
    }
    .list-card .stButton > button[kind="primary"]:hover{
        filter: brightness(0.95) !important;
    }

    /* Inputs/selects (bordas e cores alinhadas aos cards claros) */
    .main .stSelectbox > div > div,
    .main .stTextInput > div > div > input,
    .main .stNumberInput > div > div > input{
        background: var(--panel);
        border: 2px solid var(--border);
        color: var(--text);
        border-radius: 10px;
    }

    /* Força o estilo dentro de cards para evitar herança do tema escuro */
    .list-card .stTextInput > div > div > input,
    .list-card .stNumberInput > div > div > input{
        background: var(--panel) !important;
        border: 2px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        height: 40px !important;
        padding: 0 12px !important;
    }

    /* Sliders e labels */
    .main label, .main .stSlider > div > div > div { color: var(--text); }

    /* Mensagens de status com leveza (não conflita com Hardware) */
    .main .stSuccess{ background: #d1fae5; border:1px solid #10b981; color:#065f46; }
    .main .stError{ background: #fee2e2; border:1px solid #ef4444; color:#991b1b; }
    .main .stWarning{ background: #fef3c7; border:1px solid #f59e0b; color:#92400e; }
    .main .stInfo{ background: #dbeafe; border:1px solid #3b82f6; color:#1e40af; }

    /* Enhanced Model Directory Styles */
    .model-dir-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .model-dir-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: var(--primary);
    }

    .model-dir-header {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .model-dir-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
    }

    .model-dir-info {
        flex: 1;
        min-width: 0;
    }

    .model-dir-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text);
        margin-bottom: 4px;
    }

    .model-dir-path {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 0.9rem;
        color: var(--muted);
        background: rgba(0, 0, 0, 0.05);
        padding: 4px 8px;
        border-radius: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .model-dir-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .status-active {
        background: var(--ok-bg);
        color: var(--ok-tx);
        border: 1px solid var(--ok-bd);
    }

    .status-empty {
        background: var(--warn-bg);
        color: var(--warn-tx);
        border: 1px solid var(--warn-bd);
    }

    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Enhanced Model Selection Styles */
    .model-selection-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 16px;
    }

    .model-selection-header {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
    }

    .model-count-badge {
        display: inline-block;
        background: var(--primary);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .model-list {
        max-height: 250px;
        overflow-y: auto;
    }

    .model-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px;
        border-bottom: 1px solid var(--border);
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .model-item:last-child {
        border-bottom: none;
    }

    .model-item:hover {
        background: var(--primary-weak);
    }

    .model-item-selected {
        background: var(--primary-weak);
        border-left: 4px solid var(--primary);
    }

    .model-icon {
        width: 40px;
        height: 40px;
        background: var(--primary-weak);
        border: 2px solid var(--primary);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: var(--primary);
    }

    .model-info {
        flex: 1;
        min-width: 0;
    }

    .model-name {
        font-weight: 600;
        font-size: 1rem;
        color: var(--text);
        margin-bottom: 2px;
    }

    .model-version {
        font-size: 0.85rem;
        color: var(--muted);
    }

    .model-actions {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .model-size {
        padding: 4px 8px;
        background: rgba(0, 0, 0, 0.05);
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .model-selected-badge {
        width: 24px;
        height: 24px;
        background: var(--primary);
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }

    .model-selection-empty {
        padding: 40px 20px;
        text-align: center;
        color: var(--muted);
    }

    .empty-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
    }

    .empty-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text);
        margin-bottom: 8px;
    }

    .empty-subtitle {
        font-size: 0.9rem;
        color: var(--muted);
    }

    /* Status indicators */
    .status-indicator-loaded, .status-indicator-unloaded {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        height: 100%;
        text-align: center;
    }

    /* Enhanced button styling for model actions */
    .list-card .stButton > button {
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .list-card .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .list-card .stButton > button:hover::before {
        left: 100%;
    }

    /* Enhanced reindex button styling */
    .main .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
        border: none;
        color: white;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .main .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
        filter: brightness(1.05);
    }

    .main .stButton > button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .main .stButton > button[kind="primary"]:hover::before {
        left: 100%;
    }

    /* Enhanced text input styling */
    .main .stTextInput > div > div > input {
        background: var(--panel);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .main .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), 0 2px 8px rgba(0, 0, 0, 0.1);
        outline: none;
    }

    .main .stTextInput > div > div > input::placeholder {
        color: var(--muted);
        font-style: italic;
    }

    /* Enhanced selectbox styling */
    .main .stSelectbox > div > div > div {
        background: var(--panel);
        border: 2px solid var(--border);
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    .main .stSelectbox > div > div > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    /* Improved section titles */
    .section-title {
        font-weight: 800;
        font-size: 1.15rem;
        color: var(--text);
        margin: 2px 0 16px;
        letter-spacing: 0.3px;
        position: relative;
        padding-left: 12px;
    }

    .section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 20px;
        background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 100%);
        border-radius: 2px;
    }

    /* Enhanced home grid with better spacing */
    .home-grid {
        display: grid;
        grid-template-columns: 1.3fr 1fr;
        gap: 24px;
        padding: 0 8px;
    }

    @media (max-width: 992px) {
        .home-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Chat enhanced styling */
    .chat-card {
        background: var(--panel);
        border: 2px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        min-height: 500px;
        display: flex;
        flex-direction: column;
    }

    .main .stChatMessage{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }

    .main .stChatMessage:hover{
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .chat-card .stTextInput input{
        background: var(--panel);
        color: var(--text);
        border: 2px solid var(--border);
        border-radius: 999px;
        height: 48px;
        padding: 0 20px;
        font-size: 1rem;
        transition: all 0.2s ease;
    }

    .chat-card .stTextInput input:focus{
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }

    .chat-card .stTextInput input::placeholder{ 
        color: var(--muted); 
    }

    .chat-actions{ 
        display: flex; 
        justify-content: flex-end; 
        margin-top: 12px; 
    }

    .chat-actions .stButton>button{ 
        border-radius: 24px; 
        padding: 8px 16px; 
        font-weight: 700;
        transition: all 0.2s ease;
    }

    /* Divisores */
    .main hr { border-color: var(--border); border-width:1px; opacity:1; }
    """


def load_home_styles():
    st.markdown(f"<style>{get_home_styles()}</style>", unsafe_allow_html=True)
