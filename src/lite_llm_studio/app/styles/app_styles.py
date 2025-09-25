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

        --bg: #0b1220;
        --panel: #0f172a;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --border: #1f2937;

        --primary: #6366f1;
        --primary-weak: #eef2ff;

        --radius: 12px;
        --shadow-1: 0 1px 3px rgba(0,0,0,.40);
        --shadow-2: 0 6px 18px rgba(0,0,0,.45);

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
        color: #cbd5e1;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #0b1220;
        border-radius: 8px;
        padding: .25rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94a3b8;
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
        color: #cbd5e1;
        background: #0b1220;
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
        background: #1f2937;
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
        background: #1f2937;
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
        color: #9ca3af;
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
        background: #0b1220;
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
        background: #1f2937;
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
