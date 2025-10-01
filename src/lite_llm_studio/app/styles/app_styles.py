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

    /* Apply text color only to main content elements that need it */
    .stApp [data-testid="stAppViewContainer"] .main .stMarkdown,
    .stApp [data-testid="stAppViewContainer"] .main .stMarkdown *,
    .stApp [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] * {
        color: var(--text) !important;
    }
    
    /* Main content area text - specific targeting */
    .stApp [data-testid="stAppViewContainer"] .main .block-container .stMarkdown,
    .stApp [data-testid="stAppViewContainer"] .main .block-container .stMarkdown *,
    .stApp [data-testid="stAppViewContainer"] .main .block-container [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stAppViewContainer"] .main .block-container [data-testid="stMarkdownContainer"] * {
        color: var(--text) !important;
    }
    
    /* Input elements in main content only - specific targeting */
    .stApp [data-testid="stAppViewContainer"] .main .stSelectbox > div > div,
    .stApp [data-testid="stAppViewContainer"] .main .stNumberInput > div > div > input,
    .stApp [data-testid="stAppViewContainer"] .main .stTextInput > div > div > input,
    .stApp [data-testid="stAppViewContainer"] .main .stTextArea > div > div > textarea {
        color: var(--text) !important;
        background: var(--panel) !important;
    }
    
    /* Labels in main content only - specific targeting */
    .stApp [data-testid="stAppViewContainer"] .main .stSelectbox > label,
    .stApp [data-testid="stAppViewContainer"] .main .stNumberInput > label,
    .stApp [data-testid="stAppViewContainer"] .main .stTextInput > label,
    .stApp [data-testid="stAppViewContainer"] .main .stTextArea > label,
    .stApp [data-testid="stAppViewContainer"] .main .stCheckbox > label,
    .stApp [data-testid="stAppViewContainer"] .main .stRadio > label,
    .stApp [data-testid="stAppViewContainer"] .main .stFileUploader > label {
        color: var(--text) !important;
    }
    
    /* Universal text visibility fixes for main content */
    .stApp [data-testid="stAppViewContainer"] .main *:not(button):not([role="button"]) {
        color: var(--text) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main p,
    .stApp [data-testid="stAppViewContainer"] .main span,
    .stApp [data-testid="stAppViewContainer"] .main div:not(.stButton),
    .stApp [data-testid="stAppViewContainer"] .main h1,
    .stApp [data-testid="stAppViewContainer"] .main h2,
    .stApp [data-testid="stAppViewContainer"] .main h3,
    .stApp [data-testid="stAppViewContainer"] .main h4,
    .stApp [data-testid="stAppViewContainer"] .main h5,
    .stApp [data-testid="stAppViewContainer"] .main h6 {
        color: var(--text) !important;
    }
    
    /* Checkbox specific styling */
    .stApp [data-testid="stAppViewContainer"] .main .stCheckbox {
        color: var(--text) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main .stCheckbox label {
        color: var(--text) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main .stCheckbox span {
        color: var(--text) !important;
    }
    
    /* Additional button selectors for maximum coverage */
    .stApp [data-testid="stAppViewContainer"] .main button[data-baseweb="button"],
    .stApp [data-testid="stAppViewContainer"] .main [data-baseweb="button"] {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main button[data-baseweb="button"][kind="primary"],
    .stApp [data-testid="stAppViewContainer"] .main [data-baseweb="button"][kind="primary"] {
        background: #6366f1 !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
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
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 12px 10px 16px !important;
        margin: 4px 0 !important;
        background: transparent !important;
        color: #d1d5db !important;
        font-weight: 600 !important;
        font-size: .97rem !important;
        letter-spacing:.15px !important;
        transition: background .2s ease, color .2s ease !important;
        outline: none !important;
    }
    
    .stSidebar .stButton > button:hover {
        background: rgba(255,255,255,.06) !important; 
        color:#fff !important;
    }
    
    .stSidebar .stButton > button:focus-visible {
        outline:2px solid rgba(99,102,241,.55);
        outline-offset:2px;
    }

    .stSidebar .stButton > button[kind="primary"] {
        background: rgba(99,102,241,.14) !important;
        color:#fff !important;
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

    /* ===== Data Upload Components ===== */
    .upload-section, .upload-results, .dataset-stats, 
    .processing-options, .training-header, .data-preview, 
    .error-section {
        margin: 1.5rem 0;
    }

    .upload-section .section-title,
    .upload-results .section-title,
    .dataset-stats .section-title,
    .processing-options .section-title,
    .training-header .section-title,
    .data-preview .section-title,
    .error-section .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .upload-section .section-sub,
    .processing-options .section-sub,
    .training-header .section-sub,
    .data-preview .section-sub {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 16px;
        line-height: 1.5;
    }

    /* File Upload Styling */
    .stFileUploader {
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: #f9fafb !important;
        transition: all 0.3s ease !important;
    }

    .stFileUploader:hover {
        border-color: var(--primary) !important;
        background: #f0f4ff !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        background: transparent !important;
    }

    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
        font-weight: 600 !important;
    }

    /* Processing Options Styling - Specific targeting */
    .stApp [data-testid="stAppViewContainer"] .main .processing-options .stCheckbox,
    .stApp [data-testid="stAppViewContainer"] .main .processing-options .stNumberInput {
        margin-bottom: 0.5rem;
    }

    .stApp [data-testid="stAppViewContainer"] .main .processing-options .stCheckbox > label {
        font-weight: 600;
        color: var(--text) !important;
    }
    
    /* Essential form elements styling only */
    .stApp [data-testid="stAppViewContainer"] .main .stCheckbox span,
    .stApp [data-testid="stAppViewContainer"] .main .stRadio span {
        color: var(--text) !important;
    }

    /* Main Content Buttons - Clean styling with good contrast */
    .stApp [data-testid="stAppViewContainer"] .main .stButton > button,
    .stApp [data-testid="stAppViewContainer"] .main button,
    .stApp [data-testid="stAppViewContainer"] .main [role="button"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        border: 2px solid #d1d5db !important;
        background: #ffffff !important;
        color: #1f2937 !important;
        min-height: 40px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }

    .stApp [data-testid="stAppViewContainer"] .main .stButton > button:hover,
    .stApp [data-testid="stAppViewContainer"] .main button:hover,
    .stApp [data-testid="stAppViewContainer"] .main [role="button"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        background: #f8fafc !important;
        border-color: #6366f1 !important;
        color: #1f2937 !important;
    }

    .stApp [data-testid="stAppViewContainer"] .main .stButton > button[kind="primary"],
    .stApp [data-testid="stAppViewContainer"] .main button[kind="primary"],
    .stApp [data-testid="stAppViewContainer"] .main [role="button"][kind="primary"] {
        background: #6366f1 !important;
        border-color: #6366f1 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stApp [data-testid="stAppViewContainer"] .main .stButton > button[kind="primary"]:hover,
    .stApp [data-testid="stAppViewContainer"] .main button[kind="primary"]:hover,
    .stApp [data-testid="stAppViewContainer"] .main [role="button"][kind="primary"]:hover {
        background: #5855eb !important;
        border-color: #5855eb !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        border: 2px solid #9ca3af !important;
        color: #374151 !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main .stButton > button[kind="secondary"]:hover {
        background: #f9fafb !important;
        border-color: #374151 !important;
        color: #374151 !important;
    }

    /* Upload Results Styling */
    .upload-results hr {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 8px 0;
        opacity: 0.3;
    }

    /* Dataset Statistics Cards */
    .dataset-stats .element-container {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow-1);
    }

    /* Data Preview Styling */
    .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }

    .stExpander > div > div {
        background: var(--panel) !important;
    }

    .stTextArea > div > div > textarea {
        background: #f9fafb !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: ui-monospace, 'Cascadia Code', 'Source Code Pro', Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.875rem !important;
        line-height: 1.5 !important;
    }

    /* Training Configuration Styling */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }

    .stSelectbox > div > div {
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
    }

    .stNumberInput > div > div > input {
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
    }

    /* Metrics Styling */
    .stMetric {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-1) !important;
    }

    .stMetric > div {
        color: var(--text) !important;
    }

    .stMetric [data-testid="metric-container"] > div:first-child {
        color: var(--muted) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }

    .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        color: var(--text) !important;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
    }

    /* Status Messages */
    .stSuccess {
        background: var(--ok-bg) !important;
        border: 1px solid var(--ok-bd) !important;
        color: var(--ok-tx) !important;
        border-radius: 8px !important;
    }

    .stError {
        background: var(--bad-bg) !important;
        border: 1px solid var(--bad-bd) !important;
        color: var(--bad-tx) !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: var(--warn-bg) !important;
        border: 1px solid var(--warn-bd) !important;
        color: var(--warn-tx) !important;
        border-radius: 8px !important;
    }

    .stInfo {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        color: #1e40af !important;
        border-radius: 8px !important;
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: var(--primary) !important;
        border-radius: 4px !important;
    }

    .stProgress > div {
        background: #e5e7eb !important;
        border-radius: 4px !important;
    }

    /* Tabs Styling for Training Page */
    .stTabs [data-baseweb="tab-list"] {
        background: #f9fafb;
        border-radius: 8px;
        padding: 0.25rem;
        margin-bottom: 2rem;
        border: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--muted);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.5);
    }

    .stTabs [aria-selected="true"] {
        background: var(--panel) !important;
        color: var(--text) !important;
        box-shadow: var(--shadow-1) !important;
        border: 1px solid var(--border);
    }

    /* Responsive Design for Data Components */
    @media (max-width: 768px) {
        .upload-section .section-sub,
        .processing-options .section-sub,
        .training-header .section-sub,
        .data-preview .section-sub {
            font-size: 0.8rem;
        }
        
        .stFileUploader {
            padding: 1rem !important;
        }
        
        .processing-options {
            margin: 1rem 0;
        }
    }
    
    /* Essential contrast fixes for main content only */
    
    /* File uploader styling */
    .stApp [data-testid="stAppViewContainer"] .main .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
    }
    
    /* Metrics styling */
    .stApp [data-testid="stAppViewContainer"] .main .stMetric [data-testid="metric-container"] > div:first-child {
        color: var(--muted) !important;
    }
    
    .stApp [data-testid="stAppViewContainer"] .main .stMetric [data-testid="metric-container"] > div:nth-child(2) {
        color: var(--text) !important;
    }
    </style>
    """


def load_fonts_and_styles():
    st.markdown(get_google_fonts(), unsafe_allow_html=True)
    st.markdown(get_global_styles(), unsafe_allow_html=True)
