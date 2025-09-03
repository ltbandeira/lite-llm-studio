# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# Get the project root directory
project_root = Path.cwd()
src_path = project_root / "src"

# Data files to include
datas = [
    (str(src_path), 'src'),
    (str(project_root / 'config'), 'config'),
]

# Collect Streamlit static files
datas += collect_data_files("streamlit", include_py_files=True)

# Copy essential metadata
try:
    datas += copy_metadata("streamlit")
    datas += copy_metadata("click")
    datas += copy_metadata("tornado")
    datas += copy_metadata("altair")
except Exception:
    pass  # Continue if some metadata is missing

# Essential hidden imports
hiddenimports = [
    # Core Streamlit
    'streamlit',
    'streamlit.web.bootstrap',
    'streamlit.web.server',
    'streamlit.web.server.server',
    'streamlit.config',
    'streamlit.runtime.scriptrunner.script_runner',
    'streamlit.delta_generator',
    'streamlit.components.v1',
    
    # Core dependencies
    'click',
    'tornado',
    'tornado.web',
    'tornado.ioloop',
    'tornado.websocket',
    'altair',
    'jsonschema',
    
    # Our app modules
    'lite_llm_studio.core.orchestration',
    'lite_llm_studio.core.instrumentation',
    'lite_llm_studio.core.configuration',
    'lite_llm_studio.core.data',
    'lite_llm_studio.core.ml',
    
    # System modules
    'threading',
    'subprocess',
    'webbrowser',
    'socket',
    'pathlib',
]

# Modules to exclude for size reduction
excludes = [
    'tkinter',
    'matplotlib',
    'scipy',
    'pandas.plotting',
    'pandas.io.excel', 
    'jupyter',
    'IPython',
    'numpy.distutils',
]

block_cipher = None

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LiteLLM Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
