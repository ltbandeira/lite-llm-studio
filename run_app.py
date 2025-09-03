"""
LiteLLM Studio Web Interface
----------------------------

Streamlit web interface for LiteLLM Studio.
This script launches the web interface that communicates with the orchestrator.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure Streamlit to run the app
if __name__ == "__main__":
    import streamlit.web.cli as stcli

    # Replace sys.argv to run the app.py file (entry point)
    sys.argv = [
        "streamlit",
        "run",
        str(src_path / "lite_llm_studio" / "app" / "app.py"),
        "--server.address",
        "localhost",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "false",
    ]

    sys.argv.extend(
        [
            "--server.maxUploadSize",
            "200",
            "--server.maxMessageSize",
            "200",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
            "--server.runOnSave",
            "true",
            "--server.fileWatcherType",
            "auto",
        ]
    )

    sys.exit(stcli.main())
