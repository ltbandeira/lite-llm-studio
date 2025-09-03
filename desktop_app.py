"""
LiteLLM Studio Desktop Launcher
-------------------------------

Desktop executable launcher for LiteLLM Studio.
This creates a proper desktop application that launches the Streamlit interface.
"""

import signal
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add the src directory to the Python path
if getattr(sys, "frozen", False):
    # Running in PyInstaller bundle
    application_path = Path(sys._MEIPASS)
    src_path = application_path / "src"
else:
    # Running in development
    application_path = Path(__file__).parent
    src_path = application_path / "src"

sys.path.insert(0, str(src_path))


def find_free_port():
    """Find a free port for Streamlit."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def check_port_in_use(port):
    """Check if port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def launch_streamlit(port):
    """Launch Streamlit server."""
    try:
        import streamlit.web.cli as stcli

        # Configure Streamlit arguments
        sys.argv = [
            "streamlit",
            "run",
            str(src_path / "lite_llm_studio" / "app" / "app.py"),
            "--global.developmentMode",
            "false",
            "--server.address",
            "localhost",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
            "--server.headless",
            "true",
            "--server.runOnSave",
            "false",
            "--logger.level",
            "error",
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
                "--server.fileWatcherType",
                "auto",
            ]
        )

        # Start Streamlit
        stcli.main()

    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)


def open_browser(port):
    """Open browser after Streamlit is ready."""
    url = f"http://localhost:{port}"

    # Wait for Streamlit to start
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = socket.create_connection(("localhost", port), timeout=1)
            response.close()
            break
        except (OSError, ConnectionRefusedError):
            if attempt < max_attempts - 1:
                time.sleep(1)
            else:
                print("Failed to connect to Streamlit server")
                return

    # Open browser
    print(f"Opening LiteLLM Studio at {url}")
    webbrowser.open(url)


def main():
    """Main application entry point."""
    print("LiteLLM Studio Desktop")
    print("=" * 40)

    # Find available port
    port = find_free_port()
    if check_port_in_use(port):
        port = find_free_port()

    print(f"Starting server on port {port}...")

    # Start browser opener in separate thread
    browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
    browser_thread.start()

    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n\nShutting down LiteLLM Studio...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Launch Streamlit
        launch_streamlit(port)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Critical error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
