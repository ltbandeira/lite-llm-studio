"""
LiteLLM Studio Development Launcher
-----------------------------------

Development launcher for LiteLLM Studio.
For production use, build the desktop executable with `build_desktop.bat`.
"""

import socket
import sys
from pathlib import Path

# Add the src directory to the Python path
if getattr(sys, "frozen", False):
    # Running inside a PyInstaller bundle
    src_path = Path(sys._MEIPASS) / "src"  # type: ignore[attr-defined]
else:
    # Running from source (development)
    src_path = Path(__file__).parent / "src"

sys.path.insert(0, str(src_path))


def find_free_port() -> int:
    """
    Find an available TCP port on the local machine.

    Uses the OS to allocate a free port by binding to port 0 and
    reading back the assigned port from the socket.

    Returns:
        int: A free TCP port number suitable for starting a server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port: int = s.getsockname()[1]
    return port


def launch_streamlit(port: int) -> None:
    """
    Launch the Streamlit development server.

    Configures `sys.argv` to invoke `streamlit run` on the app entrypoint and
    enables convenient development flags.

    Args:
        port (int): TCP port where the Streamlit server should listen.

    Raises:
        SystemExit: If Streamlit fails to launch (exits the process with code 1).
    """
    try:
        import streamlit.web.cli as stcli

        app_path = str(src_path / "lite_llm_studio" / "app" / "app.py")

        # Development mode
        sys.argv = [
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(port),
            "--server.runOnSave",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        print(f"Starting development server on http://localhost:{port}")
        stcli.main()

    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        sys.exit(1)


def main() -> None:
    """
    Main entrypoint for the development server.
    """
    print("LiteLLM Studio - Development Server")
    print("=" * 40)

    # Setup application directories
    try:
        from lite_llm_studio.core.configuration import setup_application_directories

        directories = setup_application_directories()
        print("Application directories initialized:")
        for name, path in directories.items():
            print(f"  {name}: {path}")
        print()
    except Exception as e:
        print(f"Warning: Could not setup application directories: {e}")

    # Prefer the standard Streamlit dev port; fall back if busy
    port = 8501
    try:
        # Quick availability check by attempting to bind
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
    except OSError:
        print(f"Port {port} is busy, finding alternative...")
        port = find_free_port()

    try:
        launch_streamlit(port)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
