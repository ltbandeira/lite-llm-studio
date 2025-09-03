"""
LiteLLM Studio Execution Script
--------------------------------

Simple script to conveniently run LiteLLM Studio.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from lite_llm_studio.main import main

if __name__ == "__main__":
    exit(main())
