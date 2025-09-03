"""
LiteLLM Studio Execution Script
--------------------------------

Simple script to conveniently run LiteLLM Studio.
"""

import sys
from pathlib import Path

from lite_llm_studio.main import main

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    exit(main())
