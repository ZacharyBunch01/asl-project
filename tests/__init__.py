"""
Ensure pytest can import project modules using the src/ layout.
This file modifies sys.path so Python knows where to find the code.
"""

import sys
from pathlib import Path

# Path to the project root (directory containing src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to src/
SRC_PATH = PROJECT_ROOT / "src"

# Add src/ to Python import path if not already present
if SRC_PATH not in map(Path, map(Path.resolve, map(Path, sys.path))):
    sys.path.insert(0, str(SRC_PATH))
