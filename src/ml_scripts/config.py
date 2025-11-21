"""
config.py

Shared configuration for ASL model training.
"""

from pathlib import Path
from typing import List

# Default targets to train
DEFAULT_TARGETS: List[str] = [
    "Handshape",
    "Movement",
    "MajorLocation",
    "MinorLocation",
]


# Default directory where models are stored
DEFAULT_MODELS_DIR: Path = Path("models")
