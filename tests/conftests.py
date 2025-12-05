# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../my-package-name
SRC = ROOT / "src"

if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

