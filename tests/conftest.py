"""
tests/conftest.py — shared pytest configuration.
Adds project root to sys.path so all imports resolve correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
