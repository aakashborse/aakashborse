"""pytest configuration — adds parent dir to sys.path so 'energex' package resolves."""
import sys
from pathlib import Path

# /home/user/aakashborse is the parent; energex/ is the package directory
sys.path.insert(0, str(Path(__file__).parent.parent))
