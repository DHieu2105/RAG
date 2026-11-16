import sys
from pathlib import Path

# Ensure backend root is on sys.path so app modules are discoverable
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from app.main import app

__all__ = ["app"]
