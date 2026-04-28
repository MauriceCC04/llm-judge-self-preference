from __future__ import annotations

"""Compatibility wrapper for the deprecated repo_audit entrypoint.

The stronger code-first audit now lives in tools.study_audit. Keep this wrapper
so any old commands still work, but delegate all behavior to the canonical tool.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.study_audit import main


if __name__ == "__main__":
    main()
