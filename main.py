"""
Multi-agent tax filing: Filer → Verifier → Approver.
Entry point; implementation lives in schemas, client, prompts, agents, environment, showcase.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from showcase import run_showcase

if __name__ == "__main__":
    run_showcase(max_rounds=1)
