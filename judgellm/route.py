"""
CLI Entrypoint: `python -m judgellm.route --target-kappa-loss 0.02`
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from evaluation.router import main

if __name__ == "__main__":
    main()
