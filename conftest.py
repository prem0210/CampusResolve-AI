from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-secret-do-not-use-in-production-please-change",
)

os.environ.setdefault(
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "60",
)