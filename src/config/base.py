# src/config/base.py
"""
Base environment loading, project paths, and system-level configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# --- Resolve project root and load .env ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if (PROJECT_ROOT / ".env.local").exists():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env.local")
else:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def getenv(key: str, default=None, cast=None):
    val = os.getenv(key, default)
    if val is None:
        return None
    if cast is bool:
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes", "t")
    return cast(val) if cast else val


class BaseConfig:
    PROJECT_ROOT = str(PROJECT_ROOT)
    _PROJECT_PATH = Path(PROJECT_ROOT)

    CPU_DEVICE = getenv("CPU_DEVICE", -1, int)
    GPU_DEVICE = getenv("GPU_DEVICE", 0, int)
    GPU_COOLDOWN_SEC = getenv("GPU_COOLDOWN_SEC", 1.0, float)

    CHUNK_SIZE = getenv("CHUNK_SIZE", 3000, int)
    CHUNK_OVERLAP = getenv("CHUNK_OVERLAP", 400, int)

