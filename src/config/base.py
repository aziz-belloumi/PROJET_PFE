# src/config/base.py
"""
Base environment loading and project path configuration.
"""

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
    PROJECT_ROOT  = str(PROJECT_ROOT)
    _PROJECT_PATH = Path(PROJECT_ROOT)
