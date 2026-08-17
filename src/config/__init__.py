# src/config/__init__.py
"""
Unified configuration package for Data Exploration.
"""

from .base import BaseConfig, PROJECT_ROOT, getenv
from .db import DatabaseConfig
from .preprocessing import PreprocessingConfig


class Config(
    BaseConfig,
    DatabaseConfig,
    PreprocessingConfig,
):
    """
    Consolidated configuration class aggregating active sub-modules.
    Allows accessing all constants directly via Config.<ATTR>.
    """
    pass


__all__ = [
    "Config",
    "BaseConfig",
    "DatabaseConfig",
    "PreprocessingConfig",
    "PROJECT_ROOT",
    "getenv",
]
