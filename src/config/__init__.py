# src/config/__init__.py
"""
Unified configuration package for the NLP Pipeline.
Organized into modular sub-configs with full backward-compatibility.
"""

from __future__ import annotations

from .base import BaseConfig, PROJECT_ROOT, getenv
from .db import DatabaseConfig
from .models import ModelsConfig
from .pipeline import PipelineConfig
from .hyperparameters import HyperparametersConfig
from .heuristics import HeuristicsConfig
from .preprocessing import PreprocessingConfig
from .qwen import QwenConfig
from .ner_labels import NERLabelsConfig
from .chunking import token_chunks
from .db_config import DatabaseConnection


class Config(
    BaseConfig,
    DatabaseConfig,
    ModelsConfig,
    PipelineConfig,
    HyperparametersConfig,
    HeuristicsConfig,
    PreprocessingConfig,
    QwenConfig,
    NERLabelsConfig,
):
    """
    Consolidated configuration class aggregating all sub-modules.
    Allows accessing all constants directly via Config.<ATTR>.
    """
    pass


__all__ = [
    "Config",
    "BaseConfig",
    "DatabaseConfig",
    "DatabaseConnection",
    "ModelsConfig",
    "PipelineConfig",
    "HyperparametersConfig",
    "HeuristicsConfig",
    "PreprocessingConfig",
    "QwenConfig",
    "NERLabelsConfig",
    "PROJECT_ROOT",
    "getenv",
    "token_chunks",
]
