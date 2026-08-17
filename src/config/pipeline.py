# src/config/pipeline.py
"""
Pipeline execution toggles, sampling sizes, and quality gate thresholds.
"""

from __future__ import annotations

from .base import getenv


class PipelineConfig:
    SAMPLE_SIZE = getenv("SAMPLE_SIZE", 100, int)

    RUN_NER = getenv("RUN_NER", False, bool)
    RUN_SENTIMENT = getenv("RUN_SENTIMENT", False, bool)
    RUN_TOPIC = getenv("RUN_TOPIC", False, bool)
    RUN_QWEN = getenv("RUN_QWEN", True, bool)
    GENERATE_REPORTS = getenv("GENERATE_REPORTS", False, bool)

    LANG_THRESHOLD = getenv("LANG_THRESHOLD", 0.51, float)

    TOPIC_MIN_TEXT_CHARS = getenv("TOPIC_MIN_TEXT_CHARS", 30, int)
    TOPIC_MIN_TEXT_WORDS = getenv("TOPIC_MIN_TEXT_WORDS", 5, int)
