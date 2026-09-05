# src/language_detection/__init__.py
"""
Language detection package providing fastText-based language identification
with Arabic dialect aggregation and script character validation.
"""

from __future__ import annotations

from .detector import (
    FastTextLanguageDetector,
    LanguageDetection,
    ARABIC_LANG_CODES,
    MIN_ARABIC_COMBINED_CONF,
    MIN_ARABIC_RATIO,
)

__all__ = [
    "FastTextLanguageDetector",
    "LanguageDetection",
    "ARABIC_LANG_CODES",
    "MIN_ARABIC_COMBINED_CONF",
    "MIN_ARABIC_RATIO",
]
