# src/config/hyperparameters.py
"""
Task-specific default hyperparameters for NER and Sentiment models.
"""

from __future__ import annotations

from typing import Dict, Any, List


class HyperparametersConfig:
    DEFAULT_NER_PARAMS: Dict[str, Any] = {
        "max_chunk_tokens": 450,
        "overlap_tokens": 100,
        "score_threshold": 0.60,
        "merge_entities": True,
        "deduplicate": True,
        "min_len_person": 2,
        "min_len_other": 3,
        "expand_short_entities": True,
        "expand_max_len": 3,
    }

    DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = {
        "max_chunk_tokens": 450,
        "overlap_tokens": 50,
        "aggregation": "mean_probs",
    }

    GLINER_LABELS: List[str] = [
        "person name",
        "organization or institution or government body",
        "geographic location or city or country",
        "date or time expression",
        "named event or armed conflict or political crisis",
        "commercial product or brand name",
        "sports competition or league or tournament",
    ]

    GLINER_LANGUAGE_THRESHOLDS: Dict[str, float] = {
        "ar": 0.6,
        "en": 0.6,
        "fr": 0.6,
    }
