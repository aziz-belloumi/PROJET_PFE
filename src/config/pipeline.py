from __future__ import annotations


class PipelineConfig:
    # Sampling size (batch limit)
    SAMPLE_SIZE: int = 1000

    # Pipeline task execution toggles
    RUN_NER: bool = True
    RUN_SENTIMENT: bool = True
    RUN_TOPIC: bool = True
    RUN_QWEN: bool = False

    # Language confidence threshold
    LANG_THRESHOLD: float = 0.51

    # Topic classification minimum content filters
    TOPIC_MIN_TEXT_CHARS: int = 30
    TOPIC_MIN_TEXT_WORDS: int = 5

    # Analytics options
    EXPORT_ANALYTICS_CSV: bool = True
