# src/config/models.py
"""
NLP model paths, registry mappings, and unique model IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from .base import BaseConfig


class ModelsConfig:
    # Fine-tuned model directory paths
    FINETUNED_SENTIMENT_MODELS: Dict[str, str] = {
        "ar": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "ARABIC SENTIMENT"),
        "en": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "ENGLISH SENTIMENT"),
        "fr": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "FRENSH SENTIMENT"),
    }

    FINETUNED_TOPIC_MODELS: Dict[str, str] = {
        "ar": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "ARABIC TOPIC"),
        "en": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "ENGLSIH TOPIC"),
        "fr": str(BaseConfig._PROJECT_PATH / "finetuned_models" / "FRENSH TOPIC"),
    }

    # Active model references
    GLINER_MODEL = "urchade/gliner_multi-v2.1"
    QWEN_BENCHMARK_MODEL = "qwen2.5:7b"

    # ── NER model hub identifiers ──────────────────────────────────────────────
    ARABERT_NER_MODEL  = "aubmindlab/bert-base-arabertv02-ner"
    BERT_EN_NER_MODEL  = "dslim/bert-base-NER"
    CAMEMBERT_NER_MODEL = "Jean-Baptiste/camembert-ner"

    # Single source of truth: Unique numeric model IDs across all tables
    MODEL_ID_MAP: Dict[str, int] = {
        # NER Models (0-7)
        "hatmimoha/arabic-ner": 0,               # legacy / kept for backward compat
        "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner": 1,  # legacy
        "dslim/bert-base-NER": 2,
        "Jean-Baptiste/camembert-ner": 3,
        "urchade/gliner_multi-v2.1": 4,
        "aubmindlab/bert-base-arabertv02-ner": 5,
        # IDs 6-7 reserved for future NER models

        # Sentiment Models (8-10)
        "ar_sentiment_ft": 8,
        "en_sentiment_ft": 9,
        "fr_sentiment_ft": 10,

        # Topic Models (11-13)
        "ar_topic_ft": 11,
        "en_topic_ft": 12,
        "fr_topic_ft": 13,

        # Qwen / LLM Model (14)
        "qwen2.5:7b": 14,
    }

    # Task key mappings for benchmark recording
    SENT_MODEL_ID_KEYS: Dict[str, str] = {
        "ar": "ar_sentiment_ft",
        "en": "en_sentiment_ft",
        "fr": "fr_sentiment_ft",
    }

    TOPIC_MODEL_ID_KEYS: Dict[str, str] = {
        "ar": "ar_topic_ft",
        "en": "en_topic_ft",
        "fr": "fr_topic_ft",
    }

    # Active model registry configurations for the pipeline.
    # Each language runs its dedicated BERT-based NER model first, then GLiNER
    # as a multilingual second pass (catches entity types the BERT model may miss).
    NER_MODELS_BY_LANG: Dict[str, List[Tuple[int, str]]] = {
        "ar": [
            (MODEL_ID_MAP[ARABERT_NER_MODEL],  ARABERT_NER_MODEL),   # AraBERT-NER (ar)
            (MODEL_ID_MAP[GLINER_MODEL],        GLINER_MODEL),        # GLiNER multilingual
        ],
        "en": [
            (MODEL_ID_MAP[BERT_EN_NER_MODEL],  BERT_EN_NER_MODEL),   # BERT-base-NER (en)
            (MODEL_ID_MAP[GLINER_MODEL],        GLINER_MODEL),        # GLiNER multilingual
        ],
        "fr": [
            (MODEL_ID_MAP[CAMEMBERT_NER_MODEL], CAMEMBERT_NER_MODEL), # CamemBERT-NER (fr)
            (MODEL_ID_MAP[GLINER_MODEL],        GLINER_MODEL),        # GLiNER multilingual
        ],
    }

    SENT_MODELS_BY_LANG: Dict[str, List[Tuple[int, str]]] = {
        "ar": [(MODEL_ID_MAP["ar_sentiment_ft"], FINETUNED_SENTIMENT_MODELS["ar"])],
        "en": [(MODEL_ID_MAP["en_sentiment_ft"], FINETUNED_SENTIMENT_MODELS["en"])],
        "fr": [(MODEL_ID_MAP["fr_sentiment_ft"], FINETUNED_SENTIMENT_MODELS["fr"])],
    }

    TOPIC_EXTRACTORS: List[Tuple[int, str, str]] = [
        (MODEL_ID_MAP["ar_topic_ft"], "transformer", "ar"),
        (MODEL_ID_MAP["en_topic_ft"], "transformer", "en"),
        (MODEL_ID_MAP["fr_topic_ft"], "transformer", "fr"),
    ]
