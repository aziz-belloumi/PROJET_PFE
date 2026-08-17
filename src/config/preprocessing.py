# src/config/preprocessing.py
"""
Text preprocessing presets for language detection, NER, Sentiment, and Topic tasks.
"""

from __future__ import annotations

from typing import Dict, Any


class PreprocessingConfig:
    PREPROCESS_LANG_DETECT_PARAMS: Dict[str, Any] = {
        "remove_diacritics": False,
        "normalize_arabic": True,
        "remove_urls": True,
        "remove_emails": True,
        "remove_numbers": False,
        "remove_special": True,
        "remove_repeated": False,
        "remove_tatweel": True,
        "handle_hashtags": True,
        "fix_merged_keywords": True,
        "normalize_punct": True,
        "remove_social_noise": True,
    }

    PREPROCESS_NER_PARAMS: Dict[str, Any] = {
        "remove_diacritics": True,
        "normalize_arabic": True,
        "remove_urls": True,
        "remove_emails": True,
        "remove_numbers": False,
        "remove_special": True,
        "remove_repeated": False,
        "remove_tatweel": True,
        "handle_hashtags": True,
        "fix_merged_keywords": True,
        "normalize_punct": True,
        "remove_social_noise": True,
    }

    PREPROCESS_SENTIMENT_PARAMS: Dict[str, Any] = {
        "remove_diacritics": True,
        "normalize_arabic": True,
        "remove_urls": True,
        "remove_emails": True,
        "remove_numbers": False,
        "remove_special": True,
        "remove_repeated": False,
        "remove_tatweel": True,
        "handle_hashtags": True,
        "fix_merged_keywords": True,
        "normalize_punct": True,
        "remove_social_noise": True,
    }

    LATIN_LANG_DETECT_PARAMS: Dict[str, Any] = {
        "normalize_unicode": True,
        "standardize_social": False,
        "handle_hashtags": False,
        "reduce_repetitions": False,
        "normalize_punct": True,
        "clean_twitter": True,
        "remove_junk": True,
    }

    LATIN_NER_PARAMS: Dict[str, Any] = {
        "normalize_unicode": True,
        "standardize_social": False,
        "handle_hashtags": True,
        "reduce_repetitions": True,
        "normalize_punct": True,
        "clean_twitter": True,
        "remove_junk": True,
    }

    LATIN_SENTIMENT_PARAMS: Dict[str, Any] = {
        "normalize_unicode": True,
        "standardize_social": True,
        "handle_hashtags": True,
        "reduce_repetitions": True,
        "normalize_punct": True,
        "clean_twitter": True,
    }

    LATIN_TOPIC_PARAMS: Dict[str, Any] = {
        "normalize_unicode": True,
        "standardize_social": False,
        "handle_hashtags": True,
        "reduce_repetitions": False,
        "normalize_punct": True,
        "clean_twitter": True,
        "remove_junk": True,
    }
