from typing import Optional, Dict, Any
import logging

from .arabic import ArabicPreprocessor
from .latin import LatinPreprocessor


# =========================================================
# ARABIC PRESETS
# =========================================================
PREPROCESS_LANG_DETECT_PARAMS: Dict[str, Any] = {
    "remove_diacritics": False,
    "normalize_arabic": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_numbers": False,
    "remove_special": False,
    "remove_repeated": False,
    "remove_tatweel": False,
    "handle_hashtags": False,
    "fix_merged_libya": False,
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
    "fix_merged_libya": True,
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
    "fix_merged_libya": True,
}

# =========================================================
# LATIN PRESETS (EN / FR)
# =========================================================
LATIN_LANG_DETECT_PARAMS: Dict[str, Any] = {
    "normalize_unicode": True,
    # With your updated LatinPreprocessor:
    # standardize_social=False => REMOVE links/mentions/emails entirely
    "standardize_social": False,
    "handle_hashtags": False,
    "reduce_repetitions": False,
    "normalize_punct": True,
    "clean_twitter": True,
}

LATIN_NER_PARAMS: Dict[str, Any] = {
    "normalize_unicode": True,
    # Remove links/mentions/emails entirely (cleaner for NER)
    "standardize_social": False,
    "handle_hashtags": True,   # "#Ukraine" -> "Ukraine"
    "reduce_repetitions": True,
    "normalize_punct": True,
    "clean_twitter": True,
}

LATIN_SENTIMENT_PARAMS: Dict[str, Any] = {
    "normalize_unicode": True,
    # Keep placeholders @user/http/email (CardiffNLP expects this)
    "standardize_social": True,
    "handle_hashtags": True,  # keep # for sentiment context
    "reduce_repetitions": True,
    "normalize_punct": True,
    "clean_twitter": True,
}

LATIN_TOPIC_PARAMS: Dict[str, Any] = {
    "normalize_unicode": True,
    # Remove links/mentions/emails entirely for topic
    "standardize_social": False,
    # RECOMMENDED UPDATE: strip hashtags for topic as well
    "handle_hashtags": True,
    "reduce_repetitions": False,
    "normalize_punct": True,
    "clean_twitter": True,
}


class PreprocessRouter:
    """
    Single entry point:
        preproc.preprocess(text, lang, task)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.ar = ArabicPreprocessor(logger=self.logger)
        self.lat = LatinPreprocessor(logger=self.logger)
        self.logger.info("PreprocessRouter initialized (Arabic + Latin)")

    def preprocess(self, text: Optional[str], lang: str, task: str) -> str:
        lang = (lang or "").lower().strip()
        task = (task or "").lower().strip()

        # --- ARABIC ---
        if lang == "ar":
            if task == "ner":
                return self.ar.preprocess(text, **PREPROCESS_NER_PARAMS)
            if task == "sentiment":
                return self.ar.preprocess(text, **PREPROCESS_SENTIMENT_PARAMS)
            if task == "topic":
                return self.ar.preprocess(text, **PREPROCESS_SENTIMENT_PARAMS)
            return self.ar.preprocess(text, **PREPROCESS_LANG_DETECT_PARAMS)

        # --- LATIN (EN/FR) ---
        if task == "ner":
            return self.lat.preprocess(text, **LATIN_NER_PARAMS)
        if task == "sentiment":
            return self.lat.preprocess(text, **LATIN_SENTIMENT_PARAMS)
        if task == "topic":
            return self.lat.preprocess(text, **LATIN_TOPIC_PARAMS)

        return self.lat.preprocess(text, **LATIN_LANG_DETECT_PARAMS)

    def normalize_entity(self, text: str, lang: str) -> str:
        lang = (lang or "").lower().strip()
        if lang == "ar":
            return self.ar.normalize_entity(text)
        # Default to Latin (EN/FR)
        return self.lat.normalize_entity(text)