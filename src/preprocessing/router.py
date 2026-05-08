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
    "remove_junk": True,
}

LATIN_NER_PARAMS: Dict[str, Any] = {
    "normalize_unicode": True,
    # Remove links/mentions/emails entirely (cleaner for NER)
    "standardize_social": False,
    "handle_hashtags": True,   # "#Ukraine" -> "Ukraine"
    "reduce_repetitions": True,
    "normalize_punct": True,
    "clean_twitter": True,
    "remove_junk": True,
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
    "remove_junk": True,
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

        # --- LANGUAGE DETECTION FALLBACK ---
        # If language is unknown, we still want to clean HTML and social noise 
        # to improve detection accuracy.
        if task == "lang_detect" or not lang:
            # 1) Use Latin preprocessor for basic URL/social removal
            text = self.lat.preprocess(text, **LATIN_LANG_DETECT_PARAMS)
            
            # 2) Remove Arabic-specific noise that hinders detection (diacritics/tatweel)
            text = self.ar.remove_diacritics(text)
            text = self.ar.remove_tatweel(text)
            
            # 3) Apply Arabic-specific keyword fixes (safe for any language)
            text = self.ar.fix_merged_keywords(text)
            
            # 3.5) Remove decorative noise and social spam
            text = self.ar.remove_decorative_noise(text)
            text = self.ar.remove_social_spam(text)
            
            # 4) Remove social-media noise before punct normalization
            text = self.ar.remove_social_noise(text)
            
            # 5) Collapse Arabic punctuation runs that the Latin preprocessor misses
            text = self.ar.normalize_punctuation(text)
            
            # 6) Apply final structural fixes (strip loose brackets, commas, quotes)
            text = self.ar.apply_final_structural_fixes(text)
            
            return text

        return self.lat.preprocess(text, **LATIN_LANG_DETECT_PARAMS)

    def normalize_entity(self, text: str, lang: str) -> str:
        lang = (lang or "").lower().strip()
        if lang == "ar":
            return self.ar.normalize_entity(text)
        # Default to Latin (EN/FR)
        return self.lat.normalize_entity(text)