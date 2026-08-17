from typing import Optional, Dict, Any
import logging

from src.config import Config
from .arabic import ArabicPreprocessor
from .latin import LatinPreprocessor


# Presets alias to centralized Config
PREPROCESS_LANG_DETECT_PARAMS: Dict[str, Any] = Config.PREPROCESS_LANG_DETECT_PARAMS
LATIN_LANG_DETECT_PARAMS: Dict[str, Any] = Config.LATIN_LANG_DETECT_PARAMS


class PreprocessRouter:
    """
    Single entry point:
        preproc.preprocess(text, lang, task)
    Only task supported in this project: "lang_detect" (default).
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
            return self.ar.preprocess(text, **PREPROCESS_LANG_DETECT_PARAMS)

        # --- LANGUAGE DETECTION FALLBACK ---
        # If language is unknown, we still want to clean HTML and social noise
        # to improve detection accuracy.
        # 1) Use Latin preprocessor for basic URL/social removal
        text = self.lat.preprocess(text, **LATIN_LANG_DETECT_PARAMS)

        # 2) Remove Arabic-specific noise that hinders detection (diacritics/tatweel)
        text = self.ar.remove_diacritics(text)
        text = self.ar.remove_tatweel(text)

        # 3) Apply Arabic-specific keyword fixes (safe for any language)
        text = self.ar.fix_merged_keywords(text)

        # 4) Remove decorative noise and social spam
        text = self.ar.remove_decorative_noise(text)
        text = self.ar.remove_social_spam(text)

        # 5) Remove social-media noise before punct normalization
        text = self.ar.remove_social_noise(text)

        # 6) Collapse Arabic punctuation runs that the Latin preprocessor misses
        text = self.ar.normalize_punctuation(text)

        # 7) Apply final structural fixes (strip loose brackets, commas, quotes)
        text = self.ar.apply_final_structural_fixes(text)

        return text