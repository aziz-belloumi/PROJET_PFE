# src/preprocessing.py

import re
import unicodedata
import html
from typing import Optional, Dict, Any
import logging


PREPROCESS_LANG_DETECT_PARAMS: Dict[str, Any] = {
    "remove_diacritics": False,
    "normalize_arabic": True,   # here means NFC only (safe)
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
    "normalize_arabic": True,   # NFC only (safe)
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
    "normalize_arabic": True,   # NFC only (safe)
    "remove_urls": True,
    "remove_emails": True,
    "remove_numbers": False,
    "remove_special": True,
    "remove_repeated": False,
    "remove_tatweel": True,
    "handle_hashtags": True,
    "fix_merged_libya": True,
}


class ArabicPreprocessor:
    # Arabic diacritics (tashkīl)
    ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
    # Arabic unicode ranges (single char)
    ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Preprocessor initialized")

    # ----------------------------
    # Shared helpers (safe for all languages)
    # ----------------------------
    def clean_html(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http[s]?://\S+", " ", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def normalize_unicode_nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text or "")

    # ----------------------------
    # Arabic-specific helpers
    # ----------------------------
    def remove_diacritics(self, text: str) -> str:
        return self.ARABIC_DIACRITICS.sub("", text)

    def remove_tatweel(self, text: str) -> str:
        return text.replace("\u0640", "")

    def remove_numbers(self, text: str) -> str:
        text = re.sub(r"[0-9]+", "", text)
        text = re.sub(r"[\u0660-\u0669]+", "", text)
        return text

    def remove_special_chars_ar(self, text: str, keep_arabic_punct: bool = True) -> str:
        """
        Arabic-focused filter:
        Keeps Arabic ranges + whitespace + digits + Arabic punctuation + basic punctuation.
        WARNING: This will REMOVE Latin letters (bad for EN/FR).
        """
        if keep_arabic_punct:
            pattern = r"[^\u0600-\u06FF\u0750-\u077F\s0-9،؛؟.!?]"
        else:
            pattern = r"[^\u0600-\u06FF\u0750-\u077F\s0-9]"
        return re.sub(pattern, " ", text)

    def remove_repeated_chars(self, text: str, max_repeat: int = 2) -> str:
        pattern = r"(.)\1{" + str(max_repeat) + r",}"
        replacement = r"\1" * max_repeat
        return re.sub(pattern, replacement, text)

    def fix_merged_libya(self, text: str) -> str:
        # Insert a space when 'ليبيا' is attached to previous Arabic char
        return re.sub(r"([\u0600-\u06FF\u0750-\u077F])ليبيا", r"\1 ليبيا", text)

    # ----------------------------
    # Arabic pipeline (existing behavior)
    # ----------------------------
    def preprocess(
        self,
        text: Optional[str],
        remove_diacritics: bool = True,
        normalize_arabic: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        remove_special: bool = True,
        remove_repeated: bool = False,
        remove_tatweel: bool = True,
        handle_hashtags: bool = False,
        fix_merged_libya: bool = False,
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self.clean_html(text)

        if normalize_arabic:
            text = self.normalize_unicode_nfc(text)

        if remove_urls:
            text = self.remove_urls(text)
        if remove_emails:
            text = self.remove_emails(text)

        if remove_tatweel:
            text = self.remove_tatweel(text)

        if remove_diacritics:
            text = self.remove_diacritics(text)

        if handle_hashtags:
            text = text.replace("#", " ").replace("_", " ")

        if remove_numbers:
            text = self.remove_numbers(text)

        if remove_special:
            text = self.remove_special_chars_ar(text, keep_arabic_punct=True)

        if remove_repeated:
            text = self.remove_repeated_chars(text, max_repeat=2)

        if fix_merged_libya:
            text = self.fix_merged_libya(text)

        return self.normalize_whitespace(text)

    def preprocess_for_lang_detect(self, text: Optional[str]) -> str:
        return self.preprocess(text, **PREPROCESS_LANG_DETECT_PARAMS)

    def preprocess_for_ner(self, text: str) -> str:
        return self.preprocess(text, **PREPROCESS_NER_PARAMS)

    def preprocess_for_sentiment(self, text: str) -> str:
        return self.preprocess(text, **PREPROCESS_SENTIMENT_PARAMS)

    # ----------------------------
    # Latin-friendly preprocessing (EN/FR)
    # ----------------------------
    def preprocess_latin_basic(self, text: Optional[str], handle_hashtags: bool = True) -> str:
        """
        Safe preprocessing for English/French:
        - keeps Latin characters and accents
        - removes HTML, URLs, emails
        - NFC normalization
        - hashtag '_' splitting optional
        - DOES NOT remove diacritics (French accents are important)
        - DOES NOT apply Arabic special-char filter
        """
        if not text or not isinstance(text, str):
            return ""

        text = self.clean_html(text)
        text = self.normalize_unicode_nfc(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)

        if handle_hashtags:
            text = text.replace("#", " ").replace("_", " ")

        return self.normalize_whitespace(text)

    def preprocess_for_ner_latin(self, text: Optional[str]) -> str:
        return self.preprocess_latin_basic(text, handle_hashtags=True)

    def preprocess_for_sentiment_latin(self, text: Optional[str]) -> str:
        return self.preprocess_latin_basic(text, handle_hashtags=True)

    # ----------------------------
    # Router (what main.py should use)
    # ----------------------------
    def preprocess_for_task(self, text: Optional[str], lang: str, task: str) -> str:
        """
        task: "lang_detect" | "ner" | "sentiment"
        lang: "ar" | "en" | "fr" | ...
        """
        lang = (lang or "").lower().strip()
        task = (task or "").lower().strip()

        if task == "lang_detect":
            # lang detect preprocessing is language-agnostic enough
            return self.preprocess_for_lang_detect(text)

        if lang == "ar":
            if task == "ner":
                return self.preprocess_for_ner(text or "")
            if task == "sentiment":
                return self.preprocess_for_sentiment(text or "")
            return self.preprocess_for_lang_detect(text)

        # Non-Arabic: use latin-safe preprocessing
        if task == "ner":
            return self.preprocess_for_ner_latin(text)
        if task == "sentiment":
            return self.preprocess_for_sentiment_latin(text)

        return self.preprocess_latin_basic(text)