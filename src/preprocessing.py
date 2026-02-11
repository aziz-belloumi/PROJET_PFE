import re
import unicodedata
import html
from typing import Optional, Dict, Any
import logging


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
    "fix_merged_libya": True,   # NEW
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
    "fix_merged_libya": True,   # NEW
}

PREPROCESS_KEYWORDS_PARAMS: Dict[str, Any] = {
    "remove_diacritics": True,
    "normalize_arabic": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_numbers": True,
    "remove_special": True,
    "remove_repeated": True,
    "remove_tatweel": True,
    "handle_hashtags": False,
    "fix_merged_libya": True,
}


class ArabicPreprocessor:
    # Arabic diacritics (tashkīl)
    ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
    # Arabic unicode ranges (single char)
    ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Arabic preprocessor initialized")

    def remove_diacritics(self, text: str) -> str:
        return self.ARABIC_DIACRITICS.sub("", text)

    def remove_tatweel(self, text: str) -> str:
        # \u0640 is the Arabic Tatweel (Kashida): ـ
        return text.replace("\u0640", "")

    def normalize_arabic(self, text: str) -> str:
        if not text:
            return ""
        # Unicode NFC normalization (safe for all tasks; does not collapse Alef forms)
        return unicodedata.normalize("NFC", text)

    def clean_html(self, text: str) -> str:
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Decode entities (&nbsp; &#123; etc.)
        return html.unescape(text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http[s]?://\S+", " ", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+", " ", text)

    def remove_numbers(self, text: str) -> str:
        # Latin numbers
        text = re.sub(r"[0-9]+", "", text)
        # Arabic-Indic numbers
        text = re.sub(r"[\u0660-\u0669]+", "", text)
        return text

    def remove_special_chars(self, text: str, keep_arabic_punct: bool = True) -> str:
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
        """
        We only insert a space when 'ليبيا' is directly attached to a preceding Arabic character
        """
        # preceding char must be Arabic (not whitespace/punct)
        return re.sub(r"([\u0600-\u06FF\u0750-\u077F])ليبيا", r"\1 ليبيا", text)

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
        fix_merged_libya: bool = False,   # NEW
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self.clean_html(text)

        if normalize_arabic:
            text = self.normalize_arabic(text)

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
            text = self.remove_special_chars(text, keep_arabic_punct=True)

        if remove_repeated:
            text = self.remove_repeated_chars(text, max_repeat=2)

        # NEW: fix merged ليبيا near the end (after most cleaning)
        if fix_merged_libya:
            text = self.fix_merged_libya(text)

        return self.normalize_whitespace(text)

    def preprocess_for_lang_detect(self, text: Optional[str]) -> str:
        return self.preprocess(text, **PREPROCESS_LANG_DETECT_PARAMS)

    def preprocess_for_keywords(self, text: str) -> str:
        return self.preprocess(text, **PREPROCESS_KEYWORDS_PARAMS)

    def preprocess_for_ner(self, text: str) -> str:
        return self.preprocess(text, **PREPROCESS_NER_PARAMS)

    def preprocess_for_sentiment(self, text: str) -> str:
        return self.preprocess(text, **PREPROCESS_SENTIMENT_PARAMS)

    def arabic_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        arabic_chars = len(self.ARABIC_CHAR.findall(text))
        total_chars = len(re.findall(r"\S", text))
        return (arabic_chars / total_chars) if total_chars > 0 else 0.0

    def get_statistics(self, text: str) -> dict:
        if not text:
            return {
                "length": 0,
                "words": 0,
                "arabic_chars": 0,
                "total_chars": 0,
                "arabic_ratio": 0.0,
                "is_arabic": False,
            }

        ratio = self.arabic_ratio(text)
        arabic_chars = len(self.ARABIC_CHAR.findall(text))
        total_chars = len(re.findall(r"\S", text))

        return {
            "length": len(text),
            "words": len(text.split()),
            "arabic_chars": arabic_chars,
            "total_chars": total_chars,
            "arabic_ratio": ratio,
            "is_arabic": ratio >= 0.5,
        }