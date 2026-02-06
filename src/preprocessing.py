# Specialized preprocessing for Arabic text including:
# HTML cleaning
# Diacritics removal
# Letter normalization
# Special character filtering
# Whitespace normalization


import re
import unicodedata
import html
from typing import Optional
import logging


class ArabicPreprocessor:
    # Diacritics (tachkīl)
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')
    ARABIC_CHAR = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')        # single char
    ARABIC_LETTERS = re.compile(r'[\u0600-\u06FF\u0750-\u077F]+')    # chunks (words)

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Arabic preprocessor initialized")

    def remove_diacritics(self, text: str) -> str:
        return self.ARABIC_DIACRITICS.sub('', text)

    def normalize_arabic(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        #text = text.replace('ى', 'ي')
        return text

    def clean_html(self, text: str) -> str:
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode entities (&nbsp; &#123; etc.)
        text = html.unescape(text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def remove_urls(self, text: str) -> str:
        return re.sub(r'http[s]?://\S+', ' ', text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r'\S+@\S+', ' ', text)

    def remove_numbers(self, text: str) -> str:
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[\u0660-\u0669]+', '', text)
        return text

    def remove_special_chars(self, text: str, keep_arabic_punct: bool = True) -> str:
        if keep_arabic_punct:
            pattern = r'[^\u0600-\u06FF\u0750-\u077F\s0-9،؛؟.!?]'
        else:
            pattern = r'[^\u0600-\u06FF\u0750-\u077F\s0-9]'
        return re.sub(pattern, ' ', text)

    def remove_repeated_chars(self, text: str, max_repeat: int = 2) -> str:
        pattern = r'(.)\1{' + str(max_repeat) + ',}'
        replacement = r'\1' * max_repeat
        return re.sub(pattern, replacement, text)

    def arabic_ratio(self, text: str) -> float:
        # Correct Arabic character ratio (char-level, not chunk-level)
        if not text:
            return 0.0
        arabic_chars = len(self.ARABIC_CHAR.findall(text))
        total_chars = len(re.findall(r'\S', text))
        return (arabic_chars / total_chars) if total_chars > 0 else 0.0

    def is_arabic(self, text: str, threshold: float = 0.5) -> bool:
        return self.arabic_ratio(text) >= threshold

    def preprocess(
        self,
        text: Optional[str],
        remove_diacritics: bool = True,
        normalize: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        remove_special: bool = True,
        remove_repeated: bool = False
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = self.clean_html(text)

        if remove_urls:
            text = self.remove_urls(text)
        if remove_emails:
            text = self.remove_emails(text)
        if remove_diacritics:
            text = self.remove_diacritics(text)
        if normalize:
            text = self.normalize_arabic(text)
        if remove_numbers:
            text = self.remove_numbers(text)
        if remove_special:
            text = self.remove_special_chars(text, keep_arabic_punct=True)
        if remove_repeated:
            text = self.remove_repeated_chars(text, max_repeat=2)

        return self.normalize_whitespace(text)

    def preprocess_for_lang_detect(self, text: Optional[str]) -> str:
        # Light preprocessing (recommended before laguage detection)
        if not text or not isinstance(text, str):
            return ""
        text = self.clean_html(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        return self.normalize_whitespace(text)

    def preprocess_for_keywords(self, text: str) -> str:
        return self.preprocess(
            text,
            remove_numbers=True,
            remove_special=True,
            remove_repeated=True
        )

    def preprocess_for_ner(self, text: str) -> str:
        return self.preprocess(
            text,
            remove_numbers=False,
            remove_special=True,
            remove_repeated=False
        )

    def get_statistics(self, text: str) -> dict:
        if not text:
            return {"length": 0, "words": 0, "arabic_chars": 0, "total_chars": 0, "arabic_ratio": 0.0, "is_arabic": False}

        ratio = self.arabic_ratio(text)
        arabic_chars = len(self.ARABIC_CHAR.findall(text))
        total_chars = len(re.findall(r'\S', text))

        return {
            "length": len(text),
            "words": len(text.split()),
            "arabic_chars": arabic_chars,
            "total_chars": total_chars,
            "arabic_ratio": ratio,
            "is_arabic": ratio >= 0.5
        }