import re
import unicodedata
import html
from typing import Optional
import logging


class ArabicPreprocessor:
    ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    # Shared helpers
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

    # Arabic-specific helpers
    def remove_diacritics(self, text: str) -> str:
        return self.ARABIC_DIACRITICS.sub("", text)

    def remove_tatweel(self, text: str) -> str:
        return text.replace("\u0640", "")

    def remove_numbers(self, text: str) -> str:
        text = re.sub(r"[0-9]+", "", text)
        text = re.sub(r"[\u0660-\u0669]+", "", text)
        return text

    def remove_special_chars_ar(self, text: str, keep_arabic_punct: bool = True) -> str:
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
        return re.sub(r"([\u0600-\u06FF\u0750-\u077F])ليبيا", r"\1 ليبيا", text)

    def normalize_entity(self, text: str) -> str:
        if not text:
            return ""
        
        # 1) Basic cleanup
        s = self.normalize_unicode_nfc(text)
        s = self.remove_diacritics(s)
        s = self.remove_tatweel(s)
        
        # 2) Arabic-specific variants
        s = re.sub(r"[أإآ]", "ا", s)
        s = re.sub(r"ة", "ه", s)
        s = re.sub(r"ى", "ي", s)

        # 3) Final character filter (keep Arabic/Latin/Digits)
        s = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s]", " ", s)
        
        return self.normalize_whitespace(s).lower()

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