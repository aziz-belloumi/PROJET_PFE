import re
import unicodedata
import html
from typing import Optional
import logging


class LatinPreprocessor:
    """
    Latin (EN/FR) preprocessor with two modes:
      - standardize_social=True  -> replace URLs/mentions/emails with placeholders (http/@user/email)
      - standardize_social=False -> REMOVE URLs/mentions/emails entirely (cleaner for NER/topic)
    """

    # --- Precompiled patterns ---
    _RE_HTML_TAG = re.compile(r"<[^>]+>")
    _RE_WS = re.compile(r"\s+")

    _RE_RT = re.compile(r"^RT\s+", flags=re.IGNORECASE)
    _RE_VIA_MENTION = re.compile(r"\bvia\s+@[A-Za-z0-9_]+\b", flags=re.IGNORECASE)

    _RE_URL = re.compile(r"\bhttps?://\S+", flags=re.IGNORECASE)
    _RE_WWW = re.compile(r"\bwww\.\S+", flags=re.IGNORECASE)

    _RE_EMAIL = re.compile(r"\b\S+@\S+\b")
    _RE_MENTION = re.compile(r"@[A-Za-z0-9_]+")

    _RE_TCO = re.compile(r"\bt\.co/\S+", flags=re.IGNORECASE)
    _RE_PIC_TW = re.compile(r"\bpic\.twitter\.com/\S+", flags=re.IGNORECASE)

    # Any remaining non-space token starting with htt/http/https (kills: htt…, http…, https…, http..., https.., etc.)
    _RE_HTTP_FRAGMENT_TOKEN = re.compile(r"(?:(?<=\s)|^)(?:https?|htt)\S+", flags=re.IGNORECASE)

    # Orphan punctuation after removals (e.g., "@user:" removed -> ":" at start)
    _RE_LEADING_ORPHAN_PUNCT = re.compile(r"^\s*[:\-–—]+\s*")

    # Trailing punctuation left after deletions (e.g., "fixed it for you,")
    _RE_TRAILING_DANGLING_PUNCT = re.compile(r"\s*[,;:–—-]+\s*$")

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def clean_html(self, text: str) -> str:
        text = self._RE_HTML_TAG.sub(" ", text)
        return html.unescape(text)

    def normalize_unicode_nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text or "")

    def normalize_whitespace(self, text: str) -> str:
        return self._RE_WS.sub(" ", text).strip()

    def normalize_punctuation(self, text: str) -> str:
        text = re.sub(r'[«»“”]', '"', text)
        text = re.sub(r"[‘’]", "'", text)
        text = re.sub(r"[–—]", "-", text)
        return text

    def reduce_repetition(self, text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    # -------- Added (2): split long CamelCase tokens (useful for hashtags like UkraineRussianWar) --------
    def _split_camelcase_long_tokens(self, text: str, min_len: int = 10) -> str:
        """
        Splits long CamelCase-ish tokens into spaced words:
          - "UkraineRussianWar" -> "Ukraine Russian War"
          - "NutmegofTheWeek"   -> "Nutmegof The Week" (best-effort)

        Heuristics to reduce harm:
          - only tokens with length >= min_len
          - skip tokens containing digits
          - skip fully upper or fully lower tokens
        """
        out_parts = []
        for tok in (text or "").split():
            if (
                len(tok) < min_len
                or any(ch.isdigit() for ch in tok)
                or tok.isupper()
                or tok.islower()
                or not re.search(r"[A-Z]", tok)
                or not re.search(r"[a-z]", tok)
            ):
                out_parts.append(tok)
                continue

            # Split boundaries:
            # 1) lower->upper: "rW" in "RussianWar"
            tok = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
            # 2) acronym->Word: "USIntel" -> "US Intel"
            tok = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", tok)

            out_parts.append(tok)

        return " ".join(out_parts)

    # -------- Added (1): remove dangling punctuation after deletions --------
    def _strip_dangling_punctuation(self, text: str) -> str:
        text = self._RE_LEADING_ORPHAN_PUNCT.sub("", text)
        text = self._RE_TRAILING_DANGLING_PUNCT.sub("", text)
        return text

    def clean_twitter_noise(self, text: str) -> str:
        text = self._RE_RT.sub("", text)
        text = self._RE_VIA_MENTION.sub(" ", text)

        # Remove obvious url fragments like http... / https.. (with dots)
        text = re.sub(r"\bhttps?\.{2,}\S*", " ", text, flags=re.IGNORECASE)

        # Remove ellipsis-style fragments: htt… / http… / https…
        text = re.sub(r"\bhtt(?:ps?)?…\b", " ", text, flags=re.IGNORECASE)

        return text

    def standardize_entities(self, text: str) -> str:
        text = self._RE_URL.sub("http", text)
        text = self._RE_WWW.sub("http", text)
        text = self._RE_MENTION.sub("@user", text)
        text = self._RE_EMAIL.sub("email", text)
        return text

    def remove_links_mentions_emails(self, text: str) -> str:
        # Full URLs + www
        text = self._RE_URL.sub(" ", text)
        text = self._RE_WWW.sub(" ", text)

        # Shorteners without scheme
        text = self._RE_TCO.sub(" ", text)
        text = self._RE_PIC_TW.sub(" ", text)

        # Mentions + emails
        text = self._RE_MENTION.sub(" ", text)
        text = self._RE_EMAIL.sub(" ", text)

        # Any remaining http/https/htt fragment tokens
        text = self._RE_HTTP_FRAGMENT_TOKEN.sub(" ", text)

        # Remove punctuation left behind by deletions
        text = self._strip_dangling_punctuation(text)

        return text

    def preprocess(
        self,
        text: Optional[str],
        normalize_unicode: bool = True,
        standardize_social: bool = False,  # True -> @user/http/email; False -> remove links/mentions/emails entirely
        handle_hashtags: bool = False,     # True -> strip #/_ (good for NER/topic)
        reduce_repetitions: bool = False,  # True -> soooo -> soo (good for sentiment)
        normalize_punct: bool = True,
        clean_twitter: bool = False,       # remove RT / via / truncated url noise
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        # 1) Basic cleaning
        text = self.clean_html(text)

        if normalize_unicode:
            text = self.normalize_unicode_nfc(text)

        if normalize_punct:
            text = self.normalize_punctuation(text)

        # 2) Twitter-specific cleaning
        if clean_twitter:
            text = self.clean_twitter_noise(text)

        # 3) Social entity management
        if standardize_social:
            # For CardiffNLP sentiment models (keep placeholders)
            text = self.standardize_entities(text)
        else:
            # For NER/topic/lang-detect cleaning (remove entirely)
            text = self.remove_links_mentions_emails(text)

        # 4) Hashtags
        if handle_hashtags:
            # "#UkraineRussianWar" -> "UkraineRussianWar" (then we split CamelCase below)
            text = text.replace("#", " ").replace("_", " ")
            # Added (2): split long CamelCase tokens (best-effort)
            text = self._split_camelcase_long_tokens(text, min_len=10)

        # 5) Repetition reduction
        if reduce_repetitions:
            text = self.reduce_repetition(text)

        # Final cleanup (Added 1): strip trailing commas/colons/etc after all ops
        text = self.normalize_whitespace(text)
        text = self._strip_dangling_punctuation(text)
        return self.normalize_whitespace(text)

    def preprocess_for_lang_detect(self, text: str) -> str:
        return self.preprocess(text, standardize_social=False, handle_hashtags=False)

    def normalize_entity(self, text: str) -> str:
        if not text:
            return ""
        
        # 1) Basic cleanup
        s = self.normalize_unicode_nfc(text)
        s = self.remove_links_mentions_emails(s)
        
        # 2) Punctuation filter (keep Latin/Digits/Space)
        # Note: We keep some accents for French via NFC + regex but usually for normalization 
        # we want to be quite aggressive. Here we keep standard Latin.
        s = re.sub(r"[^A-Za-z0-9\sàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ]", " ", s)
        
        return self.normalize_whitespace(s).lower()