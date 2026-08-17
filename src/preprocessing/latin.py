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

    _RE_RT_ANYWHERE = re.compile(r"\bRT\b\s*", flags=re.IGNORECASE)
    _RE_UNOFFICIAL = re.compile(r"^\s*Unofficial\s*:\s*", flags=re.IGNORECASE)
    _RE_KEYWORDS = re.compile(r"\b(?:DW|ICYMI|Official|Govt)\s*:\s*", flags=re.IGNORECASE)
    _RE_ATTACHED_MORNING_MY = re.compile(r"\bmorningMy\b", flags=re.IGNORECASE)
    _RE_HEBREW_NOISE = re.compile(r"פ\s*[:\s]\s*")
    _RE_VIA_MENTION = re.compile(r"\bvia\s+@[A-Za-z0-9_]+\b", flags=re.IGNORECASE)

    _RE_URL = re.compile(r"https?://\S+", flags=re.IGNORECASE)
    _RE_WWW = re.compile(r"www\.\S+", flags=re.IGNORECASE)

    _RE_EMAIL = re.compile(r"\S+@\S+")
    _RE_MENTION = re.compile(r"@[A-Za-z0-9_]+")

    _RE_TCO = re.compile(r"t\.co/\S+", flags=re.IGNORECASE)
    _RE_PIC_TW = re.compile(r"pic\.twitter\.com/\S+", flags=re.IGNORECASE)

    # Any remaining non-space token starting with htt/http/https
    _RE_HTTP_FRAGMENT_TOKEN = re.compile(r"(?:https?|htt)\S+", flags=re.IGNORECASE)

    # Uncommon Latin symbols often used in junk/corrupted text
    _RE_LATIN_SYMBOLS = re.compile(r"[µ³±¬†‡•·¶§©®™¤¦¨¯´¸¿¡¾¼½÷]+")

    # Ported from Arabic or extended for Latin
    _RE_REPLACEMENT_CHAR = re.compile(r"\uFFFD+")
    _RE_BLOCK_ELEMENTS = re.compile(r"[░▒▓█▌]+")
    _RE_BOX_DRAWING = re.compile(r"[─│┌┐└┘├┤┬┴┼╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬]+")
    _RE_MISC_SYMBOLS = re.compile(
        r"[↓↑⇓⇑←→⇐⇒↔⇔"
        r"♦♠♣♥♡♢♤♧"
        r"◘◙◦◆◇○●◎◐◑◒◓"
        r"☻☺☹"
        r"✅✓✔✗✘✕✖"
        r"★☆✦✧✩✪✫✬✭✮✯✰"
        r"¤§†‡‼‽"
        r"‹›«»"
        r"°•·‥…"
        r"▀▁▂▃▄▅▆▇▉▊▋▍▎▏▐"
        r"┐┘└┌"
        r"]+"
    )
    _RE_UNICODE_JUNK = re.compile(
        r"[\u2500-\u257F"    # Box Drawing
        r"\u2580-\u259F"     # Block Elements
        r"\u25A0-\u25FF"     # Geometric Shapes
        r"\u2600-\u26FF"     # Miscellaneous Symbols
        r"\u2700-\u27BF"     # Dingbats
        r"\u2190-\u21FF"     # Arrows
        r"\u2B00-\u2BFF"     # Misc Symbols and Arrows
        r"\uFFF0-\uFFFD"     # Specials (replacement chars)
        r"\u2300-\u23FF"     # Misc Technical
        r"]+"
    )

    # Orphan punctuation after removals
    _RE_LEADING_ORPHAN_PUNCT = re.compile(r"^\s*[:\-–—\(\[\{\"'«„\u201C\u2018]+\s*")

    # Trailing punctuation
    _RE_TRAILING_DANGLING_PUNCT = re.compile(r"\s*[,;–—\-\(\[\{\"'»\u201D\u2019:]+\s*$")

    _RE_JUNK_PATTERNS = re.compile(
        r"^\s*J ai publié une nouvelle photo sur Facebook\s*$"
        r"|^\s*J ai ajouté une nouvelle photo sur Facebook\s*$"
        r"|^\s*Offre d emploi à temps choisi\s*:"
        r"|^\s*L excellence dentaire à la portée de toute la famille\b"
        r"|^\s*Oriflame\b.*?contactez"
        r"|^\s*Devenir un membre Oriflame\b"
        r"|^\s*Je cherche des personnes motivée\b"
        r"|^\s*Bienvenue sur la page officielle de\b"
        r"|^\s*Like plzz\b"
        r"|^\s*Add Here\b"
        r"|^\s*MAX \.\s*:\s*\.\s*!\s*\.\s*!"
        r"|^\s*Dear\s*&\s*:\s*F\s*YOU\s*ALL"
        r"|For all your projects and Construction Materials needed"
        r"|Trois petits points: Magnifique!"
        r"|Màhmmèd\s+Kàwildé"
        r"|Welcome\s+\[Objectif\b"
        r"|! good bye ! Au revoir ! ciao"
        r"|OfficiƋl\s+fƋc℮book"
        r"|O\.o\b"
        r"|Studio\s+Meublé\s+Tunis"
        r"|\d{6},to people who wants to learn English language.*?\w+\.\w+"
        , flags=re.IGNORECASE | re.DOTALL
    )

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def clean_html(self, text: str) -> str:
        text = self._RE_HTML_TAG.sub(" ", text)
        return html.unescape(text)

    def normalize_unicode_nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text or "")

    def normalize_whitespace(self, text: str) -> str:
        return self._RE_WS.sub(" ", text).strip()

    def normalize_punctuation(self, text: str) -> str:
        # Standard + typographic + German quotation marks
        text = re.sub(r'[«»""\„\u201C\u201D]', '"', text)
        # Standard + typographic + German single quotes
        text = re.sub(r"[''‚\u2018\u2019\u201A]", "'", text)
        # Em dash / en dash -> hyphen
        text = re.sub(r"[–—]", "-", text)

        # Deduplicate identical consecutive punctuation marks (including quotes)
        text = re.sub(r'([!?.@#\$%^&*()\-=_+\[\]{}|\\;:\'\",<>/])(?:\s*\1)+', r'\1', text)
        # Ensure internal periods are surrounded by spaces
        text = re.sub(r'(?<!\s)\.(?=\S)', ' .', text)
        text = re.sub(r'(?<=\S)\.(?!\s)', '. ', text)        
        return text

    def reduce_repetition(self, text: str) -> str:
        """
        Reduce repeated characters — keep max 2.
        Useful for sentiment (soooo -> soo, !!! -> !!).
        Leave optional: can alter acronyms (AAA) or ellipsis (...).
        """
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    # ------------------------------------------------------------------
    # CamelCase splitter for hashtags
    # ------------------------------------------------------------------

    def _split_camelcase_long_tokens(self, text: str, min_len: int = 10) -> str:
        """
        Splits long CamelCase tokens into spaced words:
          UkraineRussianWar -> Ukraine Russian War
        Only applied to tokens >= min_len, no digits, mixed case.
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

            tok = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", tok)
            tok = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", tok)
            out_parts.append(tok)

        return " ".join(out_parts)

    # ------------------------------------------------------------------
    # Dangling punctuation cleanup
    # ------------------------------------------------------------------

    def _strip_dangling_punctuation(self, text: str) -> str:
        text = self._RE_LEADING_ORPHAN_PUNCT.sub("", text)
        text = self._RE_TRAILING_DANGLING_PUNCT.sub("", text)
        return text

    # ------------------------------------------------------------------
    # Twitter / social noise
    # ------------------------------------------------------------------

    def clean_twitter_noise(self, text: str) -> str:
        text = self._RE_RT_ANYWHERE.sub("", text)
        text = self._RE_UNOFFICIAL.sub("", text)
        text = self._RE_KEYWORDS.sub("", text)
        text = self._RE_ATTACHED_MORNING_MY.sub("", text)
        text = self._RE_HEBREW_NOISE.sub(" ", text)
        text = self._RE_LATIN_SYMBOLS.sub(" ", text)
        text = self._RE_VIA_MENTION.sub(" ", text)
        text = re.sub(r"\bhttps?\.{2,}\S*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhtt(?:ps?)?…\b", " ", text, flags=re.IGNORECASE)
        return text

    def remove_loose_brackets(self, text: str) -> str:
        """
        Ported from ArabicPreprocessor:
        - Keep healthy single balanced brackets: (text), [text], {text}
        - Collapse repeated bracket runs: ((( -> (
        - Remove unmatched brackets.
        """
        if not text:
            return ""
        text = re.sub(r"\({2,}", "(", text)
        text = re.sub(r"\){2,}", ")", text)
        text = re.sub(r"\[{2,}", "[", text)
        text = re.sub(r"\]{2,}", "]", text)
        text = re.sub(r"\{{2,}", "{", text)
        text = re.sub(r"\}{2,}", "}", text)
        text = re.sub(r"<{2,}", "<", text)
        text = re.sub(r">{2,}", ">", text)

        chars = list(text)
        opening = {"(": ")", "[": "]", "{": "}", "<": ">"}
        closing = {")": "(", "]": "[", "}": "{", ">": "<"}
        stack = []
        to_remove = set()
        for i, ch in enumerate(chars):
            if ch in opening:
                stack.append((ch, i))
            elif ch in closing:
                if stack and stack[-1][0] == closing[ch]:
                    stack.pop()
                else:
                    to_remove.add(i)
        for _, idx in stack:
            to_remove.add(idx)
        for idx in to_remove:
            chars[idx] = " "
        text = "".join(chars)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\[\s+", "[", text)
        text = re.sub(r"\s+\]", "]", text)

        return self.normalize_whitespace(text)

    def remove_decorative_noise(self, text: str) -> str:
        """Ported from ArabicPreprocessor: removes block elements, box drawing, etc."""
        if not text:
            return text
        text = self._RE_REPLACEMENT_CHAR.sub(" ", text)
        text = self._RE_BLOCK_ELEMENTS.sub(" ", text)
        text = self._RE_BOX_DRAWING.sub(" ", text)
        text = self._RE_MISC_SYMBOLS.sub(" ", text)
        text = self._RE_UNICODE_JUNK.sub(" ", text)
        return self.normalize_whitespace(text)

    def remove_common_junk(self, text: str) -> str:
        """Remove known repetitive status messages or spam."""
        if self._RE_JUNK_PATTERNS.search(text):
            return ""
        return text

    # ------------------------------------------------------------------
    # Social entity management
    # ------------------------------------------------------------------

    def standardize_entities(self, text: str) -> str:
        """Replace URLs/mentions/emails with placeholders (for sentiment models)."""
        text = self._RE_URL.sub("http", text)
        text = self._RE_WWW.sub("http", text)
        text = self._RE_MENTION.sub("@user", text)
        text = self._RE_EMAIL.sub("email", text)
        return text

    def remove_links_mentions_emails(self, text: str) -> str:
        """Remove URLs/mentions/emails entirely (for NER/topic/lang-detect)."""
        text = self._RE_URL.sub(" ", text)
        text = self._RE_WWW.sub(" ", text)
        text = self._RE_TCO.sub(" ", text)
        text = self._RE_PIC_TW.sub(" ", text)
        text = self._RE_MENTION.sub(" ", text)
        text = self._RE_EMAIL.sub(" ", text)
        text = self._RE_HTTP_FRAGMENT_TOKEN.sub(" ", text)
        text = self._strip_dangling_punctuation(text)
        return text

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def is_valid(self, text: str, min_tokens: int = 3, min_alpha_ratio: float = 0.3) -> bool:
        """
        Returns True if the text has enough Latin tokens and a healthy alpha-to-noise ratio.
        Filters out articles that are mostly punctuation or junk symbols.
        """
        if not text:
            return False

        tokens = text.split()
        latin_tokens = [t for t in tokens if re.search(r'[A-Za-z]', t)]
        
        # 1. Length check
        if len(latin_tokens) < min_tokens:
            return False

        # 2. Quality Ratio Check (Alpha vs Non-Alpha)
        alpha_count = sum(1 for c in text if c.isalpha())
        total_len = len(text.replace(" ", ""))
        if total_len > 0:
            if (alpha_count / total_len) < min_alpha_ratio:
                return False

        return True

    # ------------------------------------------------------------------
    # Main preprocessing entry point
    # ------------------------------------------------------------------

    def preprocess(
        self,
        text: Optional[str],
        normalize_unicode: bool = True,
        standardize_social: bool = False,   # True -> @user/http/email placeholders (sentiment)
                                            # False -> remove entirely (NER/topic/lang-detect)
        handle_hashtags: bool = False,      # True -> strip #/_ and split CamelCase
        reduce_repetitions: bool = False,   # True -> soooo -> soo (sentiment only)
        normalize_punct: bool = True,
        clean_twitter: bool = False,        # True -> remove RT / via / truncated url noise
        remove_junk: bool = True,           # True -> remove known spam/repetitive patterns
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        # STAGE 1 — HTML and unicode
        text = self.clean_html(text)

        if normalize_unicode:
            text = self.normalize_unicode_nfc(text)

        text = self.normalize_whitespace(text)

        # STAGE 1.5 — Junk check (after HTML removal and whitespace normalization)
        if remove_junk:
            text = self.remove_common_junk(text)
            if not text:
                return ""

        if normalize_punct:
            text = self.normalize_punctuation(text)

        # STAGE 2 — Twitter-specific cleaning
        if clean_twitter:
            text = self.clean_twitter_noise(text)

        # STAGE 2.5 — Decorative noise and loose symbols (Ported from Arabic logic)
        text = self.remove_decorative_noise(text)
        text = self.remove_loose_brackets(text)

        # STAGE 3 — Social entity management
        if standardize_social:
            # For sentiment models (CardiffNLP etc.) — keep placeholders
            text = self.standardize_entities(text)
        else:
            # For NER / topic / lang-detect — remove entirely
            text = self.remove_links_mentions_emails(text)

        # STAGE 4 — Hashtags
        if handle_hashtags:
            text = text.replace("#", " ").replace("_", " ")
            text = self._split_camelcase_long_tokens(text, min_len=10)

        # STAGE 5 — Repetition reduction (sentiment only)
        if reduce_repetitions:
            text = self.reduce_repetition(text)

        # STAGE 6 — Final cleanup
        text = self.normalize_whitespace(text)
        text = self._strip_dangling_punctuation(text)
        return self.normalize_whitespace(text)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def preprocess_for_lang_detect(self, text: str) -> str:
        """Minimal cleaning for language detection — no social standardization."""
        return self.preprocess(
            text,
            standardize_social=False,
            handle_hashtags=False,
            reduce_repetitions=False,
            clean_twitter=False,
        )

    def preprocess_for_sentiment(self, text: str) -> str:
        """
        Optimized for sentiment models (CardiffNLP, BERTweet, etc.).
        Keeps social placeholders, reduces repetitions, cleans Twitter noise.
        """
        return self.preprocess(
            text,
            standardize_social=True,
            handle_hashtags=False,
            reduce_repetitions=True,
            clean_twitter=True,
        )

    def preprocess_for_topic(self, text: str) -> str:
        """
        Optimized for topic classification.
        Removes links/mentions, splits hashtag CamelCase.
        """
        return self.preprocess(
            text,
            standardize_social=False,
            handle_hashtags=True,
            reduce_repetitions=False,
            clean_twitter=True,
        )

    def preprocess_for_ner(self, text: str) -> str:
        """
        Optimized for NER.
        Removes links/mentions, splits hashtag CamelCase.
        Does NOT reduce repetitions (could alter entity names).
        """
        return self.preprocess(
            text,
            standardize_social=False,
            handle_hashtags=True,
            reduce_repetitions=False,
            clean_twitter=True,
        )

    def normalize_entity(self, text: str) -> str:
        """Normalize an extracted entity string for deduplication."""
        if not text:
            return ""

        s = self.normalize_unicode_nfc(text)
        s = self.remove_links_mentions_emails(s)

        # Keep Latin letters (including French accented chars), digits, spaces
        s = re.sub(
            r"[^A-Za-z0-9\s"
            r"àâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ"
            r"]",
            " ", s
        )

        return self.normalize_whitespace(s).lower()