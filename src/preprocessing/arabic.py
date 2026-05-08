import re
import unicodedata
import html
from typing import Optional
import logging


class ArabicPreprocessor:
    _RE_URL = re.compile(r"https?://\S+", flags=re.IGNORECASE)
    _RE_WWW = re.compile(r"www\.\S+", flags=re.IGNORECASE)
    _RE_EMAIL = re.compile(r"\S+@\S+")
    _RE_HTTP_FRAGMENT_TOKEN = re.compile(r"(?:https?|htt)\S+", flags=re.IGNORECASE)

    ARABIC_DIACRITICS = re.compile(
        r"[ً-ٟؐ-ؚۖ-ۜ۟-۪ۤۧۨ-ۭ]"
    )

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def clean_html(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(text)

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def remove_urls(self, text: str) -> str:
        text = self._RE_URL.sub(" ", text)
        text = self._RE_WWW.sub(" ", text)
        text = self._RE_HTTP_FRAGMENT_TOKEN.sub(" ", text)
        return text

    def remove_emails(self, text: str) -> str:
        return self._RE_EMAIL.sub(" ", text)

    def normalize_unicode_nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text or "")

    # ------------------------------------------------------------------
    # Arabic-specific helpers
    # ------------------------------------------------------------------

    def remove_diacritics(self, text: str) -> str:
        return self.ARABIC_DIACRITICS.sub("", text)

    def remove_tatweel(self, text: str) -> str:
        return re.sub(r"\u0640+", "", text)

    def remove_numbers(self, text: str) -> str:
        text = re.sub(r"[0-9]+", "", text)
        text = re.sub(r"[٠-٩]+", "", text)
        return text

    def remove_special_chars_ar(self, text: str, keep_arabic_punct: bool = True) -> str:
        """
        Keep single brackets so valid balanced pairs like (2/405) survive.
        Repeated / loose brackets are handled separately by remove_loose_brackets().
        """
        if keep_arabic_punct:
            pattern = r"[^؀-ۿݐ-ݿ\s0-9٠-٩،؛؟.!?%\"'()\[\]{}]"
        else:
            pattern = r"[^؀-ۿݐ-ݿ\s0-9٠-٩%\"'()\[\]{}]"

        text = re.sub(pattern, " ", text)

        digits = r"0-9٠-٩"
        text = re.sub(rf"(?<![{digits}])%(?![{digits}])", " ", text)

        return text

    def remove_repeated_chars(self, text: str, max_repeat: int = 2) -> str:
        pattern = r"(.)\1{" + str(max_repeat) + r",}"
        replacement = r"\1" * max_repeat
        return re.sub(pattern, replacement, text)

    def remove_loose_brackets(self, text: str) -> str:
        """
        Bracket policy:
        - Keep healthy single balanced brackets: (text), [text], {text}
        - Collapse repeated bracket runs to a SINGLE bracket:
            (((( text ))))) -> ( text )
        - Remove loose / unmatched brackets:
            text) -> text
            ( text -> text

        This preserves useful brackets while cleaning noisy repeated ones.
        """
        if not text:
            return ""

        # --------------------------------------------------------------
        # STEP 1 — Collapse repeated bracket runs to a single bracket
        # Examples:
        #   (((( -> (
        #   ))))) -> )
        #   [[ -> [
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # STEP 2 — Stack-based matching for remaining single brackets
        # Keep only properly matched single pairs
        # --------------------------------------------------------------
        for i, ch in enumerate(chars):
            if ch in opening:
                stack.append((ch, i))
            elif ch in closing:
                if stack and stack[-1][0] == closing[ch]:
                    stack.pop()
                else:
                    to_remove.add(i)

        # Any opening bracket left in stack is unmatched
        for _, idx in stack:
            to_remove.add(idx)

        # Remove only unmatched single brackets
        for idx in to_remove:
            chars[idx] = " "

        text = "".join(chars)

        # Optional cosmetic cleanup: remove extra spaces just inside brackets
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\[\s+", "[", text)
        text = re.sub(r"\s+\]", "]", text)
        text = re.sub(r"\{\s+", "{", text)
        text = re.sub(r"\s+\}", "}", text)
        text = re.sub(r"<\s+", "<", text)
        text = re.sub(r"\s+>", ">", text)

        return self.normalize_whitespace(text)

    # ------------------------------------------------------------------
    # Decorative / symbol noise removal
    # ------------------------------------------------------------------

    # Pre-compiled patterns for decorative noise (class-level)
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
    _RE_OVERLINE_UNDERLINE = re.compile(r"[_̅]+")
    _RE_ARABIC_ZERO = re.compile(r"٠")
    _RE_SPECIFIC_SPAM = re.compile(
        r"\$ho\$ho"
        r"|-\(م\)"
        r"|SäDëËm"
        r"|{بلاغراي:البيضاء}:\."
        r"|19487"
        ,
        flags=re.IGNORECASE,
    )
    # Slash: dedup repeated slashes mid-text, strip from boundaries
    _RE_SLASH_REPEATED = re.compile(r"(/\s*){2,}")
    _RE_SLASH_LEADING = re.compile(r"^(/\s*)+")
    _RE_SLASH_TRAILING = re.compile(r"(\s*/)+$")
    # Caret: replace with comma, dedup
    _RE_CARET_MULTI = re.compile(r"(\^\s*){2,}")
    _RE_CARET_LEADING = re.compile(r"^(\^\s*)+")
    _RE_CARET_TRAILING = re.compile(r"(\s*\^)+$")
    # General: non-Arabic, non-Latin, non-digit, non-common-punctuation symbols
    # Catches remaining dingbats, geometric shapes, misc technical, etc.
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
    # Franco-Arabic / decorative letter substitutions used in social spam
    _FRANCO_ARABIC_CHARS = (
        r"[\u06C9\u06CB\u06A4\u06AF\u06B5\u06B6\u0695\u0671\u06B0"
        r"\uFB70-\uFB8F\uFB90-\uFBAF\uFBB0-\uFBCF\uFBD0-\uFBFF"
        r"\uFC00-\uFCFF\uFD00-\uFDFF\uFE70-\uFEFF]"
    )

    def remove_decorative_noise(self, text: str) -> str:
        """
        Removes decorative/visual noise found in social-media sourced articles:
        - Block elements (█▓▌░▒), box-drawing chars
        - Arrows (↓↑⇓), bullets (◘◙◦), card suits (♦), check marks (✅)
        - Stars, circles, faces (☻), overline/underline art (_̅_̅)
        - Specific spam strings ($ho$ho, -(م), SäDëËm, etc.)
        - Replacement characters (�)
        - Arabic zero (٠) used decoratively
        - Slash deduplication + boundary stripping
        - Caret (^) → comma conversion + deduplication
        - Angle quotes (‹›)
        - General unicode junk (geometric shapes, dingbats, etc.)
        """
        if not text:
            return text

        # 1. Replacement characters
        text = self._RE_REPLACEMENT_CHAR.sub(" ", text)

        # 2. Specific known spam strings
        text = self._RE_SPECIFIC_SPAM.sub(" ", text)

        # 3. Block elements, box-drawing, misc symbols
        text = self._RE_BLOCK_ELEMENTS.sub(" ", text)
        text = self._RE_BOX_DRAWING.sub(" ", text)
        text = self._RE_MISC_SYMBOLS.sub(" ", text)
        text = self._RE_OVERLINE_UNDERLINE.sub(" ", text)

        # 4. Arabic zero used as decorative separator
        text = self._RE_ARABIC_ZERO.sub(" ", text)

        # 5. Slash handling: dedup mid-text, strip boundaries
        text = self._RE_SLASH_LEADING.sub("", text)
        text = self._RE_SLASH_TRAILING.sub("", text)
        text = self._RE_SLASH_REPEATED.sub("/ ", text)

        # 6. Caret handling: strip boundaries, dedup, then replace with comma
        text = self._RE_CARET_LEADING.sub("", text)
        text = self._RE_CARET_TRAILING.sub("", text)
        text = self._RE_CARET_MULTI.sub("، ", text)
        text = text.replace("^", "،")

        # 7. General unicode junk sweep
        text = self._RE_UNICODE_JUNK.sub(" ", text)

        return self.normalize_whitespace(text)

    def remove_social_spam(self, text: str) -> str:
        """
        Detects and removes social-media advertisement / decoration lines.
        Examples:
            تبادل اعلاني صفحتنآ ليست عآلمية <3 بل ۉﻃن يسگنه <3
        These lines use Franco-Arabic decorative letter substitutions
        (e.g. ۉ for و, گ for ك, ڕ for ر, ٱ for ا, ﭰ for ?) which are
        NOT used in real Arabic news text.
        """
        if not text:
            return text

        # If text contains Franco-Arabic decorative chars AND social-media
        # ad keywords, it's almost certainly spam
        franco_pattern = re.compile(self._FRANCO_ARABIC_CHARS)
        ad_keywords = re.compile(
            r"تبادل\s*اعلاني|صفحتن|ليست\s*عالمية|ليست\s*عآلمية",
            flags=re.IGNORECASE,
        )

        # Process line by line to remove only spam lines
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if franco_pattern.search(line) and ad_keywords.search(line):
                continue  # skip entire spam line
            cleaned.append(line)

        text = "\n".join(cleaned)

        # Also remove Franco-Arabic decorative characters individually
        # from surviving text (they never appear in legitimate Arabic)
        text = franco_pattern.sub("", text)

        return self.normalize_whitespace(text)

    def remove_social_noise(self, text: str) -> str:
        """
        Cleans social-media artifacts in the following strict order:

        Step 1 — emoticons (must happen before punctuation stripping)
        Step 2 — repeated / loose brackets
        Step 3 — symbol/character artifacts (^, ~, &, pipes, mentions)
        Step 4 — digit ↔ letter boundary spacing
        Step 5 — repeated commas
        """
        if not text:
            return text

        # ------------------------------------------------------------------
        # STEP 1 — Remove emoticons FIRST
        # ------------------------------------------------------------------

        # :D  :)  :(  :P  ;)  =)  :-D  etc.
        text = re.sub(r'[:;=][\-]?[)D(\[\]P\/O]', ' ', text)

        # xD / XD (not part of a word)
        text = re.sub(r"(?<![A-Za-z؀-ۿ])[xX][dD](?![A-Za-z؀-ۿ])", " ", text)

        # Face emoticons: \(^_^)/  (^o^)  ^___^  ^o^  ~_~  ~~_~~
        text = re.sub(r"[/\\]?\([\^_\-~=.<>oO*]+\)[/\\]?", " ", text)
        text = re.sub(r"\^[_\-=~.oO*]+\^", " ", text)
        text = re.sub(r"~[_\-=~.]+~", " ", text)


        # ------------------------------------------------------------------
        # STEP 1.6 — Remove specific Facebook artifacts (e.g. أعجبني · · مشاركة)
        # ------------------------------------------------------------------
        text = re.sub(r"أعجبني\s*·\s*·\s*مشاركة", " ", text)
        text = re.sub(r"·\s*·\s*مشاركة", " ", text)

        # ------------------------------------------------------------------
        # STEP 2 — Remove repeated / loose brackets
        # ------------------------------------------------------------------
        text = self.remove_loose_brackets(text)

        # ------------------------------------------------------------------
        # STEP 3 — Symbol and character noise
        # ------------------------------------------------------------------
        for lead_pat, trail_pat, multi_pat, one in [
            (r"^(~\s*)+",   r"(\s*~)+$",   r"(~\s*){2,}",  "~"),
            (r"^(&\s*)+",   r"(\s*&)+$",   r"(&\s*){2,}",  "&"),
        ]:
            text = re.sub(lead_pat,  "",        text)
            text = re.sub(trail_pat, "",        text)
            text = re.sub(multi_pat, one + " ", text)

        # Remove pipe characters
        text = re.sub(r"\s*\|\s*", " ", text)

        # Remove @mentions and #hashtags
        text = re.sub(r"[@#]\s*\S+", " ", text)

        # ------------------------------------------------------------------
        # STEP 4 — Digit ↔ Arabic/Latin letter boundary spacing
        # ------------------------------------------------------------------
        text = re.sub(r"([0-9٠-٩])([؀-ۿA-Za-z])", r"\1 \2", text)
        text = re.sub(r"([؀-ۿA-Za-z])([0-9٠-٩])", r"\1 \2", text)

        # ------------------------------------------------------------------
        # STEP 5 — Repeated commas
        # ------------------------------------------------------------------
        text = re.sub(r"(,\s*){2,}", ", ", text)
        text = re.sub(r"^(,\s*)+", "", text)
        text = re.sub(r"(\s*,)+$", "", text)

        return self.normalize_whitespace(text)

    def normalize_punctuation(self, text: str) -> str:
        """
        Cleans punctuation noise:
        1. Strips leading / trailing punctuation and whitespace.
        2. Deduplicates repeated punctuation marks (e.g. "!!!" -> "!").
        """
        punct_set = r"\.،؛\-_:~\^\""  # Exclude !؟? so they are kept at text boundaries

        text = re.sub(f"^[{punct_set}\\s]+", "", text)
        text = re.sub(f"[{punct_set}\\s]+$", "", text)
        
        # Deduplicate identical consecutive punctuation marks (even if separated by spaces)
        text = re.sub(r'([!؟?\.،؛\-_:~\^"])(?:\s*\1)+', r'\1', text)

        return self.normalize_whitespace(text)

    def fix_merged_keywords(self, text: str) -> str:
        ar_chars = r"[؀-ۿݐ-ݿ]"

        # 1. Handle "ليبيا" separately:
        # Only separate before "ليبيا" if 2 or more letters are attached.
        # This prevents splitting single-letter prefixes like "و" (وليبيا) or "ب" (بليبيا).
        text = re.sub(f"({ar_chars}{{2,}})(ليبيا)", r"\1 \2", text)
        text = re.sub(f"(ليبيا)({ar_chars})", r"\1 \2", text)

        # 2. Handle other keywords (original behavior: separate any attached letters)
        other_keywords = ["للبيع", "للإيجار"]
        other_pattern = "|".join(other_keywords)

        text = re.sub(f"({ar_chars})({other_pattern})", r"\1 \2", text)
        text = re.sub(f"({other_pattern})({ar_chars})", r"\1 \2", text)

        return text

    def normalize_entity(self, text: str) -> str:
        if not text:
            return ""

        s = self.normalize_unicode_nfc(text)
        s = self.remove_diacritics(s)
        s = self.remove_tatweel(s)

        s = re.sub(r"[أإآ]", "ا", s)
        s = re.sub(r"ة", "ه", s)
        s = re.sub(r"ى", "ي", s)

        s = re.sub(r"[^؀-ۿ0-9A-Za-z\s]", " ", s)

        return self.normalize_whitespace(s).lower()

    # ------------------------------------------------------------------
    # Main preprocessing entry point
    # ------------------------------------------------------------------

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
        normalize_punct: bool = False,
        remove_tatweel: bool = True,
        handle_hashtags: bool = False,
        fix_merged_keywords: bool = False,
        remove_social_noise: bool = False,
    ) -> str:
        if not text or not isinstance(text, str):
            return ""

        # STAGE 1 — HTML and unicode
        text = self.clean_html(text)

        if normalize_arabic:
            text = self.normalize_unicode_nfc(text)

        # STAGE 2 — URLs and emails
        if remove_urls:
            text = self.remove_urls(text)
        if remove_emails:
            text = self.remove_emails(text)

        # STAGE 3 — Arabic-specific cleaning
        if remove_tatweel:
            text = self.remove_tatweel(text)

        if remove_diacritics:
            text = self.remove_diacritics(text)

        if handle_hashtags:
            text = text.replace("#", " ").replace("_", " ")

        if remove_numbers:
            text = self.remove_numbers(text)

        # STAGE 3.5 — Decorative noise and social spam (always on)
        # Must happen BEFORE social noise / special chars so that
        # block-art, arrows, spam strings etc. are already gone.
        text = self.remove_decorative_noise(text)
        text = self.remove_social_spam(text)

        # STAGE 4 — Social media noise
        # Must happen BEFORE remove_special_chars_ar so emoticons
        # like :D are caught intact before the colon is stripped
        if remove_social_noise:
            text = self.remove_social_noise(text)

        # ------------------------------------------------------------------
        # STAGE 4.5 — Remove repeated / loose brackets BEFORE special chars
        # ------------------------------------------------------------------
        text = self.remove_loose_brackets(text)

        # STAGE 5 — Special character removal
        if remove_special:
            text = self.remove_special_chars_ar(text, keep_arabic_punct=True)

        # STAGE 6 — Optional cleaning passes
        if remove_repeated:
            text = self.remove_repeated_chars(text, max_repeat=2)

        if fix_merged_keywords:
            text = self.fix_merged_keywords(text)

        if normalize_punct:
            text = self.normalize_punctuation(text)

        # ------------------------------------------------------------------
        # STAGE 7 & 8 — Apply final structural fixes and CSV formatting
        # ------------------------------------------------------------------
        return self.apply_final_structural_fixes(text)

    def apply_final_structural_fixes(self, text: str) -> str:
        """
        Always-on final structural fixes.
        These must always run regardless of flags to ensure CSV compatibility
        and remove artifacts exposed by earlier cleaning stages.
        """
        if not text:
            return text

        # 1. Strip leading and trailing punctuation unconditionally (excluding !؟?.)
        text = re.sub(r"^[\s،,\.\-_:;\"']+", "", text)
        text = re.sub(r"[\s،,\.\-_:;\"']+$", "", text)

        # 2. Remove loose single Arabic letters at the beginning or end
        # (e.g. left over conjunctions like 'و' or stray characters like 'م')
        text = re.sub(r"^(?:[ء-ي]\s+)+", "", text)
        text = re.sub(r"(?:\s+[ء-ي])+$", "", text)

        # 3. Strip leading and trailing punctuation again in case the stray letter was hiding some
        text = re.sub(r"^[\s،,\.\-_:;\"']+", "", text)
        text = re.sub(r"[\s،,\.\-_:;\"']+$", "", text)

        # 4. Remove footnote / citation references like ([1]), ([12]), [1], [12], etc.
        text = re.sub(r"\(\[\d+\]\)", " ", text)   # ([1]) form
        text = re.sub(r"\[\d+\]", " ", text)        # [1] bare form

        # 4b. Remove asterisks (e.g. *** used as decorative separators)
        text = re.sub(r"\*+", " ", text)

        # 5. Replace equal signs '=' with a point '.'
        # E.g. "وحده ================================القناة" -> "وحده .القناة"
        text = re.sub(r"=+", ".", text)

        # 5. Quotation marks and CSV compatibility (ALWAYS LAST)
        # To prevent csv.writer from wrapping the text in quotes and doubling internal quotes
        # (which appears as "" in the CSV), we remove all quotation marks completely.
        text = text.replace('"', ' ')
        text = text.replace("'", ' ')
        
        text = text.strip()
        text = re.sub(r'(?<!\d),(?!\d)', '،', text)

        return self.normalize_whitespace(text)