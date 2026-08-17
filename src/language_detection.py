from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from fast_langdetect import LangDetectConfig, LangDetector

from src.preprocessing.router import PreprocessRouter


# ---------------------------------------------------------------------------
# Comprehensive Arabic language, dialect, and Arabic-script ISO codes
# recognised by fastText lid.176.bin
# ---------------------------------------------------------------------------
ARABIC_LANG_CODES: frozenset[str] = frozenset({
    # Standard & Arabic varieties
    "ar", "arb", "arz", "ary", "arq", "apc", "acm", "ayl", "aao", "abh",
    "abv", "acx", "ade", "adf", "ajp", "apd", "ars", "auz", "avl", "ayh",
    "ayn", "mey", "shu", "aec", "ssh",
    # Arabic-script languages often confused by fastText on colloquial / short slang
    "fa", "ckb", "ur", "ps", "sd", "ug", "pnb", "azb",
})

# Minimum thresholds
MIN_CONFIDENCE_THRESHOLD = 0.50
MIN_LATIN_CONF           = 0.35
MIN_ARABIC_COMBINED_CONF = 0.30
MIN_ARABIC_RATIO         = 0.35
MIN_ALPHA_RATIO          = 0.65


@dataclass(frozen=True)
class LanguageDetection:
    lang: str
    score: float
    raw_label: str


# ---------------------------------------------------------------------------
# Detects text language using fast_langdetect (FastText-based).
# Uses top-5 predictions and aggregates confidence across Arabic dialect codes
# (fastText splits probability across dialects for colloquial Arabic).
# ---------------------------------------------------------------------------
class FastTextLanguageDetector:

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,  # expects preprocess_for_lang_detect(text) -> str
        model: str = "auto",  # "lite", "full", or "auto"
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor

        # Configure fast_langdetect — disable truncation so we control input length
        config = LangDetectConfig(
            max_input_length=None,  # No truncation, we handle it ourselves
            model=model,
        )
        self._detector = LangDetector(config)
        self._model = model
        self.logger.info(f"fast_langdetect language detector initialized (model={model})")

    def _arabic_ratio(self, text: str) -> float:
        """Fraction of non-space characters that fall in Arabic Unicode blocks."""
        if not text:
            return 0.0
        arabic_chars = sum(
            1 for c in text
            if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
        )
        total_chars = len(text.replace(" ", ""))
        return arabic_chars / total_chars if total_chars > 0 else 0.0

    def detect(self, text: str, k: int = 5) -> LanguageDetection:
        """
        Detect the language of *text* and return a :class:`LanguageDetection`.

        Uses top-*k* (default 5) FastText predictions and aggregates confidence
        scores across all Arabic dialect / Arabic-script codes so that colloquial
        Arabic text is not misclassified due to probability being spread over
        multiple dialect labels.
        """
        if not text or not isinstance(text, str):
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_lang_detect(text)

        if not text or not text.strip():
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

        try:
            top_preds = self._detector.detect(text, k=k)

            if not top_preds:
                return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

            top_lang = top_preds[0]["lang"].lower().strip()
            top_conf = float(top_preds[0]["score"])

            # Aggregate confidence across all Arabic dialect codes
            # (fastText splits probability across dialects for colloquial Arabic)
            combined_ar_conf = sum(
                float(p["score"])
                for p in top_preds
                if p["lang"].lower().strip() in ARABIC_LANG_CODES
            )

            ar_ratio  = self._arabic_ratio(text)
            has_arabic = any(
                '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
                for c in text
            )

            # --- Arabic routing ---
            if has_arabic and (
                combined_ar_conf >= MIN_ARABIC_COMBINED_CONF
                or ar_ratio >= MIN_ARABIC_RATIO
                or (top_lang in ARABIC_LANG_CODES and top_conf >= MIN_ARABIC_COMBINED_CONF)
            ):
                effective_conf = max(combined_ar_conf, top_conf, ar_ratio)
                return LanguageDetection(
                    lang="ar",
                    score=effective_conf,
                    raw_label="__label__ar",
                )

            # --- English / French routing ---
            if top_lang in ("en", "fr"):
                return LanguageDetection(
                    lang=top_lang,
                    score=top_conf,
                    raw_label=f"__label__{top_lang}",
                )

            # --- Fallback: return top prediction as-is ---
            return LanguageDetection(
                lang=top_lang,
                score=top_conf,
                raw_label=f"__label__{top_lang}",
            )

        except Exception as e:
            self.logger.warning(f"fast_langdetect error: {e}")
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

    def is_arabic(self, text: str, threshold: float = 0.60) -> Tuple[bool, LanguageDetection]:
        res = self.detect(text)
        return (res.lang == "ar" and res.score >= threshold), res


# ---------------------------------------------------------------------------
# Article classifier — wraps FastText + routing logic for batch classification
# ---------------------------------------------------------------------------
class ArticleClassifier:
    """
    Classifies a raw article body into one of:
      dest = 'ar' | 'en' | 'fr' | 'rejected'

    Usage (inside a worker process):
        classifier = ArticleClassifier()
        result = classifier.classify(article_id, body)

    Result dict keys:
      dest    : 'ar' | 'en' | 'fr' | 'rejected'
      row     : [id, cleaned_text, lang]   (only when dest in ar/en/fr)
      words   : int                         (only when accepted)
      conf    : float                       (only when accepted)
      reason  : str                         (only when rejected)
      id      : int                         (only when rejected)
      sample  : str                         (always — for the stats report)
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._router = PreprocessRouter(logger=self.logger)
        config = LangDetectConfig(max_input_length=None, model="auto")
        self._detector = LangDetector(config)
        self.logger.info("ArticleClassifier initialised (FastText auto model)")

    def _arabic_ratio(self, text: str) -> float:
        """Fraction of non-space characters that fall in Arabic Unicode blocks."""
        if not text:
            return 0.0
        arabic_chars = sum(
            1 for c in text
            if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
        )
        total_chars = len(text.replace(" ", ""))
        return arabic_chars / total_chars if total_chars > 0 else 0.0

    def _is_valid_latin(self, text: str) -> bool:
        return self._router.lat.is_valid(
            text, min_tokens=3, min_alpha_ratio=MIN_ALPHA_RATIO
        )

    # ------------------------------------------------------------------
    def classify(self, article_id: int, body: str) -> dict:
        # --- Guard: empty / non-string body ---
        if not body or not isinstance(body, str):
            return self._reject(article_id, "rej_too_short", "<empty or non-string>")

        cleaned = self._router.preprocess(body, lang="", task="lang_detect")
        words   = cleaned.split()

        # --- Guard: too short after cleaning ---
        if len(words) <= 2:
            return self._reject(article_id, "rej_too_short", cleaned or str(body)[:100])

        # --- FastText top-5 predictions ---
        top_preds = self._detector.detect(cleaned, k=5)
        if not top_preds:
            return self._reject(article_id, "rej_low_confidence", cleaned[:100])

        top_lang = top_preds[0]["lang"].lower().strip()
        top_conf = float(top_preds[0]["score"])

        # Aggregate confidence across all Arabic dialect codes
        # (fastText splits probability across dialects for colloquial Arabic)
        combined_ar_conf = sum(
            float(p["score"])
            for p in top_preds
            if p["lang"].lower().strip() in ARABIC_LANG_CODES
        )

        ar_ratio  = self._arabic_ratio(cleaned)
        has_arabic = any(
            '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
            for c in cleaned
        )

        # --- Arabic routing ---
        if has_arabic and (
            combined_ar_conf >= MIN_ARABIC_COMBINED_CONF
            or ar_ratio >= MIN_ARABIC_RATIO
            or (top_lang in ARABIC_LANG_CODES and top_conf >= MIN_ARABIC_COMBINED_CONF)
        ):
            effective_conf = max(combined_ar_conf, top_conf, ar_ratio)
            return {
                "dest":   "ar",
                "row":    [article_id, cleaned, "ar"],
                "words":  len(words),
                "conf":   effective_conf,
                "sample": cleaned[:120],
            }

        # --- English / French routing ---
        if top_lang in ("en", "fr"):
            if top_conf < MIN_LATIN_CONF:
                return self._reject(
                    article_id, "rej_low_confidence",
                    f"[{top_lang} conf={top_conf:.2f}] {cleaned[:100]}",
                )
            if self._is_valid_latin(cleaned):
                return {
                    "dest":   top_lang,
                    "row":    [article_id, cleaned, top_lang],
                    "words":  len(words),
                    "conf":   top_conf,
                    "sample": cleaned[:120],
                }
            return self._reject(article_id, "rej_low_alpha_ratio", cleaned[:100])

        # --- Low confidence / unsupported language ---
        if top_conf < MIN_CONFIDENCE_THRESHOLD and combined_ar_conf < MIN_ARABIC_COMBINED_CONF:
            return self._reject(
                article_id, "rej_low_confidence",
                f"[{top_lang} conf={top_conf:.2f}] {cleaned[:100]}",
            )
        return self._reject(
            article_id, "rej_unsupported_lang",
            f"[{top_lang} conf={top_conf:.2f}] {cleaned[:100]}",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _reject(article_id: int, reason: str, sample: str) -> dict:
        return {"dest": "rejected", "reason": reason, "id": article_id, "sample": sample}