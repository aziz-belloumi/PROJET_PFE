# src/language_detection.py
"""
Language detection with FastText (lid.176.bin).

Provides:
  - ArticleClassifier: classify a single preprocessed article into
    'ar', 'en', 'fr', or a rejection reason.
  - FastTextLanguageDetector: thin wrapper around fast_langdetect used
    for pre-warming and direct _detector access.
"""

import logging
from typing import Optional

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

# Minimum thresholds (mirrors constants in fetch_and_preprocess_sql.py)
MIN_CONFIDENCE_THRESHOLD = 0.50
MIN_LATIN_CONF           = 0.35
MIN_ARABIC_COMBINED_CONF = 0.30
MIN_ARABIC_RATIO         = 0.35
MIN_ALPHA_RATIO          = 0.65


# ---------------------------------------------------------------------------
# Preprocessor wrapper (must stay picklable for multiprocessing workers)
# ---------------------------------------------------------------------------
class ArticlePreprocessor:
    """Thin wrapper around PreprocessRouter for lang-detect preprocessing."""

    def __init__(self):
        self.router = PreprocessRouter()

    def clean(self, text: str) -> str:
        return self.router.preprocess(text, lang="", task="lang_detect")

    def arabic_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        arabic_chars = sum(
            1 for c in text
            if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F'
        )
        total_chars = len(text.replace(" ", ""))
        return arabic_chars / total_chars if total_chars > 0 else 0.0

    def is_valid_latin(self, text: str) -> bool:
        return self.router.lat.is_valid(
            text, min_tokens=3, min_alpha_ratio=MIN_ALPHA_RATIO
        )


# ---------------------------------------------------------------------------
# Article classifier — wraps FastText + routing logic
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
        self._preprocessor = ArticlePreprocessor()
        config = LangDetectConfig(max_input_length=None, model="auto")
        self._detector = LangDetector(config)
        self.logger.info("ArticleClassifier initialised (FastText auto model)")

    # ------------------------------------------------------------------
    def classify(self, article_id: int, body: str) -> dict:
        # --- Guard: empty / non-string body ---
        if not body or not isinstance(body, str):
            return self._reject(article_id, "rej_too_short", "<empty or non-string>")

        cleaned = self._preprocessor.clean(body)
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

        ar_ratio  = self._preprocessor.arabic_ratio(cleaned)
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
            if self._preprocessor.is_valid_latin(cleaned):
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


# ---------------------------------------------------------------------------
# Thin FastText wrapper (used for pre-warming only)
# ---------------------------------------------------------------------------
class FastTextLanguageDetector:

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,
        model: str = "auto",
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor
        config = LangDetectConfig(max_input_length=None, model=model)
        self._detector = LangDetector(config)
        self._model = model
        self.logger.info(f"FastTextLanguageDetector initialised (model={model})")

    def detect(self, text: str) -> str:
        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_lang_detect(text)
        results = self._detector.detect(text, k=1)
        return results[0]["lang"] if results else "unk"