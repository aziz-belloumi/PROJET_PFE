# src/language_detection/detector.py
"""
FastText-based language detector with Arabic dialect aggregation and script ratio checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from fast_langdetect import LangDetectConfig, LangDetector


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

# Minimum confidence / script thresholds
MIN_ARABIC_COMBINED_CONF = 0.30
MIN_ARABIC_RATIO         = 0.35


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

        # Configure fast_langdetect — pin cache so lid.176.bin is downloaded only once
        _cache_dir = str(
            __import__("pathlib").Path(__file__).resolve().parent.parent.parent / "finetuned_models" / "lang_detect_cache"
        )
        config = LangDetectConfig(
            max_input_length=None,
            model=model,
            cache_dir=_cache_dir,
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
