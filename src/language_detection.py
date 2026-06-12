from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from fast_langdetect import LangDetectConfig, LangDetector


@dataclass(frozen=True)
class LanguageDetection:
    lang: str         
    score: float       
    raw_label: str    

# Detects text language using fast_langdetect (FastText-based)
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

    def detect(self, text: str, k: int = 1) -> LanguageDetection:
        
        if not text or not isinstance(text, str):
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_lang_detect(text)

        if not text or not text.strip():
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

        try:
            
            results = self._detector.detect(text, k=k)
            
            if results:
                top = results[0]
                lang = top["lang"]
                score = float(top["score"])
                raw_label = f"__label__{lang}"
            else:
                lang = "unk"
                score = 0.0
                raw_label = "__label__unk"
        except Exception as e:
            self.logger.warning(f"fast_langdetect error: {e}")
            lang = "unk"
            score = 0.0
            raw_label = "__label__unk"

        return LanguageDetection(lang=lang, score=score, raw_label=raw_label)

    def is_arabic(self, text: str, threshold: float = 0.60) -> Tuple[bool, LanguageDetection]:
        res = self.detect(text)
        return (res.lang == "ar" and res.score >= threshold), res