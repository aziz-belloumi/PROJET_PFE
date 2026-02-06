from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import logging

import fasttext


@dataclass(frozen=True)
class LanguageDetection:
    lang: str          # "ar"
    score: float       # confidence in [0, 1]
    raw_label: str     # "__label__ar"


class FastTextLanguageDetector:


    def __init__(
        self,
        model_path: str | Path,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,  # expects preprocess_for_lang_detect(text) -> str
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor

        self.model_path = self._resolve_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"fastText model not found: {self.model_path}")

        self.model = fasttext.load_model(str(self.model_path))
        self.logger.info(f"fastText language detector loaded model: {self.model_path}")

    def _resolve_path(self, p: str | Path) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        project_root = Path(__file__).resolve().parents[1]
        return (project_root / p).resolve()

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        # "__label__ar" -> "ar"
        return raw_label.replace("__label__", "").strip()

    def detect(self, text: str, k: int = 1) -> LanguageDetection:
        # Use title+body concatenation at call site for best accuracy.
        
        if not text or not isinstance(text, str):
            return LanguageDetection(lang="unk", score=0.0, raw_label="__label__unk")

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_lang_detect(text)

        labels, scores = self.model.predict(text, k=k)
        raw_label = labels[0] if labels else "__label__unk"
        score = float(scores[0]) if len(scores) > 0 else 0.0
        lang = self._normalize_label(raw_label)

        return LanguageDetection(lang=lang, score=score, raw_label=raw_label)

    def is_arabic(self, text: str, threshold: float = 0.60) -> Tuple[bool, LanguageDetection]:
        """
        Convenience helper: returns (is_arabic, result)
        """
        res = self.detect(text)
        return (res.lang == "ar" and res.score >= threshold), res