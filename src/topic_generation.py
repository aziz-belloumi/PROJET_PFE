from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Any

import torch
from transformers import pipeline

from src.config import Config




@dataclass
class TopicResult:
    label: str
    score: float



CATEGORY_DISPLAY    = Config.CATEGORY_DISPLAY
HYPOTHESIS_TEMPLATES = Config.HYPOTHESIS_TEMPLATES



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_article(text, size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
    """Split long article into overlapping chunks using character offsets."""
    chunks = []
    start = 0
    if not text:
        return chunks

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks


# ---------------------------------------------------------------------------
# LLM Topic Extractor
# ---------------------------------------------------------------------------

def get_topic_labels(lang: str) -> list[str]:
    l = lang.strip().lower()
    target_lang = l if l in ['ar', 'en', 'fr'] else 'en'
    return [cats[target_lang] for cats in Config.CATEGORY_DISPLAY.values()]


class LLMTopic:
    def __init__(self, logger: Optional[logging.Logger] = None, device: Optional[int] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self.pipelines: dict[str, Any] = {}
        self.lock = threading.Lock()

    def _get_pipeline(self, lang: str):
        with self.lock:
            if lang not in self.pipelines:
                model_name = Config.TOPIC_MODELS.get(lang, Config.TOPIC_MODELS["en"])
                self.logger.info(f"Loading zero-shot classification pipeline for lang={lang} model={model_name} on device={self.device}")
                self.pipelines[lang] = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=self.device
                )
            return self.pipelines[lang]

    def predict(self, text: str, lang: str) -> TopicResult:
        if not text or not text.strip():
            self.logger.warning("[LLMTopic] Received empty text; returning fallback.")
            fallback = Config.CATEGORY_DISPLAY[17].get(lang, "General")
            return TopicResult(label=fallback, score=0.0)

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        lang_map = {"ar": "Arabic", "en": "English", "fr": "French"}
        language = lang_map.get(lang, "English")

        self.logger.info(
            f"[LLMTopic] Processing {len(chunks)} chunk(s) "
            f"(text length={len(text)} chars, lang={lang})"
        )

        try:
            pipe = self._get_pipeline(lang)
        except Exception as e:
            self.logger.error(f"Failed to load topic pipeline for lang={lang}: {e}")
            fallback = Config.CATEGORY_DISPLAY[17].get(lang, "General")
            return TopicResult(label=fallback, score=0.0)

        labels = get_topic_labels(lang)
        template = Config.HYPOTHESIS_TEMPLATES.get(lang, Config.HYPOTHESIS_TEMPLATES["en"])

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int]   = {}

        for idx, chunk in enumerate(chunks):
            try:
                # Truncate chunk text to stay within context length safely (~1500 chars)
                truncated_chunk = chunk[:1500]
                res = pipe(truncated_chunk, candidate_labels=labels, hypothesis_template=template)
                
                best_label = res['labels'][0]
                best_score = res['scores'][0]

                label_scores[best_label] = label_scores.get(best_label, 0.0) + best_score
                label_counts[best_label] = label_counts.get(best_label, 0) + 1

                self.logger.debug(
                    f"[LLMTopic] chunk {idx + 1}/{len(chunks)} → "
                    f"label='{best_label}' score={best_score:.3f}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"[LLMTopic] chunk {idx + 1}/{len(chunks)} failed: {exc}"
                )

        if not label_scores:
            self.logger.error("[LLMTopic] All chunks failed; returning fallback.")
            fallback = Config.CATEGORY_DISPLAY[17].get(lang, "General")
            return TopicResult(label=fallback, score=0.0)

        # Vote aggregator execution
        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_scores[best_label] / label_counts[best_label]

        self.logger.info(
            f"[LLMTopic] Aggregated result → label='{best_label}' "
            f"(votes={label_counts[best_label]}/{len(chunks)}, "
            f"mean_score={mean_score:.3f})"
        )
        return TopicResult(label=best_label, score=mean_score)

    def unload(self) -> None:
        """No-op execution target wrapper."""
        pass