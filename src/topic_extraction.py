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



CATEGORY_DISPLAY = Config.CATEGORY_DISPLAY



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
                model_name = Config.FINETUNED_TOPIC_MODELS.get(lang, Config.FINETUNED_TOPIC_MODELS["en"])
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
        template = "هذا المقال الإخباري يتحدث عن {}." if lang == "ar" else ("Cet article de presse concerne {}." if lang == "fr" else "This news article is about {}.")

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


# ---------------------------------------------------------------------------
# Fine-Tuned Transformer Topic Classifier
# ---------------------------------------------------------------------------

class TransformerTopic:
    """
    Fine-tuned BertForSequenceClassification model for topic classification.
    Uses locally-stored fine-tuned models from finetuned_models/ directory.
    """
    def __init__(self, lang: str = "en", logger: Optional[logging.Logger] = None, device: Optional[int] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.lang = lang.lower()
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self.pipeline = None
        self.load_pipeline()

    def load_pipeline(self):
        """Load the fine-tuned model for the specified language."""
        try:
            model_path = Config.FINETUNED_TOPIC_MODELS.get(self.lang, Config.FINETUNED_TOPIC_MODELS["en"])
            self.logger.info(f"[TransformerTopic] Loading fine-tuned model from {model_path} on device={self.device}")
            self.pipeline = pipeline(
                "text-classification",
                model=model_path,
                device=self.device,
                truncation=True,
                max_length=512
            )
            self.logger.info(f"[TransformerTopic] Model loaded successfully for lang={self.lang}")
        except Exception as e:
            self.logger.error(f"[TransformerTopic] Failed to load model for lang={self.lang}: {e}")
            raise

    def predict(self, text: str, lang: str = None) -> TopicResult:
        """
        Classify the input text into one of the predefined topic categories.
        
        Args:
            text: Input text to classify
            lang: Language code (overrides init lang if provided)
            
        Returns:
            TopicResult with label and confidence score
        """
        if lang:
            self.lang = lang.lower()
        
        if not text or not isinstance(text, str):
            fallback = Config.CATEGORY_DISPLAY[17].get(self.lang, "General")
            return TopicResult(label=fallback, score=0.0)

        text = text.strip()
        if not text:
            fallback = Config.CATEGORY_DISPLAY[17].get(self.lang, "General")
            return TopicResult(label=fallback, score=0.0)

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        self.logger.debug(
            f"[TransformerTopic] Processing {len(chunks)} chunk(s) "
            f"(text length={len(text)} chars, lang={self.lang})"
        )

        try:
            if self.pipeline is None:
                self.load_pipeline()
        except Exception as e:
            self.logger.error(f"[TransformerTopic] Failed to load pipeline: {e}")
            fallback = Config.CATEGORY_DISPLAY[17].get(self.lang, "General")
            return TopicResult(label=fallback, score=0.0)

        label_scores: dict[str, float] = {}
        label_counts: dict[str, int] = {}

        for idx, chunk in enumerate(chunks):
            try:
                # Truncate chunk to stay within model limits
                truncated_chunk = chunk[:1500]
                result = self.pipeline(truncated_chunk)
                
                # result is a list with one dict: [{'label': 'LABEL_X', 'score': float}]
                if result:
                    pred_label = result[0]['label']
                    pred_score = result[0]['score']
                    
                    # Map LABEL_X to actual label from config
                    actual_label = self._map_label(pred_label)
                    
                    label_scores[actual_label] = label_scores.get(actual_label, 0.0) + pred_score
                    label_counts[actual_label] = label_counts.get(actual_label, 0) + 1
                    
                    self.logger.debug(
                        f"[TransformerTopic] chunk {idx + 1}/{len(chunks)} → "
                        f"label='{actual_label}' (raw={pred_label}) score={pred_score:.3f}"
                    )
            except Exception as exc:
                self.logger.warning(
                    f"[TransformerTopic] chunk {idx + 1}/{len(chunks)} failed: {exc}"
                )

        if not label_scores:
            self.logger.error("[TransformerTopic] All chunks failed; returning fallback.")
            fallback = Config.CATEGORY_DISPLAY[17].get(self.lang, "General")
            return TopicResult(label=fallback, score=0.0)

        # Aggregate results by voting
        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        mean_score = label_scores[best_label] / label_counts[best_label]

        self.logger.debug(
            f"[TransformerTopic] Aggregated result → label='{best_label}' "
            f"(votes={label_counts[best_label]}/{len(chunks)}, "
            f"mean_score={mean_score:.3f})"
        )
        return TopicResult(label=best_label, score=mean_score)

    def _map_label(self, raw_label: str) -> str:
        """
        Map model output labels (LABEL_0, LABEL_1, etc.) to human-readable labels.
        Uses the model's id2label config.
        """
        try:
            # Extract the numeric index from LABEL_X
            if raw_label.startswith("LABEL_"):
                idx = int(raw_label.split("_")[1])
                label_dict = self.pipeline.model.config.id2label
                if idx in label_dict:
                    return label_dict[idx]
        except Exception as e:
            self.logger.debug(f"[TransformerTopic] Failed to map label {raw_label}: {e}")
        
        # Fallback: return the raw label
        return raw_label