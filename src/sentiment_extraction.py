from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List

import torch
from transformers import pipeline

from src.config import Config

DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = Config.DEFAULT_SENTIMENT_PARAMS

@dataclass
class SentimentResult:
    label: str
    score: float
    probs: Dict[str, float]


SENTIMENTS = Config.SENTIMENTS

AMBIGUOUS_SHORT_TEXT_MAX_WORDS = Config.AMBIGUOUS_SHORT_TEXT_MAX_WORDS
AMBIGUOUS_SHORT_TEXT_MAX_CHARS = Config.AMBIGUOUS_SHORT_TEXT_MAX_CHARS

NEGATIVE_DOMAIN_CUES = Config.NEGATIVE_DOMAIN_CUES

# Compiled Regex Patterns
RE_TOKENS = re.compile(r"\S+")

AR_STARTERS = Config.QUESTION_STARTERS["ar"]
EN_STARTERS = Config.QUESTION_STARTERS["en"]
FR_STARTERS = Config.QUESTION_STARTERS["fr"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_article(text: str, size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text) or size <= overlap:
            break
    return chunks


def _tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return RE_TOKENS.findall(text.strip())



def _is_question_like(text: str, lang: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if "?" in stripped or "؟" in stripped:
        return True
    words = stripped.split()
    if not words:
        return False
    first_word = words[0].lower()
    if lang == "ar":
        return first_word in AR_STARTERS
    if lang == "fr":
        return first_word in FR_STARTERS
    return first_word in EN_STARTERS


def _is_ambiguous_short_question(text: str, lang: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    words = _tokenize_words(stripped)
    if not words:
        return False
    short_enough = (
        len(words) <= AMBIGUOUS_SHORT_TEXT_MAX_WORDS
        or len(stripped) <= AMBIGUOUS_SHORT_TEXT_MAX_CHARS
    )
    if not short_enough:
        return False
    return _is_question_like(stripped, lang)


def _contains_negative_domain_cue(text: str, lang: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    cues = NEGATIVE_DOMAIN_CUES.get(lang, NEGATIVE_DOMAIN_CUES["en"])
    return any(cue in lowered for cue in cues)

def _apply_safe_post_rules(text: str, lang: str, label: str, score: float) -> Tuple[str, float]:
    if label == "NEUTRAL" and _contains_negative_domain_cue(text, lang):
        return "NEGATIVE", 0.75
    return label, score

def map_label(model_name: str, raw_label: str) -> str:
    l_upper = raw_label.upper().strip()
    l_lower = raw_label.lower().strip()

    if "camelbert-mix-sentiment" in model_name:
        # Model id2label: {0: "positive", 1: "negative", 2: "neutral"}
        mapping = {
            'POSITIVE': 'POSITIVE', 'NEGATIVE': 'NEGATIVE', 'NEUTRAL': 'NEUTRAL',
            # Fallback for older checkpoints that may expose LABEL_X
            'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL',
        }
        if l_upper in mapping:
            return mapping[l_upper]

    elif "twitter-roberta-base-sentiment-latest" in model_name:
        # Model id2label: {0: "negative", 1: "neutral", 2: "positive"}
        mapping = {
            'NEGATIVE': 'NEGATIVE', 'NEUTRAL': 'NEUTRAL', 'POSITIVE': 'POSITIVE',
            # Fallback for older checkpoints that may expose LABEL_X
            'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'POSITIVE',
        }
        if l_upper in mapping:
            return mapping[l_upper]

    elif "distilcamembert-base-sentiment" in model_name:
        # Model id2label: {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
        # Collapse 5-star ratings → 3-class sentiment
        star_mapping = {
            '1 STAR':  'NEGATIVE', '1 ÉTOILE':  'NEGATIVE',
            '2 STARS': 'NEGATIVE', '2 ÉTOILES': 'NEGATIVE',
            '3 STARS': 'NEUTRAL',  '3 ÉTOILES': 'NEUTRAL',
            '4 STARS': 'POSITIVE', '4 ÉTOILES': 'POSITIVE',
            '5 STARS': 'POSITIVE', '5 ÉTOILES': 'POSITIVE',
            # Fallback for older checkpoints that may expose LABEL_X
            'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEGATIVE',
            'LABEL_2': 'NEUTRAL',
            'LABEL_3': 'POSITIVE', 'LABEL_4': 'POSITIVE',
        }
        if l_upper in star_mapping:
            return star_mapping[l_upper]

    # Generic substring fallback (handles any model returning pos/neg/neu variants)
    if 'pos' in l_lower:
        return 'POSITIVE'
    if 'neg' in l_lower:
        return 'NEGATIVE'
    if 'neu' in l_lower or 'mix' in l_lower:
        return 'NEUTRAL'

    return 'NEUTRAL'

# ---------------------------------------------------------------------------
# LLM Sentiment Analyzer
# ---------------------------------------------------------------------------

class LLMSentiment:

    def __init__(self, logger: Optional[logging.Logger] = None, preprocessor: Any = None, device: Optional[int] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.preprocessor = preprocessor
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self.pipelines: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def _get_pipeline(self, lang: str):
        with self.lock:
            if lang not in self.pipelines:
                model_name = Config.FINETUNED_SENTIMENT_MODELS.get(lang, Config.FINETUNED_SENTIMENT_MODELS["en"])
                self.logger.info(f"Loading sentiment pipeline for lang={lang} model={model_name} on device={self.device}")
                self.pipelines[lang] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=self.device,
                    truncation=True,
                    max_length=512
                )
            return self.pipelines[lang]

    def _preprocess_text(self, text: str) -> str:
        if self.preprocessor is None:
            return text
        if callable(self.preprocessor):
            return self.preprocessor(text)
        for attr in ["preprocess_for_sentiment", "preprocess_for_ner", "preprocess"]:
            if hasattr(self.preprocessor, attr):
                return getattr(self.preprocessor, attr)(text)
        return text

    def predict(self, text: str, lang: str = "en") -> SentimentResult:
        if not text or not isinstance(text, str):
            reason = f"Input text is empty or non-string (type={type(text)})"
            self.logger.warning(f"[SENTIMENT UNK] lang={lang} | Reason: {reason}")
            return SentimentResult(label="UNK", score=0.0,
                                   probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0})

        orig_text = text
        text = self._preprocess_text(text)

        if _is_ambiguous_short_question(text, lang):
            return SentimentResult(
                label="NEUTRAL",
                score=0.6,
                probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 1.0},
            )

        chunks = chunk_article(text)
        if not chunks:
            chunks = [text]

        label_scores: Dict[str, float] = {k: 0.0 for k in SENTIMENTS}
        label_counts = {k: 0 for k in SENTIMENTS}
        valid_chunks_processed = 0

        try:
            pipe = self._get_pipeline(lang)
        except Exception as e:
            reason = f"Failed to load sentiment pipeline for lang='{lang}': {e}"
            self.logger.error(f"[SENTIMENT UNK] {reason}")
            return SentimentResult(label="UNK", score=0.0,
                                   probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0})

        for chunk in chunks:
            try:
                res = pipe(chunk)[0]
                prediction = res['label']
                score = res['score']
                norm_prediction = map_label(pipe.model.config._name_or_path, prediction)

                if norm_prediction in label_scores:
                    label_scores[norm_prediction] += score
                    label_counts[norm_prediction] += 1
                    valid_chunks_processed += 1

            except Exception as e:
                self.logger.warning(f"Sentiment chunk prediction failed: {e}")

        if valid_chunks_processed == 0:
            reason = f"0 out of {len(chunks)} chunks produced valid predictions for lang='{lang}'"
            self.logger.warning(f"[SENTIMENT UNK] {reason}")
            return SentimentResult(label="UNK", score=0.0,
                                   probs={"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0})

        best_label = max(label_counts, key=lambda lbl: (label_counts[lbl], label_scores[lbl]))
        
        # Computes confidence based on total chunks ratio
        mean_score = label_counts[best_label] / valid_chunks_processed

        best_label, mean_score = _apply_safe_post_rules(text, lang, best_label, mean_score)

        probs = {k: label_counts[k] / valid_chunks_processed for k in SENTIMENTS}

        if best_label not in SENTIMENTS or best_label == "UNK":
            self.logger.warning(
                f"[SENTIMENT UNK] lang={lang} | label={best_label} score={mean_score:.4f} | probs={probs}"
            )

        return SentimentResult(label=best_label, score=mean_score, probs=probs)