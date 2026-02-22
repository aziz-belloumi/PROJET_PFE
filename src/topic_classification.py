# src/topic_classification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


DEFAULT_TOPIC_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 50,
    "aggregation": "mean_probs",
}

# Category mapping: id -> labels per language
CATEGORY_MAP: Dict[int, Dict[str, str]] = {
    2: {"ar": "رياضة",              "fr": "Sports",       "en": "Sports"},
    3: {"ar": "سياسة",              "fr": "Politique",     "en": "Politics"},
    4: {"ar": "علوم / تكنولوجيا",    "fr": "Tech/Science",  "en": "Tech/Science"},
    5: {"ar": "فن / ثقافة",         "fr": "Art/Culture",   "en": "Art/Culture"},
    6: {"ar": "اقتصاد",             "fr": "Economie",      "en": "Economy"},
    7: {"ar": "عامة",               "fr": "Géneral",       "en": "General"},
}


def get_candidate_labels(lang: str = "ar") -> List[str]:
    """
    Returns the list of candidate labels for zero-shot classification
    in the specified language. Falls back to English if language not found.
    """
    fallback = "en"
    lang_key = lang if lang in ("ar", "fr", "en") else fallback
    return [v[lang_key] for v in CATEGORY_MAP.values()]


def label_to_category_id(label: str) -> Optional[int]:
    """
    Maps a predicted label (any language) back to its category id.
    Returns None if no match found.
    """
    for cat_id, labels in CATEGORY_MAP.items():
        for lang_label in labels.values():
            if label.strip() == lang_label.strip():
                return cat_id
    return None


@dataclass
class TopicResult:
    label: str                    # best predicted label
    category_id: Optional[int]   # mapped category id (2-7) or None
    score: float                  # confidence of best label
    all_scores: Dict[str, float]  # scores for all candidate labels


class TransformersTopic:
    """
    Zero-shot topic classifier using XLM-RoBERTa (XNLI).
    - Supports Arabic, French, English
    - Token-based chunking for long documents
    - Token-weighted probability aggregation across chunks
    - No fine-tuning required
    """

    def __init__(
        self,
        model_name: str = "joeddav/xlm-roberta-large-xnli",
        logger: Optional[logging.Logger] = None,
        preprocessor=None,
        device: Optional[int] = None,
        max_chunk_tokens: int = DEFAULT_TOPIC_PARAMS["max_chunk_tokens"],
        overlap_tokens: int = DEFAULT_TOPIC_PARAMS["overlap_tokens"],
        aggregation: str = DEFAULT_TOPIC_PARAMS["aggregation"],
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.aggregation = aggregation

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self._clf = pipeline(
            task="zero-shot-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self._safe_max_tokens = max(16, min(int(model_max) - 2, self.max_chunk_tokens))

        self.logger.info(
            f"Topic model loaded: {model_name} | device={self.device} | "
            f"safe_max_tokens={self._safe_max_tokens} overlap_tokens={self.overlap_tokens} "
            f"aggregation={self.aggregation}"
        )

    # ----------------------------
    # Token-based chunking
    # ----------------------------
    def _token_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping token-based chunks.
        Returns list of (chunk_text, offset_char).
        """
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = enc.get("input_ids", [])
        offsets = enc.get("offset_mapping", [])
        if not input_ids or not offsets:
            return []

        chunks: List[Tuple[str, int]] = []
        i = 0
        n = len(input_ids)

        while i < n:
            j = min(i + self._safe_max_tokens, n)
            start_char = offsets[i][0]
            end_char = offsets[j - 1][1]
            if end_char <= start_char:
                i = j
                continue

            chunks.append((text[start_char:end_char], start_char))

            if j == n:
                break
            i = max(0, j - self.overlap_tokens)

        return chunks

    # ----------------------------
    # Classify a single chunk
    # ----------------------------
    def _classify_chunk(self, chunk_text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """
        Run zero-shot classification on a single chunk.
        Returns dict of {label: score} for all candidate labels.
        """
        try:
            result = self._clf(chunk_text, candidate_labels=candidate_labels)
            return dict(zip(result["labels"], result["scores"]))
        except Exception as e:
            self.logger.error(f"Topic classification failed: {e}")
            return {label: 0.0 for label in candidate_labels}

    # ----------------------------
    # Public API
    # ----------------------------
    def predict(self, text: str, lang: str = "ar") -> TopicResult:
        """
        Predict the topic of a text.
        Args:
            text: the article text (raw or preprocessed)
            lang: language code ('ar', 'fr', 'en') to select candidate labels
        Returns:
            TopicResult with best label, category_id, score, and all scores
        """
        if not text or not isinstance(text, str):
            return TopicResult(
                label="UNK",
                category_id=None,
                score=0.0,
                all_scores={},
            )

        # Preprocess if preprocessor is provided
        if self.preprocessor is not None:
            if hasattr(self.preprocessor, "preprocess_for_ner"):
                text = self.preprocessor.preprocess_for_ner(text)
            elif callable(self.preprocessor):
                text = self.preprocessor(text)

        # Get candidate labels for the detected language
        candidate_labels = get_candidate_labels(lang)

        # Chunk the text
        chunks = self._token_chunks(text)
        if not chunks:
            return TopicResult(
                label="UNK",
                category_id=None,
                score=0.0,
                all_scores={},
            )

        # Aggregate scores across chunks (token-weighted mean)
        probs_sum: Dict[str, float] = {label: 0.0 for label in candidate_labels}
        weight_sum = 0.0

        for chunk_text, _ in chunks:
            scores = self._classify_chunk(chunk_text, candidate_labels)
            # Weight = number of tokens in this chunk
            w = float(len(self.tokenizer(chunk_text, add_special_tokens=True)["input_ids"]))
            weight_sum += w

            for label in candidate_labels:
                probs_sum[label] += scores.get(label, 0.0) * w

        # Compute weighted average
        if weight_sum > 0:
            probs_avg = {label: (v / weight_sum) for label, v in probs_sum.items()}
        else:
            probs_avg = {label: 0.0 for label in candidate_labels}

        # Find best label
        best_label = max(probs_avg.items(), key=lambda x: x[1])[0] if probs_avg else "UNK"
        best_score = float(probs_avg.get(best_label, 0.0))

        # Map label back to category id
        cat_id = label_to_category_id(best_label)

        return TopicResult(
            label=best_label,
            category_id=cat_id,
            score=best_score,
            all_scores=probs_avg,
        )