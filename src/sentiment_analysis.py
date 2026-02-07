# src/sentiment_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ============================================================
# Recommended models
# ============================================================
CAMEL_SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
ARABERT_SENTIMENT_MODEL = "PRAli22/AraBert-Arabic-Sentiment-Analysis"  # labels: Positive/Negative/Neutral/Mixed
MBERT_SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


# ============================================================
# Module-owned defaults (so main.py can import + log them)
# ============================================================
DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 50,
    "aggregation": "mean_probs",  # how to aggregate across chunks
}


@dataclass
class SentimentResult:
    label: str                 # normalized label (e.g., POS/NEG/NEU/MIX) or raw if no mapping
    score: float               # confidence of chosen label
    probs: Dict[str, float]    # averaged probabilities per raw label
    raw_best_label: str        # best raw label before normalization


class TransformersSentiment:
    """
    Text sentiment classifier with:
    - model loaded once
    - token-based chunking (offset_mapping) to respect max length
    - chunk-level probabilities aggregated to a document-level decision
    """

    def __init__(
        self,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,   # expects a callable: preprocess(text)->str OR object with preprocess_for_ner/text method
        device: Optional[int] = None,
        max_chunk_tokens: int = DEFAULT_SENTIMENT_PARAMS["max_chunk_tokens"],
        overlap_tokens: int = DEFAULT_SENTIMENT_PARAMS["overlap_tokens"],
        aggregation: str = DEFAULT_SENTIMENT_PARAMS["aggregation"],
        label_normalizer=None,  # function(raw_best_label, probs)->normalized_label
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.aggregation = aggregation
        self.label_normalizer = label_normalizer

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # We'll request all class scores (robustly across transformers versions)
        self._clf = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self._safe_max_tokens = max(16, min(int(model_max) - 2, self.max_chunk_tokens))

        self.logger.info(
            f"Sentiment model loaded: {model_name} | device={self.device} | "
            f"safe_max_tokens={self._safe_max_tokens} overlap_tokens={self.overlap_tokens} "
            f"aggregation={self.aggregation}"
        )

    def _token_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Token-based chunking using offset_mapping.
        Returns list of (chunk_text, offset_char) offsets are mainly for debug; not needed for sentiment.
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

    def _all_scores(self, chunk_text: str) -> List[Dict[str, float]]:
        """
        Returns list of {"label": ..., "score": ...} for all labels.
        Works across transformers versions:
        - some accept top_k=None
        - some accept return_all_scores=True
        """
        try:
            out = self._clf(chunk_text, top_k=None)
            if isinstance(out, list) and out and isinstance(out[0], list):
                return out[0]
            return out
        except TypeError:
            out = self._clf(chunk_text, return_all_scores=True)
            if isinstance(out, list) and out and isinstance(out[0], list):
                return out[0]
            return out

    def predict(self, text: str) -> SentimentResult:
        if not text or not isinstance(text, str):
            return SentimentResult(label="UNK", score=0.0, probs={}, raw_best_label="UNK")

        # Preprocess (keep it flexible)
        if self.preprocessor is not None:
            if callable(self.preprocessor):
                text = self.preprocessor(text)
            elif hasattr(self.preprocessor, "preprocess_for_ner"):
                text = self.preprocessor.preprocess_for_ner(text)
            else:
                text = self.preprocessor.preprocess(text)

        chunks = self._token_chunks(text)
        if not chunks:
            return SentimentResult(label="UNK", score=0.0, probs={}, raw_best_label="UNK")

        probs_sum: Dict[str, float] = {}
        weight_sum = 0.0

        for chunk_text, _ in chunks:
            scores = self._all_scores(chunk_text)
            w = float(len(self.tokenizer(chunk_text, add_special_tokens=True)["input_ids"]))
            weight_sum += w

            for d in scores:
                lab = str(d["label"])
                sc = float(d["score"])
                probs_sum[lab] = probs_sum.get(lab, 0.0) + sc * w

        probs_avg = {lab: (v / weight_sum) for lab, v in probs_sum.items()} if weight_sum > 0 else {}

        raw_best = max(probs_avg.items(), key=lambda x: x[1])[0] if probs_avg else "UNK"
        best_score = float(probs_avg.get(raw_best, 0.0))

        if self.label_normalizer is not None:
            final_label = self.label_normalizer(raw_best, probs_avg)
        else:
            final_label = raw_best

        return SentimentResult(
            label=final_label,
            score=best_score,
            probs=probs_avg,
            raw_best_label=raw_best,
        )


# ============================================================
# Label normalization helpers
# ============================================================

def normalize_3class_label(raw_best: str, probs: Dict[str, float]) -> str:
    """
    Common normalizer for 3-class models with various label naming.
    Returns POS/NEG/NEU or UNK.
    """
    r = (raw_best or "").upper()

    if "NEG" in r:
        return "NEG"
    if "POS" in r:
        return "POS"
    if "NEU" in r:
        return "NEU"

    return r or "UNK"


def normalize_arabert_prali4(raw_best: str, probs: Dict[str, float]) -> str:
    """
    Normalizer for PRAli22/AraBert-Arabic-Sentiment-Analysis
    id2label: Positive / Negative / Neutral / Mixed
    Output: POS/NEG/NEU/MIX
    """
    r = (raw_best or "").strip().lower()
    if r == "positive":
        return "POS"
    if r == "negative":
        return "NEG"
    if r == "neutral":
        return "NEU"
    if r == "mixed":
        return "MIX"
    return "UNK"


def normalize_nlptown_stars(raw_best: str, probs: Dict[str, float]) -> str:
    """
    nlptown model labels are like '1 star', '2 stars', ... '5 stars'.
    Map to NEG/NEU/POS.
    """
    r = (raw_best or "").lower()
    if r.startswith("1") or r.startswith("2"):
        return "NEG"
    if r.startswith("3"):
        return "NEU"
    if r.startswith("4") or r.startswith("5"):
        return "POS"
    return "UNK"