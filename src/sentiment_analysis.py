# src/sentiment_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple, Callable
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


DEFAULT_SENTIMENT_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 50,
    "aggregation": "mean_probs",  # supported: mean_probs, max_chunk
}


@dataclass
class SentimentResult:
    label: str
    score: float
    probs: Dict[str, float]


class TransformersSentiment:

    def __init__(
        self,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,
        device: Optional[int] = None,
        max_chunk_tokens: int = DEFAULT_SENTIMENT_PARAMS["max_chunk_tokens"],
        overlap_tokens: int = DEFAULT_SENTIMENT_PARAMS["overlap_tokens"],
        aggregation: str = DEFAULT_SENTIMENT_PARAMS["aggregation"],
        probs_normalizer: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.aggregation = str(aggregation)
        self.probs_normalizer = probs_normalizer

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

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

    def _preprocess_text(self, text: str) -> str:
        if self.preprocessor is None:
            return text
        if callable(self.preprocessor):
            return self.preprocessor(text)
        if hasattr(self.preprocessor, "preprocess_for_sentiment"):
            return self.preprocessor.preprocess_for_sentiment(text)
        if hasattr(self.preprocessor, "preprocess_for_ner"):
            return self.preprocessor.preprocess_for_ner(text)
        return self.preprocessor.preprocess(text)

    def _token_chunks(self, text: str) -> List[Tuple[str, int]]:
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

    @staticmethod
    def _argmax(probs: Dict[str, float]) -> Tuple[str, float]:
        if not probs:
            return "UNK", 0.0
        lab, sc = max(probs.items(), key=lambda x: x[1])
        return str(lab), float(sc)

    def predict(self, text: str) -> SentimentResult:
        if not text or not isinstance(text, str):
            return SentimentResult(label="UNK", score=0.0, probs={})

        text = self._preprocess_text(text)
        chunks = self._token_chunks(text)
        if not chunks:
            return SentimentResult(label="UNK", score=0.0, probs={})

        if self.aggregation == "mean_probs":
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

            raw_probs = {lab: (v / weight_sum) for lab, v in probs_sum.items()} if weight_sum > 0 else {}

        elif self.aggregation == "max_chunk":
            best_chunk_probs: Dict[str, float] = {}
            best_score = -1.0

            for chunk_text, _ in chunks:
                scores = self._all_scores(chunk_text)
                p = {str(d["label"]): float(d["score"]) for d in scores}
                _, sc = self._argmax(p)
                if sc > best_score:
                    best_score = sc
                    best_chunk_probs = p

            raw_probs = best_chunk_probs

        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

        probs_norm = self.probs_normalizer(raw_probs) if self.probs_normalizer else raw_probs
        label, score = self._argmax(probs_norm)
        return SentimentResult(label=label, score=score, probs=probs_norm)


# ==========================
# Probability normalizers
# ==========================

def probs_norm_prali22_4class(raw_probs: Dict[str, float]) -> Dict[str, float]:
    """PRAli22 labels: Positive/Negative/Neutral/Mixed -> POS/NEG/NEU/MIX"""
    p = {k.strip().lower(): float(v) for k, v in raw_probs.items()}
    return {
        "POS": p.get("positive", 0.0),
        "NEG": p.get("negative", 0.0),
        "NEU": p.get("neutral", 0.0),
        "MIX": p.get("mixed", 0.0),
    }


def probs_norm_3class_posnegneu(raw_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Generic 3-class mapping:
      positive/negative/neutral -> POS/NEG/NEU

    Works for:
    - CAMeL 3-class sentiment
    - CardiffNLP twitter-roberta-base-sentiment-latest
    - CardiffNLP twitter-xlm-roberta-base-sentiment-latest
    """
    p = {k.strip().lower(): float(v) for k, v in raw_probs.items()}
    return {
        "POS": p.get("positive", 0.0),
        "NEG": p.get("negative", 0.0),
        "NEU": p.get("neutral", 0.0),
    }


# Backward-compatible alias (your main may still import this name)
def probs_norm_camel_3class(raw_probs: Dict[str, float]) -> Dict[str, float]:
    return probs_norm_3class_posnegneu(raw_probs)


# Keep only if you still use nlptown/mBERT somewhere; otherwise remove later.
def probs_norm_nlptown_to_3class(raw_probs: Dict[str, float]) -> Dict[str, float]:
    p = {k.strip().lower(): float(v) for k, v in raw_probs.items()}

    def get_star(n: int) -> float:
        for key, val in p.items():
            if key.startswith(f"{n} "):
                return float(val)
        return 0.0

    s1 = get_star(1)
    s2 = get_star(2)
    s3 = get_star(3)
    s4 = get_star(4)
    s5 = get_star(5)

    return {
        "NEG": s1 + s2,
        "NEU": s3,
        "POS": s4 + s5,
    }