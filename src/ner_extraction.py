# src/ner_extraction.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


@dataclass
class NEREntity:
    text: str
    label: str
    start: int
    end: int
    score: float


def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Char-based chunking (POC). We will still enforce the 512-token BERT limit per chunk
    using the tokenizer (because char length != token length).
    Returns [{"chunk": str, "offset": int}, ...]
    """
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + max_chars, n)
        chunks.append({"chunk": text[i:j], "offset": i})
        if j == n:
            break
        i = max(0, j - overlap)

    return chunks


class TransformersNER:
    """
    HF NER wrapper compatible with your transformers version:
    - Does NOT pass truncation/max_length kwargs to pipeline (your pipeline rejects them)
    - Enforces <= 512 tokens manually using tokenizer
    - Loads model once
    """

    def __init__(
        self,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,  # expects preprocess_for_ner(text) -> str
        device: Optional[int] = None,
        max_chars: int = 2000,
        overlap: int = 200,
        max_tokens: int = 512,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chars = max_chars
        self.overlap = overlap
        self.max_tokens = max_tokens

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.nlp = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )

        self.logger.info(f"NER model loaded: {model_name}")

    def _token_len(self, s: str) -> int:
        return len(self.tokenizer(s, add_special_tokens=True)["input_ids"])

    def _fit_to_max_tokens(self, text: str) -> str:
        """
        Returns a prefix of `text` whose tokenized length <= max_tokens.
        Binary search on char prefix length (fast enough for POC).
        """
        if not text:
            return ""

        if self._token_len(text) <= self.max_tokens:
            return text

        lo, hi = 1, len(text)
        best = ""

        while lo <= hi:
            mid = (lo + hi) // 2
            prefix = text[:mid]
            if self._token_len(prefix) <= self.max_tokens:
                best = prefix
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    def predict(self, text: str) -> List[NEREntity]:
        if not text or not isinstance(text, str):
            return []

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_ner(text)

        chunks = _chunk_text(text, max_chars=self.max_chars, overlap=self.overlap)
        entities: List[NEREntity] = []

        for c in chunks:
            chunk_text = c["chunk"]
            offset = c["offset"]

            # Enforce BERT limit WITHOUT passing kwargs to pipeline
            chunk_text = self._fit_to_max_tokens(chunk_text)
            if not chunk_text:
                continue

            try:
                preds = self.nlp(chunk_text)  # no kwargs
            except Exception as e:
                self.logger.error(f"NER inference failed for model={self.model_name}: {e}")
                continue

            for p in preds:
                start_local = int(p.get("start", 0))
                end_local = int(p.get("end", 0))

                start = start_local + offset
                end = end_local + offset

                label = p.get("entity_group", p.get("entity", "UNK"))
                score = float(p.get("score", 0.0))

                # Extract entity text from the FULL processed text using global offsets
                ent_text = text[start:end] if 0 <= start < end <= len(text) else p.get("word", "")

                entities.append(
                    NEREntity(
                        text=str(ent_text),
                        label=str(label),
                        start=int(start),
                        end=int(end),
                        score=score,
                    )
                )

        return entities