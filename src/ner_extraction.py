# src/ner_extraction.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
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


class TransformersNER:
    """
    Robust NER extractor (AraBERT / CAMeL) with:
    - token-based chunking (offset_mapping) to avoid 512-token crash
    - score filtering
    - short/noise filtering
    - word-boundary expansion for short entities to reduce fragments (e.g., "لب" -> "لبده")
    - merge adjacent entities (safe punctuation gaps) WITHOUT merging overlaps/duplicates
    - deduplication because of overlap (done BEFORE merge to avoid duplicated texts)
    """

    _MERGE_GAP_ALLOWED_CHARS = set(" \t\r\n" + ".,،؛:!?-–—ـ/\\()[]{}\"'")
    _BOUNDARY_CHARS = set(" \t\r\n" + ".,،؛:!?-–—/\\()[]{}\"'")

    def __init__(
        self,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,  # expects preprocess_for_ner(text) -> str
        device: Optional[int] = None,
        max_chunk_tokens: int = 450,
        overlap_tokens: int = 80,
        score_threshold: float = 0.60,
        merge_entities: bool = True,
        deduplicate: bool = True,
        min_len_person: int = 2,
        min_len_other: int = 3,
        expand_short_entities: bool = True,
        expand_max_len: int = 5,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.score_threshold = float(score_threshold)

        self.merge_entities = merge_entities
        self.deduplicate = deduplicate

        self.min_len_person = int(min_len_person)
        self.min_len_other = int(min_len_other)

        self.expand_short_entities = bool(expand_short_entities)
        self.expand_max_len = int(expand_max_len)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        self.nlp = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )

        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self._safe_max_tokens = max(16, min(int(model_max) - 2, int(self.max_chunk_tokens)))

        self.logger.info(
            f"NER model loaded: {model_name} | device={self.device} | "
            f"safe_max_tokens={self._safe_max_tokens} overlap_tokens={self.overlap_tokens} "
            f"score_threshold={self.score_threshold}"
        )

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _normalize_label(label: str) -> str:
        return (label or "UNK").upper()

    @staticmethod
    def _strip_weird(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _is_valid_entity_text(t: str) -> bool:
        if not t:
            return False
        s = t.strip()
        if len(s) <= 1:
            return False
        if s.replace("ـ", "").strip() == "":
            return False
        return True

    def _min_len_for_label(self, label: str) -> int:
        l = self._normalize_label(label)
        if l in {"PER", "PERSON"}:
            return self.min_len_person
        return self.min_len_other

    def _gap_is_mergeable(self, gap: str) -> bool:
        if gap is None or gap == "":
            return True
        for ch in gap:
            if ch not in self._MERGE_GAP_ALLOWED_CHARS:
                return False
        return True

    # ----------------------------
    # Token-based chunking
    # ----------------------------
    def _token_chunks(self, text: str) -> List[Dict[str, Any]]:
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = enc.get("input_ids", [])
        offsets: List[Tuple[int, int]] = enc.get("offset_mapping", [])

        if not input_ids or not offsets:
            return []

        chunks = []
        i = 0
        n = len(input_ids)

        while i < n:
            j = min(i + self._safe_max_tokens, n)
            start_char = offsets[i][0]
            end_char = offsets[j - 1][1]

            if end_char <= start_char:
                i = j
                continue

            chunks.append({"chunk": text[start_char:end_char], "offset": start_char})

            if j == n:
                break

            i = max(0, j - self.overlap_tokens)

        return chunks

    # ----------------------------
    # Expand short entities to word boundaries
    # ----------------------------
    def _expand_to_word_boundaries(self, base_text: str, start: int, end: int) -> Tuple[int, int, str]:
        n = len(base_text)
        if not (0 <= start < end <= n):
            return start, end, base_text[start:end] if 0 <= start < end <= n else ""

        s = start
        e = end

        while s > 0 and base_text[s - 1] not in self._BOUNDARY_CHARS:
            s -= 1
        while e < n and base_text[e] not in self._BOUNDARY_CHARS:
            e += 1

        expanded = base_text[s:e]

        while expanded and expanded[0] in self._BOUNDARY_CHARS:
            s += 1
            expanded = base_text[s:e]
        while expanded and expanded[-1] in self._BOUNDARY_CHARS:
            e -= 1
            expanded = base_text[s:e]

        return s, e, expanded

    # ----------------------------
    # Merge / dedup
    # ----------------------------
    def _merge_adjacent(self, entities: List[NEREntity], base_text: str) -> List[NEREntity]:
        """
        Merge ONLY if:
        - same label
        - no overlap (ent.start >= last.end)
        - gap is mergeable (whitespace/punct only)
        For overlaps/duplicates: keep the better entity (higher score or longer span).
        """
        if not entities:
            return []

        entities = sorted(entities, key=lambda e: (e.start, e.end, e.label))
        merged = [entities[0]]

        for ent in entities[1:]:
            last = merged[-1]

            # Overlap or duplicate span -> do not concatenate texts
            if ent.start < last.end:
                better = ent
                if (last.score > ent.score) or (last.score == ent.score and (last.end - last.start) >= (ent.end - ent.start)):
                    better = last
                merged[-1] = better
                continue

            # Different label -> no merge
            if ent.label != last.label:
                merged.append(ent)
                continue

            gap = base_text[last.end:ent.start] if 0 <= last.end <= ent.start <= len(base_text) else ""
            if not self._gap_is_mergeable(gap):
                merged.append(ent)
                continue

            new_text = (last.text or "") + (gap or "") + (ent.text or "")
            merged[-1] = NEREntity(
                text=new_text,
                label=last.label,
                start=last.start,
                end=ent.end,
                score=max(last.score, ent.score),
            )

        return merged

    @staticmethod
    def _deduplicate(entities: List[NEREntity]) -> List[NEREntity]:
        """
        Deduplicate by (start,end,label) keeping highest score.
        """
        best: Dict[Tuple[int, int, str], NEREntity] = {}
        for e in entities:
            key = (e.start, e.end, e.label)
            if key not in best or e.score > best[key].score:
                best[key] = e
        return list(best.values())

    # ----------------------------
    # Public API
    # ----------------------------
    def predict(self, text: str) -> List[NEREntity]:
        if not text or not isinstance(text, str):
            return []

        if self.preprocessor is not None:
            text = self.preprocessor.preprocess_for_ner(text)

        chunks = self._token_chunks(text)
        if not chunks:
            return []

        entities: List[NEREntity] = []

        for c in chunks:
            chunk_text = c["chunk"]
            offset = int(c["offset"])

            try:
                preds = self.nlp(chunk_text)
            except Exception as e:
                self.logger.error(f"NER inference failed for model={self.model_name}: {e}")
                continue

            for p in preds:
                label = p.get("entity_group", p.get("entity", "UNK"))
                label_norm = self._normalize_label(label)
                score = float(p.get("score", 0.0))

                if score < self.score_threshold:
                    continue

                start_local = int(p.get("start", 0))
                end_local = int(p.get("end", 0))
                start = start_local + offset
                end = end_local + offset

                if not (0 <= start < end <= len(text)):
                    continue

                ent_text = self._strip_weird(text[start:end])

                # Expand short entities to full word (fix fragments like "لب")
                if self.expand_short_entities and len(ent_text) <= self.expand_max_len:
                    new_s, new_e, expanded = self._expand_to_word_boundaries(text, start, end)
                    expanded = self._strip_weird(expanded)
                    if expanded:
                        start, end, ent_text = new_s, new_e, expanded

                if not self._is_valid_entity_text(ent_text):
                    continue

                if len(ent_text) < self._min_len_for_label(label_norm):
                    continue

                entities.append(
                    NEREntity(
                        text=ent_text,
                        label=label_norm,
                        start=int(start),
                        end=int(end),
                        score=score,
                    )
                )

        # IMPORTANT: deduplicate FIRST (prevents "لبدهلبده"), then merge true adjacency
        if self.deduplicate:
            entities = self._deduplicate(entities)

        if self.merge_entities:
            entities = self._merge_adjacent(entities, base_text=text)

        entities = sorted(entities, key=lambda e: (e.start, e.end, e.label))
        return entities