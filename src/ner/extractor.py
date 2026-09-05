# src/ner/extractor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import logging

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from gliner import GLiNER

# --- Monkey-patch for GLiNER + new huggingface_hub ---
if hasattr(GLiNER, '_from_pretrained'):
    _orig_from_pretrained = GLiNER._from_pretrained
    @classmethod
    def _patched_from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault('proxies', None)
        kwargs.setdefault('resume_download', False)
        return _orig_from_pretrained.__func__(cls, *args, **kwargs)
    GLiNER._from_pretrained = _patched_from_pretrained
# -----------------------------------------------------


class ModelLoadError(Exception):
    """Raised when a model fails to load from Hugging Face."""
    pass

from src.config import Config
from src.config.chunking import token_chunks


DEFAULT_NER_PARAMS: Dict[str, Any] = Config.DEFAULT_NER_PARAMS

# ---------------------------------------------------------------------------
# Label mapping constants — defined in src/config/ner_labels.py, accessed via Config
# ---------------------------------------------------------------------------
LABEL_UNIFICATION      = Config.LABEL_UNIFICATION       # generic cross-model table
_MODEL_LABEL_MAPS      = Config.MODEL_LABEL_MAPS         # ordered per-model registry
_SLOW_TOKENIZER_MODELS = Config.USE_SLOW_TOKENIZER_MODELS


def _get_model_label_map(model_name: str) -> Optional[Dict[str, Optional[str]]]:
    """Return the per-model label map, or None to fall back to LABEL_UNIFICATION."""
    name_lower = model_name.lower()
    for key, label_map in _MODEL_LABEL_MAPS:
        if key in name_lower:
            return label_map
    return None


@dataclass
class NEREntity:
    text: str
    label: str
    start: int
    end: int
    score: float


class TransformersNER:

    _MERGE_GAP_ALLOWED_CHARS = set(" \t\r\n" + ".,،؛:!?-–—ـ/\\()[]{}\"'")
    _BOUNDARY_CHARS = set(" \t\r\n" + ".,،؛:!?-–—/\\()[]{}\"'")

    def __init__(
        self,
        model_name: str,
        logger: Optional[logging.Logger] = None,
        preprocessor=None,
        device: Optional[int] = None,
        max_chunk_tokens: int = DEFAULT_NER_PARAMS["max_chunk_tokens"],
        overlap_tokens: int = DEFAULT_NER_PARAMS["overlap_tokens"],
        score_threshold: float = DEFAULT_NER_PARAMS["score_threshold"],
        merge_entities: bool = DEFAULT_NER_PARAMS["merge_entities"],
        deduplicate: bool = DEFAULT_NER_PARAMS["deduplicate"],
        min_len_person: int = DEFAULT_NER_PARAMS["min_len_person"],
        min_len_other: int = DEFAULT_NER_PARAMS["min_len_other"],
        expand_short_entities: bool = DEFAULT_NER_PARAMS["expand_short_entities"],
        expand_max_len: int = DEFAULT_NER_PARAMS["expand_max_len"],
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.score_threshold = float(score_threshold)

        self.merge_entities = bool(merge_entities)
        self.deduplicate = bool(deduplicate)

        self.min_len_person = int(min_len_person)
        self.min_len_other = int(min_len_other)

        self.expand_short_entities = bool(expand_short_entities)
        self.expand_max_len = int(expand_max_len)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        # Per-model label map (overrides generic LABEL_UNIFICATION when not None)
        self._model_label_map: Optional[Dict[str, Optional[str]]] = _get_model_label_map(model_name)

        # CamemBERT (SentencePiece) requires the slow tokenizer
        use_fast = not any(k in model_name.lower() for k in _SLOW_TOKENIZER_MODELS)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model '{model_name}': {e}")
            raise ModelLoadError(f"Model '{model_name}' could not be loaded: {e}") from e

        self.nlp = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="max",
            device=self.device,
        )

        model_max = getattr(self.tokenizer, "model_max_length", 512) or 512
        self._safe_max_tokens = max(16, min(int(model_max) - 2, self.max_chunk_tokens))

        if self.overlap_tokens >= self._safe_max_tokens:
            self.logger.warning(
                f"overlap_tokens ({self.overlap_tokens}) >= safe_max_tokens ({self._safe_max_tokens}). "
                f"This may reduce chunk progress."
            )

        self.logger.info(
            f"NER model loaded: {model_name} | device={self.device} | "
            f"safe_max_tokens={self._safe_max_tokens} overlap_tokens={self.overlap_tokens} "
            f"score_threshold={self.score_threshold}"
        )

    @staticmethod # Normalizes labels to PER, ORG, LOC, DAT, EVE, MIS, PRO, COM
    def _normalize_label(label: str, model_label_map: Optional[Dict[str, Optional[str]]] = None) -> str:
        raw = (label or "UNK").upper().strip()

        # 1. Try the per-model map first (highest priority)
        if model_label_map is not None:
            if raw in model_label_map:
                result = model_label_map[raw]
                return result if result is not None else "UNK"

        # 2. Fall through to the generic unification table
        if raw in LABEL_UNIFICATION:
            return LABEL_UNIFICATION[raw]
        if raw.startswith("B-") or raw.startswith("I-"):
            stripped = raw[2:]
            if stripped in LABEL_UNIFICATION:
                return LABEL_UNIFICATION[stripped]
            raw = stripped
        if raw in LABEL_UNIFICATION:
            return LABEL_UNIFICATION[raw]
        return raw[:3] if len(raw) >= 3 else raw

    @staticmethod # Simple cleanup: strip whitespace and avoid None.
    def _strip_weird(text: str) -> str:
        return (text or "").strip()

    @staticmethod # Filter out entities that are empty, too short, or only long dashes.
    def _is_valid_entity_text(t: str) -> bool:
        if not t:
            return False
        s = t.strip()
        if len(s) <= 1:
            return False
        if s.replace("ـ", "").strip() == "":
            return False
        return True

    # Return the minimum length required for this label type.
    def _min_len_for_label(self, label: str) -> int:
        l = self._normalize_label(label)
        if l == "PER":
            return self.min_len_person
        return self.min_len_other

    # Validates if two entities can be merged based on gap characters
    def _gap_is_mergeable(self, gap: str) -> bool:
        if gap is None or gap == "":
            return True
        for ch in gap:
            if ch not in self._MERGE_GAP_ALLOWED_CHARS:
                return False
        return True

    # Split text into manageable chunks with overlap to respect tokenizer limits
    def _token_chunks(self, text: str) -> List[Dict[str, Any]]:
        return [
            {"chunk": chunk_text, "offset": offset}
            for chunk_text, offset in token_chunks(
                text, self.tokenizer, self._safe_max_tokens, self.overlap_tokens
            )
        ]

    # Expand entity to full word boundaries using boundary characters
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

    # Merge adjacent entities of the same label if gap is mergeable or linker words are present
    def _merge_adjacent(self, entities: List[NEREntity], base_text: str) -> List[NEREntity]:
        if not entities:
            return []

        entities = sorted(entities, key=lambda e: (e.start, e.end, e.label))
        merged = [entities[0]]

        for ent in entities[1:]:
            last = merged[-1]

            if ent.start < last.end:
                better = ent
                if (last.score > ent.score) or (
                    last.score == ent.score and (last.end - last.start) >= (ent.end - ent.start)
                ):
                    better = last
                merged[-1] = better
                continue

            if ent.label != last.label:
                merged.append(ent)
                continue

            gap = base_text[last.end:ent.start] if 0 <= last.end <= ent.start <= len(base_text) else ""

            is_mergeable = self._gap_is_mergeable(gap)
            if not is_mergeable:
                clean_gap = gap.strip().lower()
                multilingual_linkers = {
                    "of", "and", "the",
                    "de", "du", "des", "et", "le", "la", "les", "en", "aux",
                    "و", "من", "في", "بن", "ابن"
                }
                if clean_gap in multilingual_linkers:
                    is_mergeable = True

            if not is_mergeable:
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

    @staticmethod # Removes duplicate entities keeping the best scoring one
    def _deduplicate(entities: List[NEREntity]) -> List[NEREntity]:
        best: Dict[Tuple[int, int, str], NEREntity] = {}
        for e in entities:
            key = (e.start, e.end, e.label)
            if key not in best or e.score > best[key].score:
                best[key] = e
        return list(best.values())

    def predict(self, text: str, language: Optional[str] = None) -> List[NEREntity]:
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
                label_norm = self._normalize_label(label, self._model_label_map)

                # Skip non-entity tokens (mapped to None in per-model map)
                if label_norm == "UNK" and self._model_label_map is not None:
                    raw = (label or "").upper().strip()
                    if self._model_label_map.get(raw) is None and raw in self._model_label_map:
                        continue

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

        if self.deduplicate:
            entities = self._deduplicate(entities)

        if self.merge_entities:
            entities = self._merge_adjacent(entities, base_text=text)

        return sorted(entities, key=lambda e: (e.start, e.end, e.label))


class GLiNERNER:
    """
    GLiNER-based NER implementation that follows the same interface as TransformersNER.
    """

    DEFAULT_LABELS = Config.GLINER_LABELS
    LANGUAGE_THRESHOLDS = Config.GLINER_LANGUAGE_THRESHOLDS

    def __init__(
        self,
        model_name: str,
        threshold: float = 0.6,
        labels: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        device: Optional[int] = None,
        chunk_size: int = 450,
        chunk_overlap: int = 100,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.threshold = threshold
        self.labels = labels or self.DEFAULT_LABELS.copy()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        self.logger.info(f"Loading GLiNER model: {self.model_name} | device={self.device}")
        try:
            self.model = GLiNER.from_pretrained(self.model_name, proxies=None, resume_download=False)
            if self.device >= 0:
                self.model = self.model.to(f"cuda:{self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load GLiNER model {self.model_name}: {e}")
            raise ModelLoadError(f"GLiNER model '{model_name}' could not be loaded: {e}") from e

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def predict(self, text: str, language: Optional[str] = None) -> List[NEREntity]:
        if not text or not text.strip():
            return []

        effective_threshold = self.LANGUAGE_THRESHOLDS.get(language, self.threshold) if language else self.threshold
        chunks = self._chunk_text(text)

        all_entities: List[NEREntity] = []

        for idx, chunk in enumerate(chunks):
            try:
                raw_entities = self.model.predict_entities(chunk, self.labels, threshold=effective_threshold)
            except Exception as e:
                self.logger.error(f"GLiNER prediction failed on chunk {idx + 1}: {e}")
                continue

            for entity in raw_entities:
                label_norm = TransformersNER._normalize_label(entity["label"])

                # Filter out entities that are too short or invalid
                if not TransformersNER._is_valid_entity_text(entity["text"]):
                    continue

                all_entities.append(
                    NEREntity(
                        text=entity["text"],
                        label=label_norm,
                        start=entity["start"],
                        end=entity["end"],
                        score=entity["score"],
                    )
                )

        # Deduplicate and sort
        final_best: Dict[Tuple[int, int, str], NEREntity] = {}
        for e in all_entities:
            key = (e.start, e.end, e.label)
            if key not in final_best or e.score > final_best[key].score:
                final_best[key] = e

        return sorted(final_best.values(), key=lambda e: (e.start, e.end, e.label))
