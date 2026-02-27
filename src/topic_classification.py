# src/topic_classification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import logging

import torch
from transformers import AutoTokenizer, pipeline


DEFAULT_TOPIC_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 100,
    "aggregation": "mean_probs",
    "multi_label": False,
}

CATEGORY_MAP: Dict[int, Dict[str, str]] = {
    2: {"ar": "الرياضة",                 "fr": "Sports",                    "en": "Sports"},
    3: {"ar": "السياسة",                 "fr": "Politique",                 "en": "Politics"},
    4: {"ar": "العلوم والتكنولوجيا",      "fr": "Science et technologie",    "en": "Science and technology"},
    5: {"ar": "الفنون والثقافة",          "fr": "Arts et culture",           "en": "Arts and culture"},
    6: {"ar": "الاقتصاد",                "fr": "Économie",                  "en": "Economy"},
    8: {"ar": "حرب ونزاع",               "fr": "Guerre et conflit",          "en": "War and conflict"},
    7: {"ar": "عام",                     "fr": "Général",                  "en": "General"},
}


def get_candidate_labels(lang: str = "ar") -> List[str]:
    """
    Returns candidate labels for zero-shot classification in the specified language.
    Falls back to English if language not found.
    """
    fallback = "en"
    lang_key = lang if lang in ("ar", "fr", "en") else fallback
    return [v[lang_key] for v in CATEGORY_MAP.values()]


def label_to_category_id(label: str) -> Optional[int]:
    """Maps a predicted label (any language) back to its category id."""
    if not label:
        return None
    lab = label.strip().casefold()
    for cat_id, labels in CATEGORY_MAP.items():
        for lang_label in labels.values():
            if lab == (lang_label or "").strip().casefold():
                return cat_id
    return None


def get_hypothesis_template(lang: str) -> str:
    """
    Language-specific hypothesis templates improve XNLI zero-shot quality.
    """
    lang = (lang or "").lower().strip()
    if lang == "ar":
        return "هذا النص عن {}."
    if lang == "fr":
        return "Ce texte parle de {}."
    # default English
    return "This text is about {}."


@dataclass
class TopicResult:
    label: str
    category_id: Optional[int]
    score: float
    all_scores: Dict[str, float]


class TransformersTopic:
    """
    Zero-shot topic classifier using XLM-RoBERTa (XNLI).
    - Supports Arabic, French, English
    - Token-based chunking for long documents
    - Token-weighted probability aggregation across chunks
    - Uses language-specific hypothesis templates
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
        multi_label: bool = DEFAULT_TOPIC_PARAMS["multi_label"],
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.preprocessor = preprocessor

        self.max_chunk_tokens = int(max_chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.aggregation = str(aggregation)
        self.multi_label = bool(multi_label)

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
            f"aggregation={self.aggregation} multi_label={self.multi_label}"
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
    def _classify_chunk(
        self,
        chunk_text: str,
        candidate_labels: List[str],
        hypothesis_template: str,
    ) -> Dict[str, float]:
        """
        Run zero-shot classification on a single chunk.
        Returns dict of {label: score} for all candidate labels.
        """
        try:
            result = self._clf(
                chunk_text,
                candidate_labels=candidate_labels,
                multi_label=self.multi_label,
                hypothesis_template=hypothesis_template,
            )
            return dict(zip(result["labels"], result["scores"]))
        except Exception as e:
            self.logger.error(f"Topic classification failed: {e}")
            return {label: 0.0 for label in candidate_labels}

    # ----------------------------
    # Preprocessing hook
    # ----------------------------
    def _maybe_preprocess(self, text: str, lang: str) -> str:
        """
        If a preprocessor is provided:
        - Prefer router-style: preprocess(text, lang, task="topic")
        - Else fall back to callable(text)
        """
        if self.preprocessor is None:
            return text

        # Router-like object
        if hasattr(self.preprocessor, "preprocess"):
            try:
                return self.preprocessor.preprocess(text, lang=lang, task="topic")
            except TypeError:
                # If someone passes a simpler preprocessor with preprocess(text)
                return self.preprocessor.preprocess(text)

        # Callable function
        if callable(self.preprocessor):
            return self.preprocessor(text)

        return text

    # ----------------------------
    # Public API
    # ----------------------------
    def predict(self, text: str, lang: str = "ar") -> TopicResult:
        if not text or not isinstance(text, str):
            return TopicResult(label="UNK", category_id=None, score=0.0, all_scores={})

        lang = (lang or "en").lower().strip()
        text = self._maybe_preprocess(text, lang)

        candidate_labels = get_candidate_labels(lang)
        hypothesis_template = get_hypothesis_template(lang)

        chunks = self._token_chunks(text)
        if not chunks:
            return TopicResult(label="UNK", category_id=None, score=0.0, all_scores={})

        if self.aggregation not in ("mean_probs", "max_chunk"):
            self.logger.warning(f"Unknown aggregation='{self.aggregation}', falling back to mean_probs")
            self.aggregation = "mean_probs"

        # ---- aggregation: mean_probs (token-weighted) ----
        if self.aggregation == "mean_probs":
            probs_sum: Dict[str, float] = {label: 0.0 for label in candidate_labels}
            weight_sum = 0.0

            for chunk_text, _ in chunks:
                scores = self._classify_chunk(chunk_text, candidate_labels, hypothesis_template)
                w = float(len(self.tokenizer(chunk_text, add_special_tokens=True)["input_ids"])) or 1.0
                weight_sum += w
                for label in candidate_labels:
                    probs_sum[label] += scores.get(label, 0.0) * w

            probs_avg = (
                {label: (v / weight_sum) for label, v in probs_sum.items()}
                if weight_sum > 0
                else {label: 0.0 for label in candidate_labels}
            )

            best_label = max(probs_avg.items(), key=lambda x: x[1])[0]
            best_score = float(probs_avg.get(best_label, 0.0))
            cat_id = label_to_category_id(best_label)

            return TopicResult(
                label=best_label,
                category_id=cat_id,
                score=best_score,
                all_scores=probs_avg,
            )

        # ---- aggregation: max_chunk (take the chunk with strongest top score) ----
        best_overall_label = "UNK"
        best_overall_score = -1.0
        best_overall_scores: Dict[str, float] = {label: 0.0 for label in candidate_labels}

        for chunk_text, _ in chunks:
            scores = self._classify_chunk(chunk_text, candidate_labels, hypothesis_template)
            top_label = max(scores.items(), key=lambda x: x[1])[0] if scores else "UNK"
            top_score = float(scores.get(top_label, 0.0))
            if top_score > best_overall_score:
                best_overall_score = top_score
                best_overall_label = top_label
                best_overall_scores = scores

        return TopicResult(
            label=best_overall_label,
            category_id=label_to_category_id(best_overall_label),
            score=float(best_overall_score if best_overall_score >= 0 else 0.0),
            all_scores=best_overall_scores,
        )