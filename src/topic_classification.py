# src/topic_classification.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import logging

import torch
from transformers import AutoTokenizer, pipeline

from src.chunking import token_chunks


DEFAULT_TOPIC_PARAMS: Dict[str, Any] = {
    "max_chunk_tokens": 450,
    "overlap_tokens": 100,
    "aggregation": "mean_probs",
    "multi_label": False,
}

CATEGORY_MAP: Dict[int, Dict[str, str]] = {
    0: {
        "ar": "السياسة والحكومة والعلاقات الدولية والدبلوماسية",
        "fr": "la politique, le gouvernement et les relations internationales",
        "en": "politics, government and international relations",
    },
    1: {
        "ar": "الاقتصاد والتجارة والاستثمار والأسواق المالية",
        "fr": "l'économie, le commerce, l'investissement et les marchés financiers",
        "en": "economy, trade, investment and financial markets",
    },
    2: {
        "ar": "الأمن الداخلي والشرطة والجريمة والإرهاب",
        "fr": "la sécurité intérieure, la police, la criminalité et le terrorisme",
        "en": "internal security, police, crime and terrorism",
    },
    3: {
        "ar": "الطاقة والنفط والغاز والموارد الطبيعية",
        "fr": "l'énergie, le pétrole, le gaz et les ressources naturelles",
        "en": "energy, oil, gas and natural resources",
    },
    4: {
        "ar": "الحروب والنزاعات المسلحة والعمليات العسكرية",
        "fr": "les guerres, les conflits armés et les opérations militaires",
        "en": "wars, armed conflicts and military operations",
    },
    5: {
        "ar": "الانتخابات والتصويت والحملات الانتخابية",
        "fr": "les élections, le vote et les campagnes électorales",
        "en": "elections, voting and electoral campaigns",
    },
    6: {
        "ar": "القضاء والمحاكم والقضايا القانونية",
        "fr": "la justice, les tribunaux et les affaires juridiques",
        "en": "courts, justice system and legal cases",
    },
    7: {
        "ar": "الصحة والطب والأوبئة والرعاية الصحية",
        "fr": "la santé, la médecine, les épidémies et les soins de santé",
        "en": "health, medicine, epidemics and healthcare",
    },
    8: {
        "ar": "الطقس والمناخ والكوارث الطبيعية",
        "fr": "la météo, le climat et les catastrophes naturelles",
        "en": "weather, climate and natural disasters",
    },
    9: {
        "ar": "الرياضة والبطولات والمسابقات الرياضية",
        "fr": "le sport, les championnats et les compétitions sportives",
        "en": "sports, championships and athletic competitions",
    },
    10: {
        "ar": "الثقافة والفنون والأدب والفعاليات الاجتماعية والثقافية",
        "fr": "la culture, les arts, la littérature et les événements sociaux et culturels",
        "en": "culture, arts, literature and social and cultural events",
    },
}

# What gets stored in DB (short label)
CATEGORY_DISPLAY: Dict[int, Dict[str, str]] = {
    0: {"ar": "السياسة", "fr": "Politique", "en": "Politics"},
    1: {"ar": "الاقتصاد", "fr": "Économie", "en": "Economy"},
    2: {"ar": "الأمن", "fr": "Sécurité", "en": "Security"},
    3: {"ar": "الطاقة", "fr": "Énergie", "en": "Energy"},
    4: {"ar": "النزاع", "fr": "Conflit", "en": "Conflict"},
    5: {"ar": "الانتخابات", "fr": "Élections", "en": "Elections"},
    6: {"ar": "العدالة", "fr": "Justice", "en": "Justice"},
    7: {"ar": "الصحة", "fr": "Santé", "en": "Health"},
    8: {"ar": "الطقس", "fr": "Météo", "en": "Weather"},
    9: {"ar": "الرياضة", "fr": "Sport",    "en": "Sports"},
    10: {"ar": "الثقافة", "fr": "Culture",  "en": "Culture"},
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


# OLD templates:
# def get_hypothesis_template(lang: str) -> str:
#     """
#     Language-specific hypothesis templates improve XNLI zero-shot quality.
#     """
#     lang = (lang or "").lower().strip()
#     if lang == "ar":
#         return "هذا النص عن {}."
#     if lang == "fr":
#         return "Ce texte parle de {}."
#     # default English
#     return "This text is about {}."

def get_hypothesis_template(lang: str) -> str:
    """
    Language-specific hypothesis templates improve XNLI zero-shot quality.
    """
    lang = (lang or "").lower().strip()
    if lang == "ar":
        return "هذا النص يتحدث بشكل رئيسي عن موضوع {}."
    if lang == "fr":
        return "Le sujet principal de ce texte est {}."
    # default English
    return "The main subject of this text is {}."


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
        return token_chunks(text, self.tokenizer, self._safe_max_tokens, self.overlap_tokens)

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