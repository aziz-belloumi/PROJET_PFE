# src/config/ner_labels.py
"""
NER label-mapping constants.

Centralises:
  - LABEL_UNIFICATION  : generic cross-model tag → canonical label table
  - Per-model label maps: authoritative BIO→canonical maps for each BERT-based
                          NER model (applied before LABEL_UNIFICATION)
  - MODEL_LABEL_MAPS   : ordered registry used by src.ner.extractor to pick the
                          right per-model map at load time
  - USE_SLOW_TOKENIZER_MODELS : models that need use_fast=False (SentencePiece)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class NERLabelsConfig:

    # =========================================================================
    # Generic cross-model label unification
    # Maps every raw tag variant -> one of: PER ORG LOC DAT EVE MIS PRO COM
    # =========================================================================
    LABEL_UNIFICATION: Dict[str, str] = {
        # --------------------
        # PERSON
        # --------------------
        "PERSON": "PER", "PER": "PER", "PERS": "PER",
        "HUMAN": "PER", "INDIVIDUAL": "PER",
        "PER.": "PER", "PERS.": "PER", "PERSONS": "PER", "PEOPLE": "PER",
        "B-PER": "PER", "I-PER": "PER",
        "B-PERS": "PER", "I-PERS": "PER",
        "B-PERSON": "PER", "I-PERSON": "PER",
        "B-PERSONS": "PER", "I-PERSONS": "PER",

        # --------------------
        # ORGANIZATION
        # --------------------
        "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "ORG": "ORG",
        "COMPANY": "ORG", "INSTITUTION": "ORG", "AGENCY": "ORG",
        "UNIVERSITY": "ORG", "MINISTRY": "ORG", "GOV": "ORG", "GOVERNMENT": "ORG",
        "B-ORG": "ORG", "I-ORG": "ORG",
        "B-ORGANIZATION": "ORG", "I-ORGANIZATION": "ORG",
        "B-ORGANISATION": "ORG", "I-ORGANISATION": "ORG",

        # --------------------
        # LOCATION
        # --------------------
        "LOCATION": "LOC", "LOC": "LOC", "GPE": "LOC",
        "CITY": "LOC", "COUNTRY": "LOC", "REGION": "LOC",
        "STATE": "LOC", "PROVINCE": "LOC",
        "B-LOC": "LOC", "I-LOC": "LOC",
        "B-LOCATION": "LOC", "I-LOCATION": "LOC",
        "B-GPE": "LOC", "I-GPE": "LOC",
        # Facilities → location
        "FAC": "LOC", "FACILITY": "LOC",
        "B-FAC": "LOC", "I-FAC": "LOC",
        "B-FACILITY": "LOC", "I-FACILITY": "LOC",

        # --------------------
        # DATE / TIME
        # --------------------
        "DATE": "DAT", "DAT": "DAT", "TIME": "DAT", "TIM": "DAT",
        "DATETIME": "DAT",
        "B-DATE": "DAT", "I-DATE": "DAT",
        "B-TIME": "DAT", "I-TIME": "DAT",
        "B-DATETIME": "DAT", "I-DATETIME": "DAT",

        # --------------------
        # EVENT
        # --------------------
        "EVENT": "EVE", "EVE": "EVE",
        "B-EVENT": "EVE", "I-EVENT": "EVE",

        # --------------------
        # MISC (catch-all)
        # --------------------
        "MISC": "MIS", "MIS": "MIS", "MISCELLANEOUS": "MIS",
        "OTHER": "MIS", "OTH": "MIS",
        "B-MISC": "MIS", "I-MISC": "MIS",

        # --------------------
        # PRODUCT
        # --------------------
        "PRODUCT": "PRO", "PRO": "PRO",
        "B-PRODUCT": "PRO", "I-PRODUCT": "PRO",
        "BRAND": "PRO", "APP": "PRO", "SOFTWARE": "PRO", "PLATFORM": "PRO",

        # --------------------
        # COMPETITION / SPORT LEAGUE
        # --------------------
        "COMPETITION": "COM", "COM": "COM",
        "LEAGUE": "COM", "TOURNAMENT": "COM", "CHAMPIONSHIP": "COM",
        "B-COMPETITION": "COM", "I-COMPETITION": "COM",

        # --------------------
        # GLiNER descriptive labels
        # --------------------
        "PERSON NAME": "PER",
        "ORGANIZATION OR INSTITUTION OR GOVERNMENT BODY": "ORG",
        "GEOGRAPHIC LOCATION OR CITY OR COUNTRY": "LOC",
        "DATE OR TIME EXPRESSION": "DAT",
        "NAMED EVENT OR ARMED CONFLICT OR POLITICAL CRISIS": "EVE",
        "COMMERCIAL PRODUCT OR BRAND NAME": "PRO",
        "SPORTS COMPETITION OR LEAGUE OR TOURNAMENT": "COM",
    }

    # =========================================================================
    # Per-model label maps
    # Applied BEFORE LABEL_UNIFICATION (highest priority).
    # Value of None means "non-entity token – skip this prediction entirely."
    # =========================================================================

    # aubmindlab/bert-base-arabertv02-ner
    # Emits: B/I-PER  B/I-ORG  B/I-LOC  B/I-MISC  O
    ARABERT_LABEL_MAP: Dict[str, Optional[str]] = {
        "B-PER":  "PER", "I-PER":  "PER",
        "B-ORG":  "ORG", "I-ORG":  "ORG",
        "B-LOC":  "LOC", "I-LOC":  "LOC",
        "B-MISC": "MIS", "I-MISC": "MIS",
        # aggregation_strategy="max" may yield bare tags too
        "PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MIS",
        "O": None,   # non-entity → drop
    }

    # dslim/bert-base-NER
    # Emits: B/I-PER  B/I-ORG  B/I-LOC  B/I-MISC  O
    BERT_EN_LABEL_MAP: Dict[str, Optional[str]] = {
        "B-PER":  "PER", "I-PER":  "PER",
        "B-ORG":  "ORG", "I-ORG":  "ORG",
        "B-LOC":  "LOC", "I-LOC":  "LOC",
        "B-MISC": "MIS", "I-MISC": "MIS",
        "PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MIS",
        "O": None,
    }

    # Jean-Baptiste/camembert-ner
    # Emits: B/I-PER  B/I-ORG  B/I-LOC  B/I-MISC  O
    CAMEMBERT_LABEL_MAP: Dict[str, Optional[str]] = {
        "B-PER":  "PER", "I-PER":  "PER",
        "B-ORG":  "ORG", "I-ORG":  "ORG",
        "B-LOC":  "LOC", "I-LOC":  "LOC",
        "B-MISC": "MIS", "I-MISC": "MIS",
        "PER": "PER", "ORG": "ORG", "LOC": "LOC", "MISC": "MIS",
        "O": None,
    }

    # =========================================================================
    # Registry: ordered list of (model-name-substring, label-map) pairs.
    # _get_model_label_map() in src.ner.extractor iterates this list and returns
    # the first map whose key appears in the lowercased model name.
    # =========================================================================
    MODEL_LABEL_MAPS: List[Tuple[str, Dict[str, Optional[str]]]]  # declared below

    # =========================================================================
    # Tokenizer quirks
    # Models whose tokenizer must be loaded with use_fast=False
    # (e.g. SentencePiece-based models like CamemBERT).
    # =========================================================================
    USE_SLOW_TOKENIZER_MODELS: Tuple[str, ...] = ("camembert",)


# Build MODEL_LABEL_MAPS after the class body so we can reference the maps by name.
NERLabelsConfig.MODEL_LABEL_MAPS = [
    ("arabertv02-ner",  NERLabelsConfig.ARABERT_LABEL_MAP),
    ("bert-base-ner",   NERLabelsConfig.BERT_EN_LABEL_MAP),
    ("camembert-ner",   NERLabelsConfig.CAMEMBERT_LABEL_MAP),
]
