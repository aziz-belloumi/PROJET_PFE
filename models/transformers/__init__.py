"""
models/transformers package
----------------------------
Modular runners for HuggingFace Transformers:
  - bert_runner: Sentiment Analysis & Named Entity Recognition (BERT models)
  - nli_runner: Zero-Shot Topic Classification (NLI models)
  - evaluation: Metrics computation (Accuracy, F1 Macro, Precision, Recall) & report generation
  - shared: Shared utilities, label mappings, and helper functions
"""
from .bert_runner import run_bert_models, SENTIMENT_MODELS, NER_MODELS
from .nli_runner import run_nli_models, TOPIC_MODELS
from .evaluation import (
    calculate_accuracy,
    calculate_sentiment_metrics,
    calculate_topic_accuracy,
    calculate_topic_metrics,
    generate_full_report,
    print_dataset_overview,
    check_missing_values,
)

__all__ = [
    "run_bert_models",
    "run_nli_models",
    "SENTIMENT_MODELS",
    "NER_MODELS",
    "TOPIC_MODELS",
    "calculate_accuracy",
    "calculate_sentiment_metrics",
    "calculate_topic_accuracy",
    "calculate_topic_metrics",
    "generate_full_report",
    "print_dataset_overview",
    "check_missing_values",
]
