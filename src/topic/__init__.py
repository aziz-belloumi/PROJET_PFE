# src/topic/__init__.py
"""
Topic extraction package.

Exports the Hugging Face zero-shot topic extractor:
  - LLMTopic : Zero-shot topic classifier (Hugging Face Transformers)
"""

from src.topic_generation import LLMTopic, TopicResult

__all__ = ["LLMTopic", "TopicResult"]
