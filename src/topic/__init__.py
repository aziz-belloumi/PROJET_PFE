# src/topic/__init__.py
"""
Topic extraction package.

Exports the LLM-based topic extractor:
  - LLMTopic : LLM-based topic classifier (qwen2.5:7b via Ollama)
"""

from src.topic_generation import LLMTopic, TopicResult

__all__ = ["LLMTopic", "TopicResult"]
