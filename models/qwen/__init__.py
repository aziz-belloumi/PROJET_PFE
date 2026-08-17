# models/qwen/__init__.py
"""
Qwen/Ollama-based implementations of LLMSentiment and LLMTopic.
Used by models/qwen/run_qwen.py for offline annotation with qwen2.5:7b.
These are intentionally decoupled from the main HF Transformers pipeline.
"""
from .sentiment_analysis import LLMSentiment, SentimentResult
from .topic_generation import LLMTopic, TopicResult

__all__ = ["LLMSentiment", "SentimentResult", "LLMTopic", "TopicResult"]
