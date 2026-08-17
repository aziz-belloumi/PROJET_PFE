# src/config/qwen.py
"""
Qwen / Ollama local LLM inference parameters.
"""

from __future__ import annotations

from .base import getenv


class QwenConfig:
    OLLAMA_URL = getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    QWEN_MODEL_NAME = "qwen2.5:7b"
    QWEN_NUM_CTX = 2048
    QWEN_TEMPERATURE = 0
    QWEN_NUM_PREDICT = 30
    QWEN_MAX_RETRIES = 3
    QWEN_TIMEOUT_SEC = 30
