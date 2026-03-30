#!/usr/bin/env python3
"""
Pre-download NER models to avoid download during runtime.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.config import Config

models = [
    Config.ARABERT_NER_MODEL,
    Config.CAMEL_NER_MODEL,
    Config.EN_NER_MODEL,
    Config.FR_NER_MODEL,
]

for model_name in models:
    print(f"Downloading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        print(f"Successfully downloaded {model_name}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")
        continue

print("All models downloaded.")