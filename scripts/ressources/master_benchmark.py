import pandas as pd
import torch
import time
import psutil
import os
import sys
import io
import json
import logging
import requests
from transformers import pipeline
from pathlib import Path
from typing import List, Dict, Any, Optional

# Force stdout to UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.ner_extraction import TransformersNER, GLiNERNER
from src.topic_generation import LLMTopic
from src.sentiment_analysis import LLMSentiment

CSV_PATH = PROJECT_ROOT / "scripts" / "manual_eval_global.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "ressources"
REPORT_FILE = OUTPUT_DIR / "resource_usage_report.csv"

# --- CATEGORY MAPPING (For Zero-Shot) ---
CATEGORY_DISPLAY = {
    0: {"ar": "السياسة",       "fr": "Politique",      "en": "Politics"},
    1: {"ar": "الاقتصاد",      "fr": "Économie",       "en": "Economy"},
    2: {"ar": "الأمن",         "fr": "Sécurité",       "en": "Security"},
    3: {"ar": "الطاقة",        "fr": "Énergie",        "en": "Energy"},
    4: {"ar": "النزاع",        "fr": "Conflit",        "en": "Conflict"},
    5: {"ar": "الانتخابات",    "fr": "Élections",      "en": "Elections"},
    6: {"ar": "العدالة",       "fr": "Justice",        "en": "Justice"},
    7: {"ar": "الصحة",         "fr": "Santé",          "en": "Health"},
    8: {"ar": "الطقس",         "fr": "Météo",          "en": "Weather"},
    9: {"ar": "الرياضة",       "fr": "Sport",          "en": "Sports"},
    10: {"ar": "الثقافة",      "fr": "Culture",        "en": "Culture"},
    11: {"ar": "التعليم",      "fr": "Éducation",      "en": "Education"},
    12: {"ar": "التكنولوجيا",  "fr": "Technologie",    "en": "Technology"},
    13: {"ar": "البيئة",       "fr": "Environnement",  "en": "Environment"},
    14: {"ar": "الدبلوماسية",  "fr": "Diplomatie",     "en": "Diplomacy"},
    15: {"ar": "الدين",        "fr": "Religion",       "en": "Religion"},
    16: {"ar": "الهجرة",       "fr": "Migration",      "en": "Migration"},
    17: {"ar": "عام",         "fr": "Général",          "en": "General"},
}

def get_topic_labels(lang):
    return [cats[lang] for cats in CATEGORY_DISPLAY.values() if lang in cats]

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    def get_cpu_mem(self): return self.process.memory_info().rss / (1024 * 1024)
    def get_gpu_mem(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0

# Custom LLM wrapper to force CPU/GPU via Ollama options
class BenchLLM:
    def __init__(self, task: str, use_gpu: bool):
        self.task = task
        self.use_gpu = use_gpu
        self.url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2.5:7b"
        
    def predict(self, text: str, lang: str):
        # We simulate the call with options to force CPU if needed
        # Ollama 'num_gpu' option: 0 means CPU only
        num_gpu = 1 if self.use_gpu else 0
        payload = {
            "model": self.model_name,
            "prompt": f"Task: {self.task}, Lang: {lang}, Text: {text[:200]}", # Minimal prompt for resource check
            "stream": False,
            "options": {"num_gpu": num_gpu, "num_predict": 10, "temperature": 0}
        }
        try:
            requests.post(self.url, json=payload, timeout=30)
        except:
            pass

def benchmark_item(task_name: str, model_name: str, lang: str, device_type: str, df: pd.DataFrame):
    monitor = ResourceMonitor()
    device_idx = 0 if device_type == "GPU" and torch.cuda.is_available() else -1
    
    # Filter articles for the specific language
    indices = df.index[df['langue'] == lang].tolist()
    if not indices: return None

    print(f"[{device_type}] {task_name.upper()} | {model_name} | {lang} ({len(indices)} docs)")
    
    results = {
        "device": device_type,
        "task": task_name,
        "model": model_name,
        "lang": lang,
        "load_time_sec": 0,
        "total_inf_time_sec": 0,
        "avg_ms_per_doc": 0,
        "peak_cpu_mb": 0,
        "peak_gpu_mb": 0
    }

    # Measure Load
    mem_start = monitor.get_cpu_mem()
    gpu_start = monitor.get_gpu_mem()
    start_load = time.time()
    
    obj = None
    try:
        if model_name == "Qwen-LLM":
            obj = BenchLLM(task_name, use_gpu=(device_type=="GPU"))
        elif task_name == 'topic':
            obj = pipeline("zero-shot-classification", model=model_name, device=device_idx)
        elif task_name == 'sentiment':
            m_len = 128 if 'bertweet' in model_name.lower() else 512
            obj = pipeline("sentiment-analysis", model=model_name, device=device_idx, truncation=True, max_length=m_len)
        elif task_name == 'ner':
            if 'gliner' in model_name.lower():
                obj = GLiNERNER(model_name=model_name, device=device_idx)
            else:
                obj = TransformersNER(model_name=model_name, device=device_idx)
        
        results["load_time_sec"] = time.time() - start_load
        
        # Measure Inference
        start_inf = time.time()
        max_cpu = mem_start
        max_gpu = gpu_start

        for i, idx in enumerate(indices):
            text = str(df.loc[idx, 'texte'])
            
            if model_name == "Qwen-LLM":
                obj.predict(text, lang)
            elif task_name == 'topic':
                labels = get_topic_labels(lang)
                obj(text[:1000], candidate_labels=labels)
            elif task_name == 'sentiment':
                obj(text[:1000])
            elif task_name == 'ner':
                obj.predict(text[:1000])
            
            if i % 10 == 0:
                max_cpu = max(max_cpu, monitor.get_cpu_mem())
                max_gpu = max(max_gpu, monitor.get_gpu_mem())
                print(f"   -> {i}/{len(indices)}", end='\r')

        results["total_inf_time_sec"] = time.time() - start_inf
        results["avg_ms_per_doc"] = (results["total_inf_time_sec"] / len(indices)) * 1000
        results["peak_cpu_mb"] = max_cpu - mem_start
        results["peak_gpu_mb"] = max_gpu - gpu_start

    except Exception as e:
        print(f"\n   !! Error: {e}")
        return None
    finally:
        del obj
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n   Done: {results['avg_ms_per_doc']:.1f}ms | {results['peak_cpu_mb']:.1f}MB CPU | {results['peak_gpu_mb']:.1f}MB GPU")
    return results

def main():
    if not CSV_PATH.exists():
        print("CSV not found.")
        return
    df = pd.read_csv(CSV_PATH)
    df['langue'] = df['langue'].astype(str).str.lower()

    models = [
        ('topic', 'Qwen-LLM', ['ar', 'en', 'fr']),
        ('sentiment', 'Qwen-LLM', ['ar', 'en', 'fr']),
        # Topic Transformers
        ('topic', 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', ['en']),
        ('topic', 'facebook/bart-large-mnli', ['en']),
        ('topic', 'morit/french_xlm_xnli', ['fr']),
        ('topic', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', ['ar', 'en', 'fr']),
        # Sentiment Transformers
        ('sentiment', 'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment', ['ar']),
        ('sentiment', 'cardiffnlp/twitter-roberta-base-sentiment-latest', ['en']),
        ('sentiment', 'cmarkea/distilcamembert-base-sentiment', ['fr']),
        ('sentiment', 'cardiffnlp/twitter-xlm-roberta-base-sentiment', ['ar', 'en', 'fr']),
        # NER
        ('ner', Config.ARABERT_NER_MODEL, ['ar']),
        ('ner', Config.EN_NER_MODEL, ['en']),
        ('ner', Config.FR_NER_MODEL, ['fr']),
        ('ner', Config.GLINER_MODEL, ['ar', 'en', 'fr']),
    ]

    final_results = []
    
    # PHASE 1: CPU, PHASE 2: GPU
    devices = ["CPU"]
    if torch.cuda.is_available():
        devices.append("GPU")
        
    for device in devices:
        print(f"\n{'='*20} STARTING PHASE: {device} {'='*20}")
        for task, model, langs in models:
            for lang in langs:
                res = benchmark_item(task, model, lang, device, df)
                if res:
                    final_results.append(res)
                    # Periodic save to avoid loss
                    pd.DataFrame(final_results).to_csv(REPORT_FILE, index=False)

    print(f"\nAll benchmarks completed. Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
