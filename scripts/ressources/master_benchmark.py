import io
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import psutil
import requests
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.ner import GLiNERNER, TransformersNER
from src.sentiment import LLMSentiment
from src.topic import LLMTopic

CSV_CANDIDATES = [
    PROJECT_ROOT / "data" / "manual_eval_global.csv",
    PROJECT_ROOT / "scripts" / "manual_eval_global.csv",
    PROJECT_ROOT / "manual_eval_global.csv",
]


def get_topic_labels(lang: str) -> List[str]:
    lang = (lang or "en").strip().lower()
    lang = lang if lang in {"ar", "en", "fr"} else "en"
    return [cats[lang] for cats in Config.CATEGORY_DISPLAY.values()]


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_cpu_mem_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def get_gpu_mem_mb(self) -> float:
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0


class BenchLLM:
    def __init__(self, task: str, use_gpu: bool):
        self.task = task
        self.use_gpu = use_gpu
        self.url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2.5:7b"

    def predict(self, text: str, lang: str):
        payload = {
            "model": self.model_name,
            "prompt": f"Task: {self.task}, Lang: {lang}, Text: {text[:200]}",
            "stream": False,
            "options": {"num_gpu": 1 if self.use_gpu else 0, "num_predict": 10, "temperature": 0},
        }
        try:
            requests.post(self.url, json=payload, timeout=30)
        except Exception:
            pass


def benchmark_item(task_name: str, model_name: str, lang: str, device_type: str, df: pd.DataFrame):
    monitor = ResourceMonitor()
    device_idx = 0 if device_type == "GPU" and torch.cuda.is_available() else -1
    indices = df.index[df["langue"].astype(str).str.lower() == lang].tolist()
    if not indices:
        return None

    print(f"[{device_type}] {task_name.upper()} | {model_name} | {lang} ({len(indices)} docs)")

    results: Dict[str, Any] = {
        "device": device_type,
        "task": task_name,
        "model": model_name,
        "lang": lang,
        "load_time_sec": 0.0,
        "total_inf_time_sec": 0.0,
        "avg_ms_per_doc": 0.0,
        "peak_gpu_mb": 0.0,
    }

    gpu_start = monitor.get_gpu_mem_mb()
    start_load = time.perf_counter()
    obj: Any = None

    try:
        if model_name == "Qwen-LLM":
            obj = BenchLLM(task_name, use_gpu=(device_type == "GPU"))
        elif task_name == "topic":
            obj = LLMTopic(device=device_idx)
        elif task_name == "sentiment":
            obj = LLMSentiment(device=device_idx)
        elif task_name == "ner":
            if "gliner" in model_name.lower():
                obj = GLiNERNER(model_name=model_name, device=device_idx)
            else:
                obj = TransformersNER(model_name=model_name, device=device_idx)

        results["load_time_sec"] = time.perf_counter() - start_load

        start_inf = time.perf_counter()
        max_gpu = gpu_start

        for i, idx in enumerate(indices):
            text = str(df.loc[idx, "texte"])

            if model_name == "Qwen-LLM":
                obj.predict(text, lang)
            elif task_name == "topic":
                obj.predict(text, lang)
            elif task_name == "sentiment":
                obj.predict(text, lang)
            elif task_name == "ner":
                obj.predict(text[:1000])

            if i % 10 == 0:
                max_gpu = max(max_gpu, monitor.get_gpu_mem_mb())
                print(f"   -> {i}/{len(indices)}", end="\r")

        results["total_inf_time_sec"] = time.perf_counter() - start_inf
        results["avg_ms_per_doc"] = (results["total_inf_time_sec"] / len(indices)) * 1000
        results["peak_gpu_mb"] = max_gpu - gpu_start

    except Exception as exc:
        print(f"\n   !! Error: {exc}")
        return None
    finally:
        del obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n   Done: {results['avg_ms_per_doc']:.1f}ms | {results['peak_gpu_mb']:.1f}MB GPU")
    return results


def main():
    csv_path = next((path for path in CSV_CANDIDATES if path.exists()), None)
    if csv_path is None:
        print("No evaluation CSV found. Expected one of:")
        for candidate in CSV_CANDIDATES:
            print(f"  - {candidate}")
        print("\nBenchmark not run because there is no dataset to profile.")
        return

    print(f"Loading CSV: {csv_path.name}")
    df = pd.read_csv(csv_path)
    if "langue" not in df.columns or "texte" not in df.columns:
        raise ValueError(f"CSV must contain 'langue' and 'texte' columns. Found: {list(df.columns[:10])}")

    df["langue"] = df["langue"].astype(str).str.lower()

    models = [
        ("topic", "Qwen-LLM", ["ar", "en", "fr"]),
        ("sentiment", "Qwen-LLM", ["ar", "en", "fr"]),
        ("topic", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", ["en"]),
        ("topic", "facebook/bart-large-mnli", ["en"]),
        ("topic", "morit/french_xlm_xnli", ["fr"]),
        ("topic", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", ["ar", "en", "fr"]),
        ("sentiment", Config.FINETUNED_SENTIMENT_MODELS["ar"], ["ar"]),
        ("sentiment", Config.FINETUNED_SENTIMENT_MODELS["en"], ["en"]),
        ("sentiment", Config.FINETUNED_SENTIMENT_MODELS["fr"], ["fr"]),
        ("ner", Config.ARABERT_NER_MODEL, ["ar"]),
        ("ner", "dslim/bert-base-NER", ["en"]),
        ("ner", "Jean-Baptiste/camembert-ner", ["fr"]),
        ("ner", Config.GLINER_MODEL, ["ar", "en", "fr"]),
    ]

    final_results: List[Dict[str, Any]] = []
    devices = ["CPU"]
    if torch.cuda.is_available():
        devices.append("GPU")

    for device in devices:
        print(f"\n{'=' * 20} STARTING PHASE: {device} {'=' * 20}")
        for task, model, langs in models:
            for lang in langs:
                res = benchmark_item(task, model, lang, device, df)
                if res:
                    final_results.append(res)

    print("\nAll benchmarks completed.")


if __name__ == "__main__":
    main()
