"""
benchmark/master_benchmark.py
-----------------------------
Master standalone benchmarking script to measure CPU and GPU resource consumption,
load latency, inference latency, throughput, and memory deltas across all models.

Usage:
    python benchmark/master_benchmark.py
"""
from __future__ import annotations

import sys
import io
import os
import time
import psutil
import requests
from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline

# Force stdout to UTF-8 for console output
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ner.ner_extraction import TransformersNER, GLiNERNER
from models.transformers.bert_runner import SENTIMENT_MODELS, NER_MODELS
from models.transformers.nli_runner import TOPIC_MODELS
from models.transformers.shared import (
    CSV_PATH,
    BENCHMARK_REPORT_PATH,
    TOPIC_LABELS,
    reset_cuda,
)


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_cpu_mem(self) -> float:
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def get_gpu_mem(self) -> float:
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
        num_gpu = 1 if self.use_gpu else 0
        payload = {
            "model": self.model_name,
            "prompt": f"Task: {self.task}, Lang: {lang}, Text: {text[:200]}",
            "stream": False,
            "options": {"num_gpu": num_gpu, "num_predict": 10, "temperature": 0}
        }
        try:
            requests.post(self.url, json=payload, timeout=30)
        except Exception:
            pass


def benchmark_item(task_name: str, model_name: str, lang: str, device_type: str, df: pd.DataFrame) -> dict | None:
    monitor = ResourceMonitor()
    device_idx = 0 if device_type == "GPU" and torch.cuda.is_available() else -1

    indices = df.index[df['langue'] == lang].tolist() if lang != 'multi' else df.index.tolist()
    if not indices:
        return None

    print(f"[{device_type}] {task_name.upper()} | {model_name} | {lang} ({len(indices)} docs)")

    results = {
        "device": device_type,
        "task": task_name,
        "model": model_name,
        "lang": lang,
        "load_time_sec": 0.0,
        "total_inf_time_sec": 0.0,
        "avg_ms_per_doc": 0.0,
        "throughput_doc_per_sec": 0.0,
        "peak_cpu_mb": 0.0,
        "peak_gpu_mb": 0.0
    }

    mem_start = monitor.get_cpu_mem()
    gpu_start = monitor.get_gpu_mem()
    start_load = time.time()

    obj = None
    try:
        if model_name == "Qwen-LLM":
            obj = BenchLLM(task_name, use_gpu=(device_type == "GPU"))
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

        results["load_time_sec"] = round(time.time() - start_load, 2)

        start_inf = time.time()
        max_cpu = mem_start
        max_gpu = gpu_start

        for i, idx in enumerate(indices):
            text = str(df.loc[idx, 'texte'])
            row_lang = str(df.loc[idx, 'langue'])

            if model_name == "Qwen-LLM":
                obj.predict(text, row_lang)
            elif task_name == 'topic':
                labels = TOPIC_LABELS.get(row_lang, TOPIC_LABELS['en'])
                obj(text[:1000], candidate_labels=labels)
            elif task_name == 'sentiment':
                obj(text[:1000])
            elif task_name == 'ner':
                obj.predict(text[:1000])

            if i % 10 == 0:
                max_cpu = max(max_cpu, monitor.get_cpu_mem())
                max_gpu = max(max_gpu, monitor.get_gpu_mem())
                print(f"   -> {i+1}/{len(indices)}", end='\r')

        total_inf = time.time() - start_inf
        results["total_inf_time_sec"] = round(total_inf, 2)
        results["avg_ms_per_doc"] = round((total_inf / len(indices)) * 1000, 1) if indices else 0.0
        results["throughput_doc_per_sec"] = round(len(indices) / total_inf, 1) if total_inf > 0 else 0.0
        results["peak_cpu_mb"] = round(max(0.0, max_cpu - mem_start), 1)
        results["peak_gpu_mb"] = round(max(0.0, max_gpu - gpu_start), 1)

    except Exception as e:
        print(f"\n   !! Error: {e}")
        return None
    finally:
        del obj
        reset_cuda()

    print(f"\n   Done: {results['avg_ms_per_doc']:.1f}ms | {results['peak_cpu_mb']:.1f}MB CPU | {results['peak_gpu_mb']:.1f}MB GPU")
    return results


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    df['langue'] = df['langue'].astype(str).str.lower()

    # Dynamically build model list from canonical definitions
    models: list[tuple[str, str, list[str]]] = [
        ('topic', 'Qwen-LLM', ['ar', 'en', 'fr']),
        ('sentiment', 'Qwen-LLM', ['ar', 'en', 'fr']),
    ]

    for lang, m_list in TOPIC_MODELS.items():
        for m in m_list:
            models.append(('topic', m, ['ar', 'en', 'fr'] if lang == 'multi' else [lang]))

    for lang, m_list in SENTIMENT_MODELS.items():
        for m in m_list:
            models.append(('sentiment', m, ['ar', 'en', 'fr'] if lang == 'multi' else [lang]))

    for lang, m_list in NER_MODELS.items():
        for m in m_list:
            models.append(('ner', m, ['ar', 'en', 'fr'] if lang == 'multi' else [lang]))

    existing_results: list[dict] = []
    done: set[tuple] = set()
    if BENCHMARK_REPORT_PATH.exists():
        try:
            existing_df = pd.read_csv(BENCHMARK_REPORT_PATH)
            existing_results = existing_df.to_dict('records')
            for row in existing_results:
                done.add((str(row.get('device')), str(row.get('task')),
                          str(row.get('model')),  str(row.get('lang'))))
            print(f"[Append mode] {len(existing_results)} existing result(s) found — already-benchmarked entries will be skipped.")
        except Exception as e:
            print(f"[Append mode] Could not load existing results ({e}). Starting fresh.")

    new_results: list[dict] = []

    devices = ["CPU"]
    if torch.cuda.is_available():
        devices.append("GPU")

    for device in devices:
        print(f"\n{'='*20} STARTING PHASE: {device} {'='*20}")
        for task, model, langs in models:
            for lang in langs:
                key = (device, task, model, lang)
                if key in done:
                    print(f"  [SKIP] {device} | {task} | {model.split('/')[-1]} | {lang}")
                    continue
                res = benchmark_item(task, model, lang, device, df)
                if res:
                    new_results.append(res)
                    done.add(key)
                    pd.DataFrame(existing_results + new_results).to_csv(BENCHMARK_REPORT_PATH, index=False)

    pd.DataFrame(existing_results + new_results).to_csv(BENCHMARK_REPORT_PATH, index=False)
    print(f"\nAll benchmarks completed. Report saved to {BENCHMARK_REPORT_PATH}")


if __name__ == "__main__":
    main()
