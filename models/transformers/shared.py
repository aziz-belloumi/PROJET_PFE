"""
models/transformers/shared.py
------------------------------
Shared constants, helpers, and utilities used by both bert_runner.py and nli_runner.py.
"""
import time
import os
import torch
from pathlib import Path
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR              = PROJECT_ROOT / "data"
REPORTS_DIR           = PROJECT_ROOT / "reports"
BENCHMARK_DIR         = PROJECT_ROOT / "benchmark"
CSV_PATH              = DATA_DIR / "manual_eval_global.csv"
REPORT_PATH           = REPORTS_DIR / "report.txt"
BENCHMARK_REPORT_PATH = BENCHMARK_DIR / "resource_usage_report.csv"

# Number of Arabic MSA articles (first N rows of the AR subset)
MSA_COUNT = 150


# ---------------------------------------------------------------------------
# Benchmark & Resource Tracker
# ---------------------------------------------------------------------------
class BenchmarkTracker:
    """Tracks latency, throughput, and CPU/GPU memory deltas for a model run."""

    def __init__(self, device_override: int | str | None = None):
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        if device_override is not None:
            if isinstance(device_override, int):
                self.device_name = f"GPU (cuda:{device_override})" if device_override >= 0 else "CPU"
            else:
                self.device_name = str(device_override)
        else:
            self.device_name = "GPU (cuda:0)" if torch.cuda.is_available() else "CPU"

        self.cpu_start = self.get_cpu_mem()
        self.gpu_start = self.get_gpu_mem()
        self.max_cpu = self.cpu_start
        self.max_gpu = self.gpu_start

        self.load_start = 0.0
        self.load_time = 0.0
        self.inf_start = 0.0
        self.inf_time = 0.0
        self.count = 0

    def get_cpu_mem(self) -> float:
        if self.process:
            try:
                return self.process.memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def get_gpu_mem(self) -> float:
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def start_load(self):
        self.load_start = time.time()

    def end_load(self):
        self.load_time = max(0.0, time.time() - self.load_start)
        self.sample_mem()

    def start_inference(self):
        self.inf_start = time.time()

    def sample_mem(self):
        self.max_cpu = max(self.max_cpu, self.get_cpu_mem())
        self.max_gpu = max(self.max_gpu, self.get_gpu_mem())

    def end_inference(self, count: int):
        self.inf_time = max(0.0, time.time() - self.inf_start)
        self.count = count
        self.sample_mem()

    def get_metrics(self) -> dict:
        avg_ms = (self.inf_time / self.count * 1000) if self.count > 0 else 0.0
        throughput = (self.count / self.inf_time) if self.inf_time > 0 else 0.0
        peak_cpu_delta = max(0.0, self.max_cpu - self.cpu_start)
        peak_gpu_delta = max(0.0, self.max_gpu - self.gpu_start)
        return {
            "device": self.device_name,
            "load_time_sec": round(self.load_time, 2),
            "total_inf_time_sec": round(self.inf_time, 2),
            "avg_ms_per_doc": round(avg_ms, 1),
            "throughput_doc_per_sec": round(throughput, 1),
            "peak_cpu_mb": round(peak_cpu_delta, 1),
            "peak_gpu_mb": round(peak_gpu_delta, 1),
        }


def save_benchmark_records(records: list[dict], csv_path: Path = BENCHMARK_REPORT_PATH):
    """Save or append benchmark timing and resource usage records to CSV."""
    if not records:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(records)

    if csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path)
            # Merge on (device, task, model, lang) if present, replacing old entries
            key_cols = [c for c in ['device', 'task', 'model', 'lang'] if c in old_df.columns and c in new_df.columns]
            if key_cols:
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=key_cols, keep='last')
                combined.to_csv(csv_path, index=False)
                return
        except Exception:
            pass

    new_df.to_csv(csv_path, index=False)


def load_benchmark_records(csv_path: Path = BENCHMARK_REPORT_PATH) -> dict[tuple[str, str], dict]:
    """Load benchmark records indexed by (task, model_short_name)."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        lookup: dict[tuple[str, str], dict] = {}
        for _, row in df.iterrows():
            task = str(row.get('task', '')).strip().lower()
            model_full = str(row.get('model', '')).strip()
            model_short = model_full.split('/')[-1]
            rec = row.to_dict()
            lookup[(task, model_full)] = rec
            lookup[(task, model_short)] = rec
        return lookup
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Model name → label mapping (for BERT sentiment models that return LABEL_X)
# ---------------------------------------------------------------------------
MODEL_LABEL_MAP = {
    'camelbert-msa-sentiment': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'camelbert-da-sentiment':  {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'camelbert-mix-sentiment': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE', 'LABEL_2': 'NEUTRAL'},
    'AraBert-Arabic-Sentiment-Analysis': {
        'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEGATIVE',
        'LABEL_2': 'NEUTRAL',  'LABEL_3': 'NEUTRAL'
    },
    'twitter-roberta-base-sentiment-latest':            {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL',   'LABEL_2': 'POSITIVE'},
    'sentiment-roberta-large-english-3-classes':        {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL',   'LABEL_2': 'POSITIVE'},
    'bertweet-base-sentiment-analysis':                 {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL',   'LABEL_2': 'POSITIVE'},
    'distilcamembert-base-sentiment':                   {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEGATIVE',  'LABEL_2': 'NEUTRAL',   'LABEL_3': 'POSITIVE', 'LABEL_4': 'POSITIVE'},
    'bert-base-multilingual-uncased-sentiment':         {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEGATIVE',  'LABEL_2': 'NEUTRAL',   'LABEL_3': 'POSITIVE', 'LABEL_4': 'POSITIVE'},
    'twitter-xlm-roberta-base-sentiment':               {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'NEUTRAL',   'LABEL_2': 'POSITIVE'},
    'distilbert-base-multilingual-cased-sentiments-student': {'LABEL_0': 'POSITIVE', 'LABEL_1': 'NEUTRAL', 'LABEL_2': 'NEGATIVE'},
}

# ---------------------------------------------------------------------------
# Candidate topic labels per language (for zero-shot NLI topic models)
# ---------------------------------------------------------------------------
TOPIC_LABELS = {
    'en': ['Politics', 'Economy', 'Security', 'Energy', 'Conflict', 'Elections',
           'Justice', 'Health', 'Weather', 'Sports', 'Culture', 'Education',
           'Technology', 'Environment', 'Diplomacy', 'Religion', 'Migration', 'General'],
    'ar': ['السياسة', 'الاقتصاد', 'الأمن', 'الطاقة', 'النزاع', 'الانتخابات',
           'العدالة', 'الصحة', 'الطقس', 'الرياضة', 'الثقافة', 'التعليم',
           'التكنولوجيا', 'البيئة', 'الدبلوماسية', 'الدين', 'الهجرة', 'عام'],
    'da': ['السياسة', 'الاقتصاد', 'الأمن', 'الطاقة', 'النزاع', 'الانتخابات',
           'العدالة', 'الصحة', 'الطقس', 'الرياضة', 'الثقافة', 'التعليم',
           'التكنولوجيا', 'البيئة', 'الدبلوماسية', 'الدين', 'الهجرة', 'عام'],
    'fr': ['Politique', 'Économie', 'Sécurité', 'Énergie', 'Conflit', 'Élections',
           'Justice', 'Santé', 'Météo', 'Sport', 'Culture', 'Éducation',
           'Technologie', 'Environnement', 'Diplomatie', 'Religion', 'Migration', 'Général'],
    'multi': ['Politics', 'Economy', 'Security', 'Energy', 'Conflict', 'Elections',
              'Justice', 'Health', 'Weather', 'Sports', 'Culture', 'Education',
              'Technology', 'Environment', 'Diplomacy', 'Religion', 'Migration', 'General'],
}

# ---------------------------------------------------------------------------
# Text utilities & Column Helpers
# ---------------------------------------------------------------------------

def find_column(df, candidates: list[str]) -> str | None:
    """Find a column in df matching any candidate name or substring."""
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def normalize_topic_label(label: str) -> str:
    """
    Standardize minor topic spelling/accent variations across English, French, and Arabic.
    E.g. 'Economie' -> 'économie', 'العدل' -> 'العدالة'.
    """
    if not isinstance(label, str):
        return ""
    s = label.strip().lower()

    # Accent/spelling normalization maps.
    # Keys are un-accented or variant lowercase forms found in raw model outputs
    # or ground-truth labels. Values are canonical lowercase forms.
    #
    # NOTE: some keys (e.g. 'elections', 'general') also match bare English words.
    # This is intentional and safe: this function is ALWAYS applied symmetrically
    # to both y_true and y_pred before comparison, so the mapping never creates
    # false matches or hides true mismatches between prediction and ground truth.
    fr_map = {
        'economie': 'économie',
        'energie': 'énergie',
        'elections': 'élections',
        'general': 'général',
        'securite': 'sécurité',
        'sante': 'santé',
        'education': 'éducation',
        'meteo': 'météo',
    }
    if s in fr_map:
        return fr_map[s]

    # Arabic alias fixes (e.g. informal spelling → canonical form)
    ar_map = {
        'العدل': 'العدالة',
    }
    if s in ar_map:
        return ar_map[s]

    return s


def standardize_sentiment(label: str, model_name: str = None) -> str:
    """Map raw model output label to POSITIVE / NEGATIVE / NEUTRAL."""
    if not isinstance(label, str):
        return "NEUTRAL"
    l_upper = label.upper().strip()
    if l_upper.startswith("LABEL_") and model_name is not None:
        for pattern, mapping in MODEL_LABEL_MAP.items():
            if pattern.lower() in model_name.lower():
                return mapping.get(l_upper, "NEUTRAL")
    l = label.lower()
    if 'star' in l:
        if '1' in l or '2' in l: return 'NEGATIVE'
        if '3' in l:              return 'NEUTRAL'
        return 'POSITIVE'
    if 'pos' in l:            return 'POSITIVE'
    if 'neg' in l:            return 'NEGATIVE'
    if 'neu' in l or 'mixed' in l: return 'NEUTRAL'
    # Unknown label — warn explicitly so mapping gaps are visible during evaluation
    import warnings
    warnings.warn(
        f"[standardize_sentiment] Unrecognised label '{label}' "
        f"for model '{model_name}' — defaulting to NEUTRAL",
        stacklevel=2
    )
    return "NEUTRAL"


def format_entities(entities) -> str:
    """Serialize a list of NEREntity objects to a pipe-separated string."""
    if not entities:
        return "None"
    return " | ".join(f"{e.text} ({e.label})" for e in entities)


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> list[str]:
    """Character-based chunking for models with token limits."""
    if not text or len(text) <= max_length * 4:
        return [text]
    chunks = []
    chunk_size = max_length * 4
    stride = chunk_size - (overlap * 4)
    for i in range(0, len(text), stride):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks


def aggregate_sentiment(results: list[str]) -> str:
    """Majority-vote aggregation over chunk-level sentiment predictions."""
    if not results: return "NEUTRAL"
    if len(results) == 1: return results[0]
    counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for r in results:
        if r in counts:
            counts[r] += 1
    return max(counts, key=counts.get)


def aggregate_topics(results: list[str]) -> str:
    """Majority-vote aggregation over chunk-level topic predictions."""
    if not results: return "ERROR_TOPIC"
    if len(results) == 1: return results[0]
    counts: dict[str, int] = {}
    for topic in results:
        counts[topic] = counts.get(topic, 0) + 1
    return max(counts, key=counts.get)


def reset_cuda():
    """Free CUDA memory between model runs."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        except RuntimeError as e:
            print(f"  WARNING: CUDA reset failed: {e}")


def get_missing_mask(df, col_name: str):
    """Return a boolean Series marking rows that still need processing."""
    import pandas as pd
    return (
        df[col_name].isna() |
        (df[col_name].astype(str).str.strip() == "") |
        (df[col_name].astype(str).str.strip().str.lower() == "nan") |
        (df[col_name].astype(str).str.strip() == "ERROR_SENTIMENT") |
        (df[col_name].astype(str).str.strip() == "ERROR_NER") |
        (df[col_name].astype(str).str.strip() == "ERROR_TOPIC")
    )
