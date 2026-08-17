"""
models/transformers/nli_runner.py
---------------------------------
Runner for Zero-Shot NLI Topic Classification models.

Tasks covered:
  - Topic classification (Arabic, English, French, Multilingual NLI models)

Usage:
  python models/transformers/nli_runner.py
"""
from __future__ import annotations

import sys
import io
import warnings
from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .shared import (
    CSV_PATH,
    TOPIC_LABELS,
    chunk_text,
    aggregate_topics,
    reset_cuda,
    get_missing_mask,
    BenchmarkTracker,
    save_benchmark_records,
)

# ---------------------------------------------------------------------------
# Topic models
# ---------------------------------------------------------------------------
TOPIC_MODELS: dict[str, list[str]] = {
    'ar': [
        'AhmedZaky1/arabic-bert-nli-matryoshka',
        'HassanB4/s04-arbert-nli',
    ],
    'en': [
        'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
        'roberta-large-mnli',
        'facebook/bart-large-mnli'
    ],
    'fr': [
        'cmarkea/distilcamembert-base-nli'
    ],
    'multi': [
        'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
        'joeddav/xlm-roberta-large-xnli'
    ]
}

FORCE_RERUN_TOPIC = True


def run_nli_models(
    df: pd.DataFrame | None = None,
    force_rerun_topic: bool = FORCE_RERUN_TOPIC,
    save_csv: bool = True,
    csv_path: Path = CSV_PATH
) -> tuple[pd.DataFrame, dict]:
    """
    Run all Zero-Shot NLI Topic Classification models on the dataset.

    Returns:
      (df, model_status_dict)
    """
    if df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"[NLI Runner] Loading dataset: {csv_path.name}")
        df = pd.read_csv(csv_path, keep_default_na=False)

    df['langue'] = df['langue'].astype(str).str.lower()
    device = 0 if torch.cuda.is_available() else -1
    print(f"[NLI Runner] Device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    model_status: dict = {}
    print(f"\n{'=' * 20} NLI TASK: TOPIC CLASSIFICATION {'=' * 20}")

    for lang, model_list in TOPIC_MODELS.items():
        for model_name in model_list:
            col_name = f"{model_name.split('/')[-1]}_topic_pred"
            track_key = ('TOPIC', lang.upper(), model_name)

            if col_name not in df.columns:
                df[col_name] = None
            df[col_name] = df[col_name].astype(object)

            if force_rerun_topic:
                is_missing = pd.Series(True, index=df.index)
                print(f" -> [{lang.upper()}] FORCE RERUN: {model_name}")
            else:
                is_missing = get_missing_mask(df, col_name)

            target_indices = df.index[is_missing].tolist()
            if lang != 'multi':
                target_indices = [i for i in target_indices if df.loc[i, 'langue'] == lang]

            if not target_indices:
                print(f" -> [{lang.upper()}] Skipping {model_name} (all rows already processed).")
                model_status[track_key] = {'status': 'SKIPPED', 'rows': 0, 'error': None}
                continue

            print(f" -> [{lang.upper()}] Running {model_name} on {len(target_indices)} rows...")

            tracker = BenchmarkTracker(device_override=device)

            try:
                max_len = 512
                overlap = 50

                tracker.start_load()
                model_obj = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer_obj = AutoTokenizer.from_pretrained(model_name)

                current_device = device
                try:
                    if current_device == 0:
                        model_obj = model_obj.to('cuda')
                except RuntimeError:
                    print(f"    WARNING: CUDA failed for {model_name}, switching to CPU")
                    current_device = -1
                    model_obj = model_obj.to('cpu')

                classifier = pipeline(
                    "zero-shot-classification",
                    model=model_obj,
                    tokenizer=tokenizer_obj,
                    device=current_device
                )
                tracker.end_load()

                row_errors = 0
                cuda_failed = False

                tracker.start_inference()
                for i_step, idx in enumerate(target_indices):
                    if cuda_failed:
                        df.loc[idx, col_name] = "ERROR_TOPIC"
                        row_errors += 1
                        continue
                    try:
                        row_lang = df.loc[idx, 'langue']
                        text = str(df.loc[idx, 'texte'])

                        if lang == 'multi':
                            candidate_labels = TOPIC_LABELS.get(row_lang, TOPIC_LABELS['en'])
                        else:
                            candidate_labels = TOPIC_LABELS.get(lang, TOPIC_LABELS['multi'])

                        chunks = chunk_text(text, max_len, overlap)

                        if len(chunks) == 1:
                            result = classifier(chunks[0], candidate_labels, multi_label=False)
                            df.loc[idx, col_name] = result['labels'][0]
                        else:
                            results = []
                            for chunk in chunks:
                                try:
                                    result = classifier(chunk, candidate_labels, multi_label=False)
                                    results.append(result['labels'][0])
                                except Exception:
                                    continue
                            df.loc[idx, col_name] = aggregate_topics(results)

                    except RuntimeError as e:
                        if 'cuda' in str(e).lower() or 'assert' in str(e).lower():
                            print(f"      Row {idx} CUDA error - stopping model: {e}")
                            df.loc[idx, col_name] = "ERROR_TOPIC"
                            row_errors += 1
                            cuda_failed = True
                        else:
                            row_errors += 1
                            print(f"      Row {idx} failed: {e}")
                            df.loc[idx, col_name] = "ERROR_TOPIC"
                    except Exception as e:
                        row_errors += 1
                        print(f"      Row {idx} failed: {e}")
                        df.loc[idx, col_name] = "ERROR_TOPIC"

                    if i_step % 10 == 0:
                        tracker.sample_mem()

                tracker.end_inference(len(target_indices))

                try:
                    del classifier
                    del model_obj
                    del tokenizer_obj
                except Exception:
                    pass
                reset_cuda()

                if save_csv:
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                bench_metrics = tracker.get_metrics()
                bench_record = {
                    "device": bench_metrics["device"],
                    "task": "topic",
                    "model": model_name,
                    "lang": lang,
                    "load_time_sec": bench_metrics["load_time_sec"],
                    "total_inf_time_sec": bench_metrics["total_inf_time_sec"],
                    "avg_ms_per_doc": bench_metrics["avg_ms_per_doc"],
                    "throughput_doc_per_sec": bench_metrics["throughput_doc_per_sec"],
                    "peak_cpu_mb": bench_metrics["peak_cpu_mb"],
                    "peak_gpu_mb": bench_metrics["peak_gpu_mb"]
                }
                save_benchmark_records([bench_record])

                success_rows = len(target_indices) - row_errors
                status_val = 'SUCCESS' if row_errors == 0 else ('PARTIAL' if success_rows > 0 else 'FAILED')
                err_msg = None if row_errors == 0 else (f'{row_errors} row(s) failed' if success_rows > 0 else 'All rows failed')
                model_status[track_key] = {
                    'status': status_val,
                    'rows': success_rows,
                    'error': err_msg,
                    'benchmark': bench_metrics
                }
                print(f"    Done: {bench_metrics['avg_ms_per_doc']:.1f}ms/doc | {bench_metrics['total_inf_time_sec']:.2f}s total | {bench_metrics['throughput_doc_per_sec']:.1f} docs/s")

            except Exception as e:
                print(f"    Failed: {e}")
                model_status[track_key] = {'status': 'FAILED', 'rows': 0, 'error': str(e), 'benchmark': None}
                try:
                    reset_cuda()
                except Exception:
                    pass

    if save_csv:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[NLI Runner] Results saved to: {csv_path}")

    return df, model_status


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Zero-Shot NLI Topic models")
    parser.add_argument("--force-rerun-topic", action=argparse.BooleanOptionalAction, default=FORCE_RERUN_TOPIC, help="Force rerun topic models")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to input CSV")
    args = parser.parse_args()

    df, status = run_nli_models(
        force_rerun_topic=args.force_rerun_topic,
        csv_path=Path(args.csv)
    )
    print("\n[NLI Runner] Done.")


if __name__ == "__main__":
    main()
