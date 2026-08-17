"""
models/transformers/bert_runner.py
----------------------------------
Runner for BERT/RoBERTa/CamemBERT-based Sentiment Analysis and Named Entity Recognition (NER) models.

Tasks covered:
  - Sentiment analysis (Arabic MSA/Dialectal, English, French, Multilingual)
  - Named Entity Recognition (Arabic, English, French, Multilingual/GLiNER)

Usage:
  python models/transformers/bert_runner.py
"""
from __future__ import annotations

import sys
import io
import warnings
from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline

warnings.filterwarnings("ignore")

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ner.ner_extraction import TransformersNER, GLiNERNER
from .shared import (
    CSV_PATH,
    standardize_sentiment,
    format_entities,
    chunk_text,
    aggregate_sentiment,
    reset_cuda,
    get_missing_mask,
    BenchmarkTracker,
    save_benchmark_records,
)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
ARABERT_NER_MODEL = "hatmimoha/arabic-ner"
CAMEL_NER_MODEL   = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
EN_NER_MODEL      = "dslim/bert-base-NER"
FR_NER_MODEL      = "Jean-Baptiste/camembert-ner"
GLINER_MODEL      = "urchade/gliner_multi-v2.1"

SENTIMENT_MODELS: dict[str, list[str]] = {
    'ar': [
        'CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment',
        'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment',
        'CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment',
        'PRAli22/AraBert-Arabic-Sentiment-Analysis'
    ],
    'en': [
        'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'j-hartmann/sentiment-roberta-large-english-3-classes',
        'finiteautomata/bertweet-base-sentiment-analysis'
    ],
    'fr': [
        'cmarkea/distilcamembert-base-sentiment',
        'nlptown/bert-base-multilingual-uncased-sentiment'
    ],
    'multi': [
        'cardiffnlp/twitter-xlm-roberta-base-sentiment',
        'lxyuan/distilbert-base-multilingual-cased-sentiments-student'
    ]
}

NER_MODELS: dict[str, list[str]] = {
    'ar': [
        ARABERT_NER_MODEL,
        CAMEL_NER_MODEL,
        'CAMeL-Lab/bert-base-arabic-camelbert-mix-ner',
        'MostafaAhmed98/AraBert-Arabic-NER-CoNLLpp'
    ],
    'en': [
        EN_NER_MODEL,
        'dslim/bert-large-NER',
        'Jean-Baptiste/roberta-large-ner-english'
    ],
    'fr': [
        FR_NER_MODEL,
        'cmarkea/distilcamembert-base-ner'
    ],
    'multi': [
        GLINER_MODEL,
        'Davlan/bert-base-multilingual-cased-ner-hrl',
        'Babelscape/wikineural-multilingual-ner'
    ]
}
FORCE_RERUN_SENTIMENT = True
FORCE_RERUN_NER = True


def run_bert_models(
    df: pd.DataFrame | None = None,
    force_rerun_sentiment: bool = FORCE_RERUN_SENTIMENT,
    force_rerun_ner: bool = FORCE_RERUN_NER,
    save_csv: bool = True,
    csv_path: Path = CSV_PATH
) -> tuple[pd.DataFrame, dict]:
    """
    Run all BERT-based Sentiment and NER models on the dataset.

    Returns:
      (df, model_status_dict)
    """
    if df is None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"[BERT Runner] Loading dataset: {csv_path.name}")
        df = pd.read_csv(csv_path, keep_default_na=False)

    df['langue'] = df['langue'].astype(str).str.lower()
    device = 0 if torch.cuda.is_available() else -1
    print(f"[BERT Runner] Device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    model_status: dict = {}
    tasks = {
        'sentiment': SENTIMENT_MODELS,
        'ner': NER_MODELS
    }

    for task_name, lang_dict in tasks.items():
        print(f"\n{'=' * 20} BERT TASK: {task_name.upper()} {'=' * 20}")

        for lang, model_list in lang_dict.items():
            for model_name in model_list:
                suffix = "pred" if task_name == 'sentiment' else "ner_pred"
                col_name = f"{model_name.split('/')[-1]}_{suffix}"
                track_key = (task_name.upper(), lang.upper(), model_name)

                if col_name not in df.columns:
                    df[col_name] = None
                df[col_name] = df[col_name].astype(object)

                force_rerun = (task_name == 'sentiment' and force_rerun_sentiment) or \
                              (task_name == 'ner' and force_rerun_ner)

                if force_rerun:
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
                    # -------------------------------------------------------
                    # SENTIMENT PIPELINE
                    # -------------------------------------------------------
                    if task_name == 'sentiment':
                        max_len = 128 if 'bertweet' in model_name.lower() else 512
                        overlap = 32 if 'bertweet' in model_name.lower() else 50

                        tracker.start_load()
                        pipe = pipeline(
                            "sentiment-analysis",
                            model=model_name,
                            device=device,
                            truncation=True,
                            max_length=max_len
                        )
                        tracker.end_load()

                        row_errors = 0
                        tracker.start_inference()
                        for i_step, idx in enumerate(target_indices):
                            try:
                                text = str(df.loc[idx, 'texte'])
                                chunks = chunk_text(text, max_len, overlap)
                                if len(chunks) == 1:
                                    result = pipe(chunks[0])[0]
                                    df.loc[idx, col_name] = standardize_sentiment(result['label'], model_name)
                                else:
                                    results = []
                                    for chunk in chunks:
                                        try:
                                            result = pipe(chunk)[0]
                                            results.append(standardize_sentiment(result['label'], model_name))
                                        except Exception:
                                            continue
                                    df.loc[idx, col_name] = aggregate_sentiment(results)
                            except Exception as e:
                                row_errors += 1
                                print(f"      Row {idx} failed: {e}")
                                df.loc[idx, col_name] = "ERROR_SENTIMENT"

                            if i_step % 10 == 0:
                                tracker.sample_mem()

                        tracker.end_inference(len(target_indices))
                        del pipe
                        reset_cuda()

                    # -------------------------------------------------------
                    # NER PIPELINE
                    # -------------------------------------------------------
                    elif task_name == 'ner':
                        tracker.start_load()
                        if 'gliner' in model_name.lower():
                            ner_model = GLiNERNER(model_name=model_name, device=device)
                        else:
                            ner_model = TransformersNER(model_name=model_name, device=device)
                        tracker.end_load()

                        row_errors = 0
                        tracker.start_inference()
                        for i_step, idx in enumerate(target_indices):
                            try:
                                row_lang = df.loc[idx, 'langue']
                                text = str(df.loc[idx, 'texte'])
                                if isinstance(ner_model, GLiNERNER):
                                    entities = ner_model.predict(text, language=row_lang)
                                else:
                                    chunks = chunk_text(text, 512, 50)
                                    if len(chunks) == 1:
                                        entities = ner_model.predict(text)
                                    else:
                                        all_entities = []
                                        seen = set()
                                        for chunk in chunks:
                                            try:
                                                ents = ner_model.predict(chunk)
                                                for e in ents:
                                                    key = f"{e.text.lower()}_{e.label}"
                                                    if key not in seen:
                                                        seen.add(key)
                                                        all_entities.append(e)
                                            except Exception:
                                                continue
                                        entities = all_entities
                                df.loc[idx, col_name] = format_entities(entities)
                            except Exception as e:
                                row_errors += 1
                                print(f"      Row {idx} failed: {e}")
                                df.loc[idx, col_name] = "ERROR_NER"

                            if i_step % 10 == 0:
                                tracker.sample_mem()

                        tracker.end_inference(len(target_indices))
                        del ner_model
                        reset_cuda()

                    if save_csv:
                        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                    bench_metrics = tracker.get_metrics()
                    bench_record = {
                        "device": bench_metrics["device"],
                        "task": task_name,
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
        print(f"\n[BERT Runner] Results saved to: {csv_path}")

    return df, model_status


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run BERT Sentiment and NER models")
    parser.add_argument("--force-rerun-sentiment", action=argparse.BooleanOptionalAction, default=FORCE_RERUN_SENTIMENT, help="Force rerun sentiment models")
    parser.add_argument("--force-rerun-ner", action=argparse.BooleanOptionalAction, default=FORCE_RERUN_NER, help="Force rerun NER models")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to input CSV")
    args = parser.parse_args()

    df, status = run_bert_models(
        force_rerun_sentiment=args.force_rerun_sentiment,
        force_rerun_ner=args.force_rerun_ner,
        csv_path=Path(args.csv)
    )
    print("\n[BERT Runner] Done.")


if __name__ == "__main__":
    main()
