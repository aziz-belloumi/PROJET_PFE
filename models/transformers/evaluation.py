"""
models/transformers/evaluation.py
---------------------------------
Evaluation, metrics computation (Accuracy, F1 Macro, Precision, Recall),
distribution analysis (Sentiment on MSA/Dialectal/FR/EN, Topic on 600 articles),
and comprehensive report generation for all benchmarked models.
"""
from __future__ import annotations

import io
import sys
import pandas as pd
from pathlib import Path

# Try importing sklearn metrics
try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .shared import (
    CSV_PATH,
    REPORT_PATH,
    BENCHMARK_REPORT_PATH,
    MSA_COUNT,
    find_column,
    normalize_topic_label,
    load_benchmark_records,
)

# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def calculate_accuracy(df: pd.DataFrame, pred_col: str, true_col: str | None = None) -> tuple[float | None, int]:
    """Calculate simple accuracy (% of correct predictions)."""
    if true_col is None:
        true_col = find_column(df, ['sentiment attendu', 'sentiment'])
    if true_col is None or true_col not in df.columns or pred_col not in df.columns:
        return None, 0
    pred_std = df[pred_col].astype(str).str.strip().str.upper()
    true_std = df[true_col].astype(str).str.strip().str.upper()
    valid_mask = true_std.isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    valid_mask = valid_mask & ~pred_std.isin(['ERROR_SENTIMENT', 'NAN', ''])
    if valid_mask.sum() == 0:
        return None, 0
    correct = (pred_std[valid_mask] == true_std[valid_mask]).sum()
    total = int(valid_mask.sum())
    return (correct / total * 100), total


def calculate_sentiment_metrics(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str | None = None
) -> tuple[float | None, float | None, float | None, float | None, int]:
    """
    Returns (accuracy_pct, f1_macro, precision_macro, recall_macro, valid_count).
    Falls back to (acc, None, None, None, n) if sklearn is not installed.
    """
    if true_col is None:
        true_col = find_column(df, ['sentiment attendu', 'sentiment'])
    if true_col is None or true_col not in df.columns or pred_col not in df.columns or not SKLEARN_AVAILABLE:
        acc, n = calculate_accuracy(df, pred_col, true_col)
        return acc, None, None, None, n

    pred_std = df[pred_col].astype(str).str.strip().str.upper()
    true_std = df[true_col].astype(str).str.strip().str.upper()

    valid_mask = true_std.isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
    valid_mask = valid_mask & ~pred_std.isin(['ERROR_SENTIMENT', 'NAN', ''])

    if valid_mask.sum() == 0:
        return None, None, None, None, 0

    y_true = true_std[valid_mask].tolist()
    y_pred = pred_std[valid_mask].tolist()
    n = len(y_true)

    labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = (correct / n) * 100

    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100
    prec = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0) * 100

    return acc, f1, prec, rec, n


def calculate_topic_accuracy(df: pd.DataFrame, pred_col: str, true_col: str | None = None) -> tuple[float | None, int]:
    """Calculate simple accuracy for topic predictions."""
    if true_col is None:
        true_col = find_column(df, ['thème attendu', 'theme attendu', 'topic attendu', 'topic'])
    if true_col is None or true_col not in df.columns or pred_col not in df.columns:
        return None, 0

    pred_std = df[pred_col].astype(str).map(normalize_topic_label)
    true_std = df[true_col].astype(str).map(normalize_topic_label)

    valid_mask = (true_std != '') & (true_std != 'nan') & ~true_std.isna()
    valid_mask = valid_mask & (pred_std != '') & (pred_std != 'nan') & (pred_std != 'error_topic') & ~pred_std.isna()

    if valid_mask.sum() == 0:
        return None, 0

    correct = (pred_std[valid_mask] == true_std[valid_mask]).sum()
    total = int(valid_mask.sum())
    return (correct / total * 100), total


def calculate_topic_metrics(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str | None = None
) -> tuple[float | None, float | None, float | None, float | None, int]:
    """
    Returns (accuracy_pct, f1_macro, precision_macro, recall_macro, valid_count) for topic models.
    Falls back to (acc, None, None, None, n) if sklearn is not installed.
    """
    if true_col is None:
        true_col = find_column(df, ['thème attendu', 'theme attendu', 'topic attendu', 'topic'])
    if true_col is None or true_col not in df.columns or pred_col not in df.columns or not SKLEARN_AVAILABLE:
        acc, n = calculate_topic_accuracy(df, pred_col, true_col)
        return acc, None, None, None, n

    pred_std = df[pred_col].astype(str).map(normalize_topic_label)
    true_std = df[true_col].astype(str).map(normalize_topic_label)

    valid_mask = (true_std != '') & (true_std != 'nan') & ~true_std.isna()
    valid_mask = valid_mask & (pred_std != '') & (pred_std != 'nan') & (pred_std != 'error_topic') & ~pred_std.isna()

    if valid_mask.sum() == 0:
        return None, None, None, None, 0

    y_true = true_std[valid_mask].tolist()
    y_pred = pred_std[valid_mask].tolist()
    n = len(y_true)

    all_labels = sorted(list(set(y_true) | set(y_pred)))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = (correct / n) * 100

    f1 = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0) * 100
    prec = precision_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0) * 100

    return acc, f1, prec, rec, n


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_metric(value: float | None, suffix: str = "%") -> str:
    """Format a metric float or return 'N/A'."""
    if value is None:
        return "N/A"
    return f"{value:.1f}{suffix}"


def append_metrics_block(
    report_lines: list[str],
    accuracy: float | None,
    f1: float | None,
    precision: float | None,
    recall: float | None,
    valid_count: int
):
    """Append standard accuracy, F1 Macro, Precision, Recall lines."""
    if accuracy is not None:
        report_lines.append(f"  Accuracy  : {fmt_metric(accuracy)} ({valid_count} valid rows with ground truth)")
        if SKLEARN_AVAILABLE:
            report_lines.append(f"  F1 Macro  : {fmt_metric(f1)}")
            report_lines.append(f"  Precision : {fmt_metric(precision)}")
            report_lines.append(f"  Recall    : {fmt_metric(recall)}")
    else:
        report_lines.append("  Accuracy  : N/A (no valid ground truth rows)")
        if SKLEARN_AVAILABLE:
            report_lines.append("  F1 Macro  : N/A")
            report_lines.append("  Precision : N/A")
            report_lines.append("  Recall    : N/A")


def find_benchmark_for_model(
    task: str,
    model_col: str,
    lang: str,
    model_status: dict | None,
    persisted_benchmarks: dict[tuple[str, str], dict] | None
) -> dict | None:
    """Find benchmark metrics from live model_status dict or persisted CSV records."""
    # 1. Check live model_status dict first
    if model_status:
        for (st_task, st_lang, st_model), info in model_status.items():
            if st_task.lower() == task.lower() and info.get('benchmark'):
                m_short = st_model.split('/')[-1]
                if m_short in model_col or st_model in model_col:
                    return info['benchmark']

    # 2. Check persisted CSV lookup
    if persisted_benchmarks:
        m_name = model_col.replace('_pred', '').replace('_ner_pred', '').replace('_topic_pred', '')
        for (b_task, b_model), b_data in persisted_benchmarks.items():
            if b_task == task.lower():
                b_short = b_model.split('/')[-1]
                if b_short == m_name or b_short in model_col or m_name in b_model:
                    def _safe_float(k):
                        v = b_data.get(k)
                        return float(v) if (pd.notna(v) and v is not None and v != '') else None

                    return {
                        "device": str(b_data.get('device', 'N/A')),
                        "load_time_sec": _safe_float('load_time_sec'),
                        "total_inf_time_sec": _safe_float('total_inf_time_sec'),
                        "avg_ms_per_doc": _safe_float('avg_ms_per_doc'),
                        "throughput_doc_per_sec": _safe_float('throughput_doc_per_sec'),
                        "peak_cpu_mb": _safe_float('peak_cpu_mb'),
                        "peak_gpu_mb": _safe_float('peak_gpu_mb'),
                    }
    return None


def append_benchmark_block(report_lines: list[str], bench: dict | None):
    """Append integrated benchmark and resource usage metrics under a model."""
    if not bench:
        return
    dev = bench.get('device', 'N/A')
    load_s = bench.get('load_time_sec')
    inf_s = bench.get('total_inf_time_sec')
    avg_ms = bench.get('avg_ms_per_doc')
    tput = bench.get('throughput_doc_per_sec')
    cpu_mb = bench.get('peak_cpu_mb')
    gpu_mb = bench.get('peak_gpu_mb')

    report_lines.append("  Benchmark & Resource Metrics:")
    report_lines.append(f"    Device          : {dev}")
    if load_s is not None and load_s > 0:
        report_lines.append(f"    Load Time       : {load_s:.2f}s")
    if inf_s is not None and avg_ms is not None:
        tput_str = f" | {tput:.1f} docs/sec" if (tput is not None and tput > 0) else ""
        report_lines.append(f"    Inference Time  : {inf_s:.2f}s ({avg_ms:.1f} ms/doc{tput_str})")
    if (cpu_mb is not None and cpu_mb > 0) or (gpu_mb is not None and gpu_mb > 0):
        gpu_str = f" | GPU +{gpu_mb:.1f} MB" if (gpu_mb is not None and gpu_mb > 0) else ""
        report_lines.append(f"    Peak Memory     : CPU +{cpu_mb or 0.0:.1f} MB{gpu_str}")


# ---------------------------------------------------------------------------
# Dataset overview & Distributions
# ---------------------------------------------------------------------------

def print_dataset_overview(df: pd.DataFrame):
    """Print language breakdown and summary to console."""
    print("\n" + "=" * 70)
    print("              LANGUAGE DISTRIBUTION IN DATASET")
    print("=" * 70)
    lang_counts = df['langue'].value_counts()
    total_rows = len(df)
    for lang, count in lang_counts.items():
        print(f"  {str(lang).upper()}: {count} rows ({count / total_rows * 100:.1f}%)")
    print(f"  TOTAL: {total_rows} rows")
    print("=" * 70 + "\n")


def generate_full_report(
    df: pd.DataFrame,
    expected_model_columns: list[tuple[str, str, str]] | None = None,
    model_status: dict | None = None,
    output_file: Path = REPORT_PATH
) -> str:
    """
    Generate comprehensive evaluation report including:
      - Language distribution
      - Ground-truth sentiment distribution (MSA / Dialectal / EN / FR)
      - Ground-truth topic distribution (all 600 articles + per language)
      - Sentiment models performance (Accuracy, F1 Macro, Precision, Recall, distributions, MSA/Dialectal breakdown)
      - NER models performance (Entity rates, MSA/Dialectal breakdown)
      - Topic models performance (Accuracy, F1 Macro, Precision, Recall, distributions)
      - Missing values summary
    """
    if expected_model_columns is None:
        from .bert_runner import SENTIMENT_MODELS, NER_MODELS
        from .nli_runner import TOPIC_MODELS
        expected_model_columns = []
        for lang, models in SENTIMENT_MODELS.items():
            for m in models:
                expected_model_columns.append((f"{m.split('/')[-1]}_pred", lang, 'sentiment'))
        for lang, models in NER_MODELS.items():
            for m in models:
                expected_model_columns.append((f"{m.split('/')[-1]}_ner_pred", lang, 'ner'))
        for lang, models in TOPIC_MODELS.items():
            for m in models:
                expected_model_columns.append((f"{m.split('/')[-1]}_topic_pred", lang, 'topic'))

    total_rows = len(df)
    lang_counts = df['langue'].value_counts()

    report_lines: list[str] = []
    report_lines.append("=" * 70)
    report_lines.append("       SENTIMENT, NER & TOPIC EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Language distribution
    report_lines.append("LANGUAGE DISTRIBUTION:")
    for lang, count in lang_counts.items():
        report_lines.append(f"  {str(lang).upper()}: {count} rows ({count / total_rows * 100:.1f}%)")
    report_lines.append(f"  TOTAL: {total_rows} rows")
    report_lines.append("")

    # ===================================================================
    # SECTION 1 — GROUND-TRUTH SENTIMENT DISTRIBUTION (by variant)
    # ===================================================================
    report_lines.append("=" * 70)
    report_lines.append("    GROUND-TRUTH SENTIMENT DISTRIBUTION (by language variant)")
    report_lines.append("=" * 70)
    report_lines.append("")

    sent_true_col = find_column(df, ['sentiment attendu', 'sentiment'])
    if sent_true_col and sent_true_col in df.columns:
        ar_df  = df[df['langue'] == 'ar'].reset_index(drop=True)
        msa_df = ar_df.iloc[:MSA_COUNT]
        dia_df = ar_df.iloc[MSA_COUNT:]
        en_df  = df[df['langue'] == 'en']
        fr_df  = df[df['langue'] == 'fr']

        variant_label_map = [
            ("MSA Arabic (first 150 AR rows)",      msa_df),
            ("Dialectal Arabic (remaining AR rows)", dia_df),
            ("English",                              en_df),
            ("French",                               fr_df),
        ]

        for variant_name, sub_df in variant_label_map:
            valid = sub_df[sent_true_col].astype(str).str.strip().str.upper()
            valid = valid[valid.isin(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])]
            n_valid = len(valid)
            report_lines.append(f"  [{variant_name}] — {n_valid} labelled rows")
            if n_valid == 0:
                report_lines.append("    (no ground-truth labels found)")
            else:
                for sent_val in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                    cnt = (valid == sent_val).sum()
                    report_lines.append(f"    {sent_val}: {cnt} ({cnt / n_valid * 100:.1f}%)")
            report_lines.append("")
    else:
        report_lines.append("  No 'sentiment attendu' column found — skipping distribution.")
        report_lines.append("")

    # ===================================================================
    # SECTION 2 — GROUND-TRUTH TOPIC DISTRIBUTION (all 600 + per lang)
    # ===================================================================
    report_lines.append("=" * 70)
    report_lines.append("    GROUND-TRUTH TOPIC DISTRIBUTION (600 articles + per language)")
    report_lines.append("=" * 70)
    report_lines.append("")

    topic_true_col = find_column(df, ['thème attendu', 'theme attendu', 'topic attendu', 'topic'])
    if topic_true_col and topic_true_col in df.columns:
        topic_valid_all = df[topic_true_col].astype(str).str.strip()
        topic_valid_all = topic_valid_all[(topic_valid_all != '') & (topic_valid_all.str.lower() != 'nan')]
        n_topic_all = len(topic_valid_all)

        report_lines.append(f"  [ALL 600 ARTICLES] — {n_topic_all} labelled rows")
        if n_topic_all > 0:
            tc_all = topic_valid_all.value_counts()
            for topic_val, cnt in tc_all.items():
                report_lines.append(f"    {topic_val}: {cnt} ({cnt / n_topic_all * 100:.1f}%)")
        else:
            report_lines.append("    (no ground-truth topic labels found)")
        report_lines.append("")

        for lang_code, lang_label in [('ar', 'Arabic'), ('en', 'English'), ('fr', 'French')]:
            lang_sub = df[df['langue'] == lang_code]
            topic_valid_lang = lang_sub[topic_true_col].astype(str).str.strip()
            topic_valid_lang = topic_valid_lang[(topic_valid_lang != '') & (topic_valid_lang.str.lower() != 'nan')]
            n_lang = len(topic_valid_lang)
            report_lines.append(f"  [{lang_label}] — {n_lang} labelled rows")
            if n_lang > 0:
                tc_lang = topic_valid_lang.value_counts()
                for topic_val, cnt in tc_lang.items():
                    report_lines.append(f"    {topic_val}: {cnt} ({cnt / n_lang * 100:.1f}%)")
            else:
                report_lines.append("    (no ground-truth topic labels found)")
            report_lines.append("")
    else:
        report_lines.append("  No 'thème attendu' column found — skipping distribution.")
        report_lines.append("")

    persisted_bench = load_benchmark_records()

    # ===================================================================
    # SECTION 3 — SENTIMENT MODELS PERFORMANCE
    # ===================================================================
    report_lines.append("=" * 70)
    report_lines.append("                    SENTIMENT MODELS PERFORMANCE")
    report_lines.append("=" * 70)
    report_lines.append("")

    for col_name, lang, task_name in expected_model_columns:
        if task_name != 'sentiment' or col_name not in df.columns:
            continue

        lang_rows = df if lang == 'multi' else df[df['langue'] == lang]
        total_r = len(lang_rows)

        acc, f1, prec, rec, valid_count = calculate_sentiment_metrics(lang_rows, col_name)
        sentiment_counts = lang_rows[col_name].value_counts()

        report_lines.append(f"{col_name} ({lang.upper()}):")
        report_lines.append(f"  Total rows: {total_r}")
        bench = find_benchmark_for_model('sentiment', col_name, lang, model_status, persisted_bench)
        append_benchmark_block(report_lines, bench)
        append_metrics_block(report_lines, acc, f1, prec, rec, valid_count)

        report_lines.append("  Predicted Distribution:")
        # Build valid-only counts so percentages always sum to 100%
        valid_pred_counts = {
            k: v for k, v in sentiment_counts.items()
            if str(k) not in ['ERROR_SENTIMENT', 'nan', '']
        }
        valid_pred_total = sum(valid_pred_counts.values()) or total_r
        for sent, count in valid_pred_counts.items():
            report_lines.append(f"    {sent}: {count} ({count / valid_pred_total * 100:.1f}%)")

        if lang == 'ar':
            ar_df  = lang_rows[lang_rows['langue'] == 'ar'].reset_index(drop=True)
            msa_df = ar_df.iloc[:MSA_COUNT]
            dia_df = ar_df.iloc[MSA_COUNT:]

            report_lines.append("")
            report_lines.append("  --- MSA / Dialectal Breakdown ---")

            for sub_name, sub_df in [("MSA", msa_df), ("Dialectal", dia_df)]:
                if len(sub_df) == 0:
                    continue
                sub_preds = sub_df[col_name].astype(str).str.strip()
                sub_valid = sub_preds[
                    (sub_preds != 'nan') &
                    (sub_preds != '') &
                    (sub_preds != 'ERROR_SENTIMENT') &
                    sub_preds.notna()
                ]
                total_processed = len(sub_valid)
                if total_processed > 0:
                    acc_sub, f1_sub, prec_sub, rec_sub, n_sub = calculate_sentiment_metrics(sub_df, col_name)
                    report_lines.append(f"    [{sub_name}]:")
                    report_lines.append(f"      Rows processed : {total_processed}")
                    if acc_sub is not None:
                        report_lines.append(f"      Accuracy       : {fmt_metric(acc_sub)} ({n_sub} valid)")
                        if SKLEARN_AVAILABLE:
                            report_lines.append(f"      F1 Macro       : {fmt_metric(f1_sub)}")
                            report_lines.append(f"      Precision      : {fmt_metric(prec_sub)}")
                            report_lines.append(f"      Recall         : {fmt_metric(rec_sub)}")
                    report_lines.append("      Predicted Distribution:")
                    for s_val in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                        cnt = (sub_valid == s_val).sum()
                        report_lines.append(f"        {s_val}: {cnt} ({cnt / total_processed * 100:.1f}%)")
                else:
                    report_lines.append(f"    [{sub_name}]: N/A (no valid rows)")

        report_lines.append("")

    # ===================================================================
    # SECTION 4 — NER MODELS PERFORMANCE
    # ===================================================================
    report_lines.append("=" * 70)
    report_lines.append("                    NER MODELS PERFORMANCE")
    report_lines.append("=" * 70)
    report_lines.append("")

    for col_name, lang, task_name in expected_model_columns:
        if task_name != 'ner' or col_name not in df.columns:
            continue

        lang_rows = df if lang == 'multi' else df[df['langue'] == lang]
        total_r = len(lang_rows)

        # Processed = non-NaN rows (NaN = not yet run)
        col_series = lang_rows[col_name].astype(str).str.strip()
        processed_mask = lang_rows[col_name].notna() & (col_series != '') & (col_series.str.lower() != 'nan')
        n_processed = processed_mask.sum()
        n_unprocessed = total_r - int(n_processed)

        # Among processed rows: entities vs None vs error
        processed_vals = col_series[processed_mask]
        has_entities = (processed_vals != 'None').sum() - (processed_vals == 'ERROR_NER').sum()
        no_entities  = (processed_vals == 'None').sum()
        error_ner    = (processed_vals == 'ERROR_NER').sum()

        report_lines.append(f"{col_name} ({lang.upper()}):")
        report_lines.append(f"  Total rows    : {total_r}")
        bench = find_benchmark_for_model('ner', col_name, lang, model_status, persisted_bench)
        append_benchmark_block(report_lines, bench)
        report_lines.append(f"  Processed     : {n_processed} ({n_processed / total_r * 100:.1f}%)")
        if n_unprocessed > 0:
            report_lines.append(f"  Not yet run   : {n_unprocessed} ({n_unprocessed / total_r * 100:.1f}%)")
        if n_processed > 0:
            report_lines.append(f"  With entities : {has_entities} ({has_entities / n_processed * 100:.1f}% of processed)")
            report_lines.append(f"  No entities   : {no_entities} ({no_entities / n_processed * 100:.1f}% of processed)")
        if error_ner > 0:
            report_lines.append(f"  Errors        : {error_ner} ({error_ner / total_r * 100:.1f}%)")

        if lang == 'ar':
            ar_df  = lang_rows[lang_rows['langue'] == 'ar'].reset_index(drop=True)
            msa_df = ar_df.iloc[:MSA_COUNT]
            dia_df = ar_df.iloc[MSA_COUNT:]

            report_lines.append("")
            report_lines.append("  --- MSA / Dialectal Breakdown ---")

            for sub_name, sub_df in [("MSA", msa_df), ("Dialectal", dia_df)]:
                if len(sub_df) == 0:
                    continue
                sub_preds = sub_df[col_name].astype(str).str.strip()
                sub_valid = sub_preds[
                    sub_df[col_name].notna() &
                    (sub_preds != '') &
                    (sub_preds.str.lower() != 'nan')
                ]
                total_processed = len(sub_valid)
                if total_processed > 0:
                    with_ents = (sub_valid != 'None').sum() - (sub_valid == 'ERROR_NER').sum()
                    # Avg entities per text that actually has entities (not per all processed texts)
                    with_ent_vals = sub_valid[(sub_valid != 'None') & (sub_valid != 'ERROR_NER')]
                    avg_ents = (
                        with_ent_vals.apply(lambda x: len(str(x).split(' | '))).sum() / with_ents
                        if with_ents > 0 else 0.0
                    )
                    report_lines.append(f"    [{sub_name}]:")
                    report_lines.append(f"      Processed           : {total_processed}")
                    report_lines.append(f"      With entities       : {with_ents} ({with_ents / total_processed * 100:.1f}%)")
                    report_lines.append(f"      Avg entities/text   : {avg_ents:.2f}")
                else:
                    report_lines.append(f"    [{sub_name}]: N/A (no valid rows)")

        report_lines.append("")

    # ===================================================================
    # SECTION 5 — TOPIC MODELS PERFORMANCE
    # ===================================================================
    report_lines.append("=" * 70)
    report_lines.append("                    TOPIC MODELS PERFORMANCE")
    report_lines.append("=" * 70)
    report_lines.append("")

    for col_name, lang, task_name in expected_model_columns:
        if task_name != 'topic' or col_name not in df.columns:
            continue

        lang_rows = df if lang == 'multi' else df[df['langue'] == lang]
        total_r = len(lang_rows)

        acc, f1, prec, rec, valid_count = calculate_topic_metrics(lang_rows, col_name)
        topic_counts = lang_rows[col_name].value_counts()

        report_lines.append(f"{col_name} ({lang.upper()}):")
        report_lines.append(f"  Total rows: {total_r}")
        bench = find_benchmark_for_model('topic', col_name, lang, model_status, persisted_bench)
        append_benchmark_block(report_lines, bench)

        # Multi models use per-row language candidate labels (AR for AR rows, FR for FR, EN for EN).
        # Their accuracy is therefore directly comparable to language-specific models.
        if lang == 'multi':
            report_lines.append("  NOTE: Model uses language-matched candidate labels per row (AR/EN/FR),")
            report_lines.append("        so accuracy is directly comparable to language-specific models.")

        append_metrics_block(report_lines, acc, f1, prec, rec, valid_count)
        report_lines.append("  Predicted Distribution:")
        # Build valid-only counts so percentages always sum to 100%
        valid_topic_counts = {
            k: v for k, v in topic_counts.items()
            if str(k) not in ['ERROR_TOPIC', 'nan', '']
        }
        valid_topic_total = sum(valid_topic_counts.values()) or total_r
        for topic, count in valid_topic_counts.items():
            report_lines.append(f"    {topic}: {count} ({count / valid_topic_total * 100:.1f}%)")
        report_lines.append("")

    # ===================================================================
    # SECTION 6 — MISSING VALUES SUMMARY
    # ===================================================================
    total_missing_all = 0
    for col_name, lang, task_name in expected_model_columns:
        if col_name in df.columns:
            lang_rows = df if lang == 'multi' else df[df['langue'] == lang]
            missing    = lang_rows[col_name].isna().sum()
            empty      = (lang_rows[col_name].astype(str).str.strip() == "").sum()
            nan_str    = (lang_rows[col_name].astype(str).str.strip().str.lower() == "nan").sum()
            error_sent = (lang_rows[col_name].astype(str).str.strip() == "ERROR_SENTIMENT").sum()
            error_ner  = (lang_rows[col_name].astype(str).str.strip() == "ERROR_NER").sum()
            error_top  = (lang_rows[col_name].astype(str).str.strip() == "ERROR_TOPIC").sum()
            total_missing_all += int(missing + empty + nan_str + error_sent + error_ner + error_top)

    report_lines.append("=" * 70)
    report_lines.append("                    MISSING VALUES SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append(f"TOTAL MISSING VALUES ACROSS ALL COLUMNS: {total_missing_all}")
    report_lines.append("=" * 70)

    report_content = "\n".join(report_lines)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(report_content)

    print(f"\nReport generated and saved to: {output_file}")
    return report_content


def check_missing_values(df: pd.DataFrame, expected_model_columns: list[tuple[str, str, str]]) -> int:
    """Print missing values status per column and return total missing count."""
    print(f"\n{'=' * 70}")
    print("              FINAL MISSING VALUES CHECK (BY LANGUAGE)")
    print(f"{'=' * 70}")

    total_missing_all = 0
    for col_name, lang, task_name in expected_model_columns:
        if col_name in df.columns:
            lang_rows = df if lang == 'multi' else df[df['langue'] == lang]

            missing    = lang_rows[col_name].isna().sum()
            empty      = (lang_rows[col_name].astype(str).str.strip() == "").sum()
            nan_str    = (lang_rows[col_name].astype(str).str.strip().str.lower() == "nan").sum()
            error_sent = (lang_rows[col_name].astype(str).str.strip() == "ERROR_SENTIMENT").sum()
            error_ner  = (lang_rows[col_name].astype(str).str.strip() == "ERROR_NER").sum()
            error_top  = (lang_rows[col_name].astype(str).str.strip() == "ERROR_TOPIC").sum()
            total_miss = missing + empty + nan_str + error_sent + error_ner + error_top
            total_missing_all += int(total_miss)

            total_r = len(lang_rows)
            if total_miss > 0:
                print(f"⚠️  {col_name} ({lang.upper()}): {total_miss} missing out of {total_r} rows ({total_miss / total_r * 100:.1f}%)")
            else:
                print(f"✅ {col_name} ({lang.upper()}): COMPLETE ({total_r} rows)")

    print(f"{'=' * 70}")
    print(f"TOTAL MISSING VALUES ACROSS ALL COLUMNS: {total_missing_all}")
    print(f"{'=' * 70}")
    return total_missing_all
