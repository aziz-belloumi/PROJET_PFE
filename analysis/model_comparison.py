from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def safe_cohen_kappa(y_true: pd.Series | list, y_pred: pd.Series | list) -> float:
    """
    Computes Cohen's Kappa score safely, handling edge cases
    (e.g., constant labels, single-class predictions) without throwing exceptions.
    """
    try:
        if len(y_true) == 0:
            return 0.0
        # If all items are identical in both sets and agree
        if len(set(y_true)) == 1 and set(y_true) == set(y_pred):
            return 1.0
        score = cohen_kappa_score(y_true, y_pred)
        if pd.isna(score) or np.isnan(score):
            return 0.0
        return float(score)
    except Exception as e:
        logger.debug(f"Cohen's Kappa computation encountered an issue: {e}")
        return 0.0


def fetch_sentiment_comparison_data(engine) -> pd.DataFrame:
    """
    Fetches paired Sentiment predictions between BERT (article_sentiments)
    and Qwen 2.5-7B (qwen_article_sentiments) for overlapping articles.
    """
    query = """
        SELECT
            b.article_id,
            COALESCE(b.language, q.language, 'unknown') AS language,
            b.sentiment_label AS bert_sentiment,
            q.sentiment_label AS qwen_sentiment
        FROM article_sentiments b
        JOIN qwen_article_sentiments q
          ON b.article_id = q.article_id
        WHERE b.sentiment_label IS NOT NULL
          AND q.sentiment_label IS NOT NULL
          AND b.sentiment_label != 'SKIPPED'
          AND q.sentiment_label != 'SKIPPED'
    """
    try:
        df = pd.read_sql(query, engine)
        if not df.empty:
            df["bert_sentiment"] = df["bert_sentiment"].astype(str).str.strip().str.upper()
            df["qwen_sentiment"] = df["qwen_sentiment"].astype(str).str.strip().str.upper()
            df["language"] = df["language"].astype(str).str.strip().str.lower()
        return df
    except Exception as e:
        logger.warning(f"Could not fetch sentiment comparison data (tables may not exist or be populated): {e}")
        return pd.DataFrame(columns=["article_id", "language", "bert_sentiment", "qwen_sentiment"])


def fetch_topic_comparison_data(engine) -> pd.DataFrame:
    """
    Fetches paired Topic predictions between BERT (article_topics)
    and Qwen 2.5-7B (qwen_article_topics) for overlapping articles.
    """
    query = """
        SELECT
            b.article_id,
            COALESCE(b.language, q.language, 'unknown') AS language,
            b.topic_label AS bert_topic,
            q.topic_label AS qwen_topic
        FROM article_topics b
        JOIN qwen_article_topics q
          ON b.article_id = q.article_id
        WHERE b.topic_label IS NOT NULL
          AND q.topic_label IS NOT NULL
          AND b.topic_label != 'SKIPPED'
          AND q.topic_label != 'SKIPPED'
    """
    try:
        df = pd.read_sql(query, engine)
        if not df.empty:
            df["bert_topic"] = df["bert_topic"].astype(str).str.strip()
            df["qwen_topic"] = df["qwen_topic"].astype(str).str.strip()
            df["language"] = df["language"].astype(str).str.strip().str.lower()
        return df
    except Exception as e:
        logger.warning(f"Could not fetch topic comparison data (tables may not exist or be populated): {e}")
        return pd.DataFrame(columns=["article_id", "language", "bert_topic", "qwen_topic"])


def compute_task_agreement(
    df: pd.DataFrame,
    task_name: str,
    bert_col: str,
    qwen_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes accuracy, Cohen's Kappa, and confusion matrices for a given classification task.
    Evaluates both overall (language='ALL') and sliced per language.
    """
    if df.empty:
        empty_metrics = pd.DataFrame(columns=[
            "task", "language", "total_samples", "agreement_count", "accuracy", "cohen_kappa"
        ])
        empty_confusion = pd.DataFrame(columns=[
            "task", "language", "bert_label", "qwen_label", "count"
        ])
        return empty_metrics, empty_confusion

    metrics_records: List[Dict] = []
    confusion_records: List[Dict] = []

    # Prepare language scopes: 'ALL' + each individual language present
    languages = ["ALL"] + sorted([lang for lang in df["language"].dropna().unique() if lang and lang != "unknown"])
    if "unknown" in df["language"].values and "unknown" not in languages:
        languages.append("unknown")

    for lang in languages:
        sub_df = df if lang == "ALL" else df[df["language"] == lang]
        if sub_df.empty:
            continue

        n_samples = len(sub_df)
        agreements = int((sub_df[bert_col] == sub_df[qwen_col]).sum())
        accuracy = float(agreements / n_samples) if n_samples > 0 else 0.0
        kappa = safe_cohen_kappa(sub_df[bert_col], sub_df[qwen_col])

        metrics_records.append({
            "task": task_name,
            "language": lang,
            "total_samples": n_samples,
            "agreement_count": agreements,
            "accuracy": round(accuracy, 4),
            "cohen_kappa": round(kappa, 4),
        })

        # Confusion Matrix (cross-tabulation: BERT rows vs Qwen columns)
        ct = pd.crosstab(
            sub_df[bert_col],
            sub_df[qwen_col],
            rownames=["bert_label"],
            colnames=["qwen_label"],
        )
        ct_reset = ct.reset_index()
        melted = ct_reset.melt(id_vars=["bert_label"], var_name="qwen_label", value_name="count")
        for _, row in melted.iterrows():
            cnt = int(row["count"])
            if cnt > 0:
                confusion_records.append({
                    "task": task_name,
                    "language": lang,
                    "bert_label": str(row["bert_label"]),
                    "qwen_label": str(row["qwen_label"]),
                    "count": cnt,
                })

    metrics_df = pd.DataFrame(metrics_records)
    confusion_df = pd.DataFrame(confusion_records)
    return metrics_df, confusion_df


def compute_model_comparison(engine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes automated model comparison across all supported tasks (Sentiment & Topic).

    Returns:
      (metrics_df, confusion_df)
      - metrics_df: task, language, total_samples, agreement_count, accuracy, cohen_kappa
      - confusion_df: task, language, bert_label, qwen_label, count
    """
    logger.info("Running Model Comparison: Fine-tuned BERT vs Qwen 2.5-7B benchmark...")

    # 1. Sentiment Comparison
    sent_df = fetch_sentiment_comparison_data(engine)
    sent_metrics, sent_confusion = compute_task_agreement(
        sent_df,
        task_name="sentiment",
        bert_col="bert_sentiment",
        qwen_col="qwen_sentiment",
    )

    # 2. Topic Comparison
    topic_df = fetch_topic_comparison_data(engine)
    topic_metrics, topic_confusion = compute_task_agreement(
        topic_df,
        task_name="topic",
        bert_col="bert_topic",
        qwen_col="qwen_topic",
    )

    all_metrics = pd.concat([sent_metrics, topic_metrics], ignore_index=True) if not (sent_metrics.empty and topic_metrics.empty) else pd.DataFrame()
    all_confusion = pd.concat([sent_confusion, topic_confusion], ignore_index=True) if not (sent_confusion.empty and topic_confusion.empty) else pd.DataFrame()

    if not all_metrics.empty:
        logger.info(f"Model Comparison completed across {len(all_metrics)} evaluation slices.")
    else:
        logger.info("No overlapping predictions found between BERT and Qwen; comparison skipped.")

    return all_metrics, all_confusion


def save_model_comparison_reports(
    metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    run_dir: Union[str, Path] = "analysis/reports",
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Persists model comparison outputs as dedicated CSV files.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path: Optional[Path] = None
    confusion_path: Optional[Path] = None

    if metrics_df is not None and not metrics_df.empty:
        metrics_path = run_dir / "model_comparison_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        logger.info(f"Exported model comparison metrics: {metrics_path} ({len(metrics_df)} rows)")

    if confusion_df is not None and not confusion_df.empty:
        confusion_path = run_dir / "model_comparison_confusion.csv"
        confusion_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")
        logger.info(f"Exported model comparison confusion matrix: {confusion_path} ({len(confusion_df)} rows)")

    return metrics_path, confusion_path
