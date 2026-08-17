from __future__ import annotations

import sys
import io
import argparse
from pathlib import Path
import pandas as pd

# Force stdout to UTF-8 for Arabic rendering in Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformers.shared import CSV_PATH, REPORT_PATH
from models.transformers.bert_runner import run_bert_models, SENTIMENT_MODELS, NER_MODELS
from models.transformers.nli_runner import run_nli_models, TOPIC_MODELS
from models.transformers.evaluation import (
    print_dataset_overview,
    generate_full_report,
    check_missing_values,
)
FORCE_RERUN = True


def build_expected_columns() -> list[tuple[str, str, str]]:
    """Build list of (col_name, lang, task) for all registered models."""
    expected: list[tuple[str, str, str]] = []
    for lang, models in SENTIMENT_MODELS.items():
        for m in models:
            expected.append((f"{m.split('/')[-1]}_pred", lang, 'sentiment'))
    for lang, models in NER_MODELS.items():
        for m in models:
            expected.append((f"{m.split('/')[-1]}_ner_pred", lang, 'ner'))
    for lang, models in TOPIC_MODELS.items():
        for m in models:
            expected.append((f"{m.split('/')[-1]}_topic_pred", lang, 'topic'))
    return expected


def print_health_summary(combined_status: dict):
    """Print consolidated model health check table."""
    if not combined_status:
        return
    print(f"\n{'=' * 70}")
    print("                    MODEL HEALTH CHECK SUMMARY")
    print(f"{'=' * 70}")
    status_icons = {'SUCCESS': '✅', 'PARTIAL': '⚠️ ', 'FAILED': '❌', 'SKIPPED': '⏭️ '}
    counts = {'SUCCESS': 0, 'PARTIAL': 0, 'FAILED': 0, 'SKIPPED': 0}
    for (task, lang, model), info in combined_status.items():
        icon = status_icons.get(info['status'], '?')
        short_name = model.split('/')[-1]
        status_str = f"{icon} [{info['status']:8s}] [{task}/{lang}] {short_name}"
        if info['rows'] > 0:
            status_str += f" ({info['rows']} rows"
            if info.get('benchmark'):
                b = info['benchmark']
                status_str += f" | {b.get('total_inf_time_sec', 0):.2f}s | {b.get('avg_ms_per_doc', 0):.1f} ms/doc"
            status_str += ")"
        if info['error']:
            status_str += f" | {info['error']}"
        print(status_str)
        counts[info['status']] = counts.get(info['status'], 0) + 1
    print(f"{'=' * 70}")
    print(f"  Total: {len(combined_status)} models | "
          f"✅ {counts['SUCCESS']} succeeded | "
          f"⚠️  {counts['PARTIAL']} partial | "
          f"❌ {counts['FAILED']} failed | "
          f"⏭️  {counts['SKIPPED']} skipped")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Master benchmark runner for Testing_Models")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to manual_eval_global.csv")
    parser.add_argument("--report-out", type=str, default=str(REPORT_PATH), help="Path to output report.txt")
    parser.add_argument("--bert-only", action="store_true", help="Run only BERT models (Sentiment & NER)")
    parser.add_argument("--nli-only", action="store_true", help="Run only NLI Topic classification models")
    parser.add_argument("--report-only", action="store_true", help="Skip inference and only generate report from CSV")
    parser.add_argument("--force-rerun", action=argparse.BooleanOptionalAction, default=FORCE_RERUN, help="Force rerun all models even if predictions exist")
    args = parser.parse_args()

    csv_file = Path(args.csv)
    report_file = Path(args.report_out)

    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print("            TESTING_MODELS — COMPREHENSIVE BENCHMARK")
    print(f"{'#' * 70}")

    df = pd.read_csv(csv_file, keep_default_na=False)
    print_dataset_overview(df)

    combined_status: dict = {}

    if not args.report_only:
        # Phase 1: BERT Sentiment + NER
        if not args.nli_only:
            print("\n>>> Phase 1/2: Running BERT Sentiment & NER Models...")
            df, bert_status = run_bert_models(
                df=df,
                force_rerun_sentiment=args.force_rerun,
                force_rerun_ner=args.force_rerun,
                save_csv=True,
                csv_path=csv_file
            )
            combined_status.update(bert_status)

        # Phase 2: NLI Topic Classification
        if not args.bert_only:
            print("\n>>> Phase 2/2: Running Zero-Shot NLI Topic Models...")
            df, nli_status = run_nli_models(
                df=df,
                force_rerun_topic=args.force_rerun,
                save_csv=True,
                csv_path=csv_file
            )
            combined_status.update(nli_status)

        # Model health summary
        print_health_summary(combined_status)

    # Missing values check
    expected_cols = build_expected_columns()
    check_missing_values(df, expected_cols)

    # Generate full report
    print("\n>>> Generating Final Evaluation Report...")
    generate_full_report(
        df=df,
        expected_model_columns=expected_cols,
        model_status=combined_status,
        output_file=report_file
    )

    print(f"\n{'=' * 70}")
    print(f"🎉 Benchmark complete! Full report available at: {report_file}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
