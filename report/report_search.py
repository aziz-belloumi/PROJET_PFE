"""
report_search.py
----------------
Scans all fine-tuning runs under experiments/<task>/ and finds the best model
based on eval_results.json written after each run.

Ranking metric: average F1 macro across MSA and Dialect evaluation sets.
Falls back to highest single-set F1 if only one set was evaluated.

Usage (CLI):
    python report/report_search.py                      # all tasks (default)
    python report/report_search.py --task sentiment
    python report/report_search.py --task topic
    python report/report_search.py --metric dialect_f1  # rank by dialect F1 only

Usage (import):
    from report.report_search import find_best_run
    model_dir = find_best_run()          # returns Path to the best run folder
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# UTF-8 output reconfiguration for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Resolves to the parent of the report subfolder (the project root)
EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"


# ── Metric extraction ─────────────────────────────────────────────────────────

def _get_metric(eval_results: dict, metric: str) -> Optional[float]:
    """
    Extract a numeric value from the nested eval_results dict.

    eval_results.json structure:
      {
        "msa":     {"msa_f1_macro": 0.84, "msa_accuracy": 0.85, ...},
        "dialect": {"dialect_f1_macro": 0.81, "dialect_accuracy": 0.82, ...}
      }
    """
    if metric == "avg_f1":
        msa_f1  = eval_results.get("msa",     {}).get("msa_f1_macro")
        dial_f1 = eval_results.get("dialect", {}).get("dialect_f1_macro")
        available = [v for v in [msa_f1, dial_f1] if v is not None]
        return sum(available) / len(available) if available else None

    if metric == "msa_f1":
        return eval_results.get("msa", {}).get("msa_f1_macro")

    if metric == "dialect_f1":
        return eval_results.get("dialect", {}).get("dialect_f1_macro")

    if metric == "avg_accuracy":
        msa_acc  = eval_results.get("msa",     {}).get("msa_accuracy")
        dial_acc = eval_results.get("dialect", {}).get("dialect_accuracy")
        available = [v for v in [msa_acc, dial_acc] if v is not None]
        return sum(available) / len(available) if available else None

    # generic fallback: try every nested dict
    for subset in eval_results.values():
        if isinstance(subset, dict) and metric in subset:
            return subset[metric]
    return None


# ── Run scanning ──────────────────────────────────────────────────────────────

def scan_runs(task: str = "sentiment") -> List[dict]:
    """
    Walk experiments/<task>/ and collect metadata from every completed run
    (i.e. every subdirectory that contains eval_results.json).

    Returns a list of dicts, one per run, sorted by run timestamp (oldest first).
    """
    task_dir = EXPERIMENTS_ROOT / task
    if not task_dir.exists():
        return []

    runs = []
    for run_dir in sorted(task_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        eval_file   = run_dir / "eval_results.json"
        param_file  = run_dir / "hyperparameters.json"
        if not eval_file.exists():
            # run crashed or still in progress — skip
            continue
        try:
            eval_results = json.loads(eval_file.read_text(encoding="utf-8"))
            hyperparams  = json.loads(param_file.read_text(encoding="utf-8")) if param_file.exists() else {}
        except Exception:
            continue

        runs.append({
            "run_dir":      run_dir,
            "timestamp":    run_dir.name,
            "eval_results": eval_results,
            "hyperparams":  hyperparams,
        })
    return runs


def rank_runs(runs: List[dict], metric: str = "avg_f1") -> List[Tuple[float, dict]]:
    """
    Attach the chosen metric value to each run and sort descending.
    Runs without the metric are placed last.
    """
    scored = []
    for run in runs:
        value = _get_metric(run["eval_results"], metric)
        scored.append((value if value is not None else -1.0, run))
    return sorted(scored, key=lambda x: x[0], reverse=True)


# ── Public API ────────────────────────────────────────────────────────────────

def find_best_run(task: str = "sentiment", metric: str = "avg_f1") -> Path:
    """
    Return the Path to the best run directory for the given task.
    Raises FileNotFoundError if no completed runs exist.
    """
    runs = scan_runs(task)
    if not runs:
        raise FileNotFoundError(
            f"No completed runs found in {EXPERIMENTS_ROOT / task}/. "
            "Run fine_tune_arabic_sentiment.py or fine_tune_arabic_topic.py first."
        )
    ranked = rank_runs(runs, metric)
    best_score, best_run = ranked[0]
    return best_run["run_dir"]


# ── CLI report ────────────────────────────────────────────────────────────────

def print_report(task: str = "sentiment", metric: str = "avg_f1", limit: Optional[int] = 5) -> None:
    runs = scan_runs(task)
    if not runs:
        print(f"\nNo completed runs found in {EXPERIMENTS_ROOT / task}/")
        return

    ranked = rank_runs(runs, metric)
    total_runs = len(ranked)
    if limit is not None:
        ranked = ranked[:limit]

    col_w = 26
    print(f"\n{'='*70}")
    print(f"  Fine-tuning run report  |  task={task}  |  ranked by={metric}")
    if limit is not None and total_runs > limit:
        print(f"  (showing top {limit} of {total_runs} runs)")
    print(f"{'='*70}")
    header = f"{'Rank':<5}{'Timestamp':<{col_w}}{'Score':>8}  {'MSA F1':>8}  {'Dial F1':>8}  {'Epochs':>6}  {'LR':>8}"
    print(header)
    print("-" * 70)

    for rank, (score, run) in enumerate(ranked, start=1):
        ts       = run["timestamp"]
        hp       = run["hyperparams"]
        er       = run["eval_results"]
        msa_f1   = er.get("msa",     {}).get("msa_f1_macro",      "-")
        dial_f1  = er.get("dialect", {}).get("dialect_f1_macro",  "-")
        epochs   = hp.get("epochs", "-")
        lr       = hp.get("learning_rate", "-")

        def fmt(v):
            if isinstance(v, float):
                if 0 < v < 0.001:
                    return f"{v:.1e}"
                return f"{v:.4f}"
            return str(v)

        marker = " <-- BEST" if rank == 1 else ""
        print(f"{rank:<5}{ts:<{col_w}}{fmt(score):>8}  {fmt(msa_f1):>8}  {fmt(dial_f1):>8}  {str(epochs):>6}  {fmt(lr):>8}{marker}")

    best_dir = ranked[0][1]["run_dir"]
    print(f"\nBest model directory: {best_dir}\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and rank fine-tuning experiments.")
    parser.add_argument("--task",   default="all", choices=["all", "sentiment", "topic"],
                        help="Task subfolder to report (all / sentiment / topic).")
    parser.add_argument("--metric", default="avg_f1",
                        choices=["avg_f1", "msa_f1", "dialect_f1", "avg_accuracy"],
                        help="Metric to rank runs by (default: avg_f1).")
    parser.add_argument("--limit", type=int, default=5,
                        help="Limit the number of ranked runs displayed (default: 5).")
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    if args.task == "all":
        print_report(task="sentiment", metric=args.metric, limit=args.limit)
        print_report(task="topic", metric=args.metric, limit=args.limit)
    else:
        print_report(task=args.task, metric=args.metric, limit=args.limit)
