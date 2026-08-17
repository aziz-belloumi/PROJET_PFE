"""
report_search.py
================
Scan experiments/ across Arabic and English tasks, extract evaluation metrics,
rank runs by Macro F1 / Accuracy, and surface the best model checkpoints.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# UTF-8 output reconfiguration for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent / "experiments"


def _extract_metric_from_dict(d: dict, metric: str) -> Optional[float]:
    """Helper to extract a metric float recursively or through standard keys."""
    if not isinstance(d, dict):
        return None
    if metric in d and isinstance(d[metric], (int, float)):
        return float(d[metric])

    # Check known nested structures
    if metric in ("f1", "f1_macro", "eval_f1_macro", "avg_f1"):
        # 1. Direct key matches
        for k in ["f1_macro", "eval_f1_macro", "f1", "f1_weighted"]:
            if k in d and isinstance(d[k], (int, float)):
                return float(d[k])
        # 2. Check msa / dialect composite
        if "msa" in d or "dialect" in d:
            msa_f1 = d.get("msa", {}).get("msa_f1_macro")
            dial_f1 = d.get("dialect", {}).get("dialect_f1_macro")
            vals = [v for v in [msa_f1, dial_f1] if v is not None]
            if vals:
                return sum(vals) / len(vals)
        # 3. Check summary.json or metrics dict
        if "metrics" in d:
            return _extract_metric_from_dict(d["metrics"], metric)
        if "test" in d and isinstance(d["test"], dict):
            return _extract_metric_from_dict(d["test"], metric)
        if "validation" in d and isinstance(d["validation"], dict):
            return _extract_metric_from_dict(d["validation"], metric)

    if metric in ("accuracy", "eval_accuracy", "avg_accuracy"):
        for k in ["accuracy", "eval_accuracy"]:
            if k in d and isinstance(d[k], (int, float)):
                return float(d[k])
        if "msa" in d or "dialect" in d:
            msa_acc = d.get("msa", {}).get("msa_accuracy")
            dial_acc = d.get("dialect", {}).get("dialect_accuracy")
            vals = [v for v in [msa_acc, dial_acc] if v is not None]
            if vals:
                return sum(vals) / len(vals)
        if "metrics" in d:
            return _extract_metric_from_dict(d["metrics"], metric)
        if "test" in d and isinstance(d["test"], dict):
            return _extract_metric_from_dict(d["test"], metric)

    # Generic search inside sub-dictionaries
    for v in d.values():
        if isinstance(v, dict):
            res = _extract_metric_from_dict(v, metric)
            if res is not None:
                return res
    return None


def scan_runs(task: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Walk experiments/ and collect metadata from every completed run directory.
    """
    if not EXPERIMENTS_ROOT.exists():
        return []

    task_dirs = []
    if task and task.lower() != "all":
        td = EXPERIMENTS_ROOT / task
        if td.exists():
            task_dirs.append(td)
    else:
        task_dirs = [p for p in EXPERIMENTS_ROOT.iterdir() if p.is_dir()]

    runs = []
    for t_dir in task_dirs:
        task_name = t_dir.name
        for run_dir in sorted(t_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            eval_data = {}
            hyperparams = {}

            # Search potential eval files
            for eval_fname in ["eval_results.json", "all_metrics.json", "summary.json", "metrics.json"]:
                fpath = run_dir / eval_fname
                if fpath.exists():
                    try:
                        eval_data.update(json.loads(fpath.read_text(encoding="utf-8")))
                    except Exception:
                        pass

            # Search potential hyperparam files
            for param_fname in ["hyperparameters.json", "summary.json"]:
                fpath = run_dir / param_fname
                if fpath.exists():
                    try:
                        loaded = json.loads(fpath.read_text(encoding="utf-8"))
                        if "hyperparameters" in loaded and isinstance(loaded["hyperparameters"], dict):
                            hyperparams.update(loaded["hyperparameters"])
                        else:
                            hyperparams.update(loaded)
                    except Exception:
                        pass

            if not eval_data and not hyperparams:
                continue

            runs.append({
                "task": task_name,
                "run_dir": run_dir,
                "timestamp": run_dir.name,
                "eval_results": eval_data,
                "hyperparams": hyperparams,
            })
    return runs


def rank_runs(runs: List[Dict[str, Any]], metric: str = "f1_macro") -> List[Tuple[float, Dict[str, Any]]]:
    scored = []
    for run in runs:
        val = _extract_metric_from_dict(run["eval_results"], metric)
        scored.append((val if val is not None else -1.0, run))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def print_leaderboard(ranked_runs: List[Tuple[float, Dict[str, Any]]], metric: str = "f1_macro"):
    print("=" * 95)
    print(f" EXPERIMENT LEADERBOARD (Ranked by {metric.upper()})")
    print("=" * 95)
    print(f"{'Rank':<5} | {'Task':<22} | {'Run / Timestamp':<24} | {metric.upper():<10} | {'Model':<24}")
    print("-" * 95)

    for i, (score, run) in enumerate(ranked_runs, 1):
        score_str = f"{score:.4f}" if score >= 0 else "N/A"
        task_str = run["task"][:22]
        time_str = run["timestamp"][:24]
        model_str = str(run["hyperparams"].get("model_name", "Unknown"))[:24]
        print(f"{i:<5} | {task_str:<22} | {time_str:<24} | {score_str:<10} | {model_str:<24}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Scan and rank fine-tuning experiment runs.")
    parser.add_argument("--task", type=str, default="all", help="Task name (e.g. sentiment, topic, all)")
    parser.add_argument("--metric", type=str, default="f1_macro", help="Metric to rank by (f1_macro, accuracy)")
    args = parser.parse_args()

    runs = scan_runs(args.task)
    if not runs:
        print(f"No completed experiment runs found in {EXPERIMENTS_ROOT}")
        return

    ranked = rank_runs(runs, metric=args.metric)
    print_leaderboard(ranked, metric=args.metric)


if __name__ == "__main__":
    main()
