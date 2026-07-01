"""
fine_tune_arabic_sentiment.py
------------------------------
Base model : CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment
Task       : Sentiment classification — 3 classes (NEGATIVE / NEUTRAL / POSITIVE)
Data source: fine_tune_data/global_data_merged.csv
             Arabic rows (MSA / EGY / LEV / GLF / MGR) are filtered via VALID_LANG.
Eval split : 4,000 rows (2,000 MSA + 2,000 Dialect), held out before training.

Output — everything is inside experiments/sentiment/<timestamp>/:
  checkpoint-XXXX/        periodic checkpoints saved by Trainer
  model.safetensors       best model weights (saved at end of run)
  config.json / tokenizer files
  hyperparameters.json    hyperparams recorded at the START of the run
  eval_results.json       final MSA + Dialect evaluation scores
  plots/
    train_loss.png
    eval_loss.png
    eval_f1_accuracy.png
    learning_curve.png
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server / no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# =======================
# HARD-CODED SETTINGS
# =======================
TASK       = "sentiment"
DATA_FILE  = "fine_tune_data/global_data_merged.csv"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

OUT_DIR = "experiments"

EVAL_MSA_N        = 2000
EVAL_DIALECT_N    = 2000
DIALECT_TRAIN_RATIO = 0.50

MAX_LEN    = 256
EPOCHS     = 4
LR         = 2e-5
BATCH_SIZE = 2
GRAD_ACCUM = 8
EVAL_STEPS = 2000
SEED       = 42

VALID_LANG   = {"msa", "egy", "lev", "glf", "mgr"}

# =======================
# LABEL MAPS
# =======================
SENT2ID  = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2SENT  = {v: k for k, v in SENT2ID.items()}
NUM_LABELS = 3
ID2LABEL   = {i: ID2SENT[i] for i in range(NUM_LABELS)}
LABEL2ID   = {v: k for k, v in ID2LABEL.items()}


def parse_sentiment(x) -> int:
    if pd.isna(x):
        return -1
    s = str(x).strip()
    if s.isdigit():
        i = int(s)
        return i if i in (0, 1, 2) else -1
    return SENT2ID.get(s.upper(), -1)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def split_fixed_eval(df_group: pd.DataFrame, n_eval: int):
    if n_eval <= 0:
        return df_group, df_group.iloc[0:0].copy()
    if len(df_group) <= n_eval:
        return df_group.iloc[0:0].copy(), df_group.copy()
    try:
        tr, ev = train_test_split(df_group, test_size=n_eval, random_state=SEED, stratify=df_group["label"])
    except Exception:
        ev = df_group.sample(n=n_eval, random_state=SEED)
        tr = df_group.drop(ev.index)
    return tr, ev


def upsample_dialect(train_msa: pd.DataFrame, train_dial: pd.DataFrame, target_ratio: float):
    if target_ratio <= 0 or target_ratio >= 1:
        return pd.concat([train_msa, train_dial], ignore_index=True)

    n_msa, n_dial = len(train_msa), len(train_dial)
    if n_msa == 0 or n_dial == 0:
        return pd.concat([train_msa, train_dial], ignore_index=True)

    cur = n_dial / (n_msa + n_dial)
    if cur >= target_ratio:
        return pd.concat([train_msa, train_dial], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    needed_dial = int(round(n_msa * (target_ratio / (1 - target_ratio))))
    extra = max(0, needed_dial - n_dial)
    dial_extra = train_dial.sample(n=extra, replace=True, random_state=SEED) if extra > 0 else train_dial.iloc[0:0]
    out = pd.concat([train_msa, train_dial, dial_extra], ignore_index=True)
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)


# =======================
# RUN-TRACKING HELPERS
# =======================

def _extract_logs(log_history):
    """Split trainer log_history into training-step and eval-step records."""
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs  = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def save_run_artifacts(trainer, run_dir: Path, eval_msa_ds, eval_dial_ds) -> None:
    """
    Called once after trainer.train().
    Saves eval_results.json and all learning-curve plots into run_dir.
    """
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_logs, eval_logs = _extract_logs(trainer.state.log_history)

    # ── 1. Final evaluation results ──────────────────────────────────────────
    res_msa  = trainer.evaluate(eval_msa_ds,  metric_key_prefix="msa")
    res_dial = trainer.evaluate(eval_dial_ds, metric_key_prefix="dialect")
    eval_results = {"msa": res_msa, "dialect": res_dial}
    (run_dir / "eval_results.json").write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFinal Eval MSA   :", res_msa)
    print("Final Eval Dialect:", res_dial)

    def _steps(logs, key): return [e["step"] for e in logs if key in e]
    def _vals(logs, key):  return [e[key]    for e in logs if key in e]

    STYLE = dict(linewidth=1.8)

    # ── 2. Training loss ─────────────────────────────────────────────────────
    if train_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"),
                color="#e07b39", label="Train loss", **STYLE)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "train_loss.png", dpi=150)
        plt.close(fig)

    # ── 3. Eval loss ─────────────────────────────────────────────────────────
    if eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"),
                color="#4a90d9", label="Eval loss", **STYLE)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "eval_loss.png", dpi=150)
        plt.close(fig)

    # ── 4. Eval F1-macro + Accuracy (dual axis) ──────────────────────────────
    if eval_logs:
        steps = _steps(eval_logs, "eval_f1_macro")
        f1s   = _vals(eval_logs,  "eval_f1_macro")
        accs  = _vals(eval_logs,  "eval_accuracy")
        if steps and f1s:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax2 = ax1.twinx()
            ax1.plot(steps, f1s,  color="#2ecc71", label="F1 macro",  **STYLE)
            ax2.plot(steps, accs, color="#9b59b6", label="Accuracy", linestyle="--", **STYLE)
            ax1.set_xlabel("Step")
            ax1.set_ylabel("F1 Macro",  color="#2ecc71")
            ax2.set_ylabel("Accuracy",  color="#9b59b6")
            ax1.set_title("Eval F1 Macro & Accuracy")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "eval_f1_accuracy.png", dpi=150)
            plt.close(fig)

    # ── 5. Train loss vs Eval loss ───────────────────────────────────────────
    if train_logs and eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"),
                color="#e07b39", label="Train loss", **STYLE)
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"),
                color="#4a90d9", label="Eval loss",  **STYLE)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Learning Curve — Train vs Eval Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=150)
        plt.close(fig)

    print(f"\nRun artifacts saved to: {run_dir}")


def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    base = Path(__file__).resolve().parent

    # ── Create per-run directory ──────────────────────────────────────────────
    run_ts  = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = base / OUT_DIR / TASK / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Save hyperparameters immediately (recorded even on crash) ─────────────
    hyperparams = {
        "run_timestamp": run_ts,
        "model_name": MODEL_NAME,
        "task": TASK,
        "data_file": DATA_FILE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_len": MAX_LEN,
        "eval_steps": EVAL_STEPS,
        "seed": SEED,
        "eval_msa_n": EVAL_MSA_N,
        "eval_dialect_n": EVAL_DIALECT_N,
        "dialect_train_ratio": DIALECT_TRAIN_RATIO,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "fp16": torch.cuda.is_available(),
    }
    (run_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparams, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")

    # ── Load & filter data ────────────────────────────────────────────────────
    df = pd.read_csv(base / DATA_FILE)
    df["text"]     = df["text"].astype(str)
    df["language"] = df["language"].astype(str).str.lower().str.strip()
    df = df[df["language"].isin(VALID_LANG)].copy()

    df["label"] = df["sentiment"].apply(parse_sentiment).astype(int)
    df = df[df["label"] != -1].copy()

    df["variant"] = np.where(df["language"].eq("msa"), "msa", "dialect")
    msa_df  = df[df["variant"] == "msa"].copy()
    dial_df = df[df["variant"] == "dialect"].copy()

    train_msa,  eval_msa  = split_fixed_eval(msa_df,  EVAL_MSA_N)
    train_dial, eval_dial = split_fixed_eval(dial_df, EVAL_DIALECT_N)

    train_df = upsample_dialect(train_msa, train_dial, DIALECT_TRAIN_RATIO)
    eval_mix = pd.concat([eval_msa, eval_dial], ignore_index=True)

    print("Rows:", len(df))
    print("Train:", len(train_df), "| EvalMix:", len(eval_mix),
          "| EvalMSA:", len(eval_msa), "| EvalDialect:", len(eval_dial))
    print("Dialect ratio in train:", round((train_df["variant"] == "dialect").mean(), 3))

    # ── Tokenise ──────────────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds     = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_mix_ds  = Dataset.from_pandas(eval_mix[["text",  "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_msa_ds  = Dataset.from_pandas(eval_msa[["text",  "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_dial_ds = Dataset.from_pandas(eval_dial[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.gradient_checkpointing_enable()

    # ── Training arguments ────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(2, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_mix_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(run_dir))
    tok.save_pretrained(str(run_dir))

    save_run_artifacts(trainer, run_dir, eval_msa_ds, eval_dial_ds)


if __name__ == "__main__":
    main()
