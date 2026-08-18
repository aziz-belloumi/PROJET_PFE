"""
fine_tune_arabic_topic.py
-------------------------
Base model : CAMeL-Lab/bert-base-arabic-camelbert-mix
Task       : Topic classification — 18 categories
Data source: data/global_data_libelised.csv
Output     : experiments/topic/<timestamp>/
"""
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    TrainerCallback,
)

# =======================
# PATH CONFIGURATION
# =======================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE    = PROJECT_ROOT / "data" / "global_data_libelised.csv"
OUT_DIR      = PROJECT_ROOT / "experiments"

# =======================
# SETTINGS
# =======================
TASK = "arabic_topic"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

MAX_LEN = 512
EPOCHS = 15
LR = 1e-5
BATCH_SIZE = 8
GRAD_ACCUM = 1
EVAL_STEPS = 30000
SEED = 42
DATA_SUBSET_RATIO = 1.0
RESUME_CHECKPOINT = None

VALID_LANG = {"ar", "da"}

CATEGORY_DISPLAY = {
    0: {"ar": "السياسة", "en": "Politics"},
    1: {"ar": "الاقتصاد", "en": "Economy"},
    2: {"ar": "الأمن", "en": "Security"},
    3: {"ar": "الطاقة", "en": "Energy"},
    4: {"ar": "النزاع", "en": "Conflict"},
    5: {"ar": "الانتخابات", "en": "Elections"},
    6: {"ar": "العدالة", "en": "Justice"},
    7: {"ar": "الصحة", "en": "Health"},
    8: {"ar": "الطقس", "en": "Weather"},
    9: {"ar": "الرياضة", "en": "Sports"},
    10: {"ar": "الثقافة", "en": "Culture"},
    11: {"ar": "التعليم", "en": "Education"},
    12: {"ar": "التكنولوجيا", "en": "Technology"},
    13: {"ar": "البيئة", "en": "Environment"},
    14: {"ar": "الدبلوماسية", "en": "Diplomacy"},
    15: {"ar": "الدين", "en": "Religion"},
    16: {"ar": "الهجرة", "en": "Migration"},
    17: {"ar": "عام", "en": "General"},
}

NUM_LABELS = 18
ID2LABEL = {i: CATEGORY_DISPLAY[i]["en"] for i in range(NUM_LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def norm(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    return x.strip().lower()


TOPIC_MAP = {}
for cid, langs in CATEGORY_DISPLAY.items():
    TOPIC_MAP[norm(langs["ar"])] = cid
    TOPIC_MAP[norm(langs["en"])] = cid
TOPIC_MAP[norm("العدل")] = 6


def parse_topic(x) -> int:
    if pd.isna(x):
        return -1
    s = str(x).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i <= 17 else -1
    return TOPIC_MAP.get(norm(s), -1)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1,
    }


def split_percentage(df_group: pd.DataFrame, test_ratio: float):
    if test_ratio <= 0 or test_ratio >= 1:
        return df_group, df_group.iloc[0:0].copy()
    if len(df_group) <= 1:
        return df_group.iloc[0:0].copy(), df_group.copy()
    try:
        tr, ev = train_test_split(df_group, test_size=test_ratio, random_state=SEED, stratify=df_group["label"])
    except Exception:
        ev = df_group.sample(frac=test_ratio, random_state=SEED)
        tr = df_group.drop(ev.index)
    return tr, ev


def build_class_weights(train_df: pd.DataFrame, num_labels: int, device) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.float64)
    for lbl, cnt in train_df["label"].value_counts().items():
        counts[int(lbl)] = cnt
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_labels
    median_w = np.median(weights)
    weights = np.minimum(weights, median_w * 10)
    print("\nClass weights:")
    for i in range(num_labels):
        print(f"  {i:2d} {ID2LABEL[i]:15s}: count={int(counts[i]):6d}  weight={weights[i]:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def get_layerwise_lr(model, base_lr=1e-5, decay=0.95):
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_params = []
    num_layers = 12
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = None
        for i in range(num_layers):
            if f'bert.encoder.layer.{i}.' in name:
                layer_idx = i
                break
        if layer_idx is not None:
            lr = base_lr * (decay ** (num_layers - 1 - layer_idx))
        else:
            lr = base_lr
        weight_decay = 0.0 if any(nd in name for nd in no_decay) else 0.01
        grouped_params.append({
            'params': [param],
            'lr': lr,
            'weight_decay': weight_decay
        })
    return grouped_params


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.001, patience=None, threshold=None):
        self.patience = patience if patience is not None else early_stopping_patience
        self.threshold = threshold if threshold is not None else early_stopping_threshold
        self.best_metric = None
        self.best_step = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current_metric = metrics.get("eval_f1_macro")
        if current_metric is None:
            return

        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_step = state.global_step
            return

        if current_metric > self.best_metric + self.threshold:
            self.best_metric = current_metric
            self.best_step = state.global_step
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = state.epoch
            control.should_training_stop = True
            print(f"\nEarly stopping triggered at step {state.global_step}")
            print(f"Best metric: {self.best_metric:.4f} at step {self.best_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.stopped_epoch > 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch:.2f}")


class ConfusionMatrixCallback(TrainerCallback):
    def __init__(self, eval_dataset, id2label, run_dir):
        self.eval_dataset = eval_dataset
        self.id2label = id2label
        self.run_dir = Path(run_dir)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        
        print(f"\nGenerating confusion matrix for epoch {state.epoch:.1f}...")
        predictions = trainer.predict(self.eval_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        
        cm = confusion_matrix(labels, preds, labels=list(range(len(self.id2label))))
        
        cm_dir = self.run_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        
        epoch_str = f"epoch_{int(round(state.epoch))}"
        df_cm = pd.DataFrame(cm, index=[self.id2label[i] for i in range(len(self.id2label))],
                             columns=[self.id2label[i] for i in range(len(self.id2label))])
        df_cm.to_csv(cm_dir / f"confusion_matrix_{epoch_str}.csv")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        num_classes = len(self.id2label)
        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([self.id2label[i] for i in range(num_classes)], rotation=45, ha='right')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([self.id2label[i] for i in range(num_classes)])
        
        ax.set_title(f"Confusion Matrix — Topic (Epoch {state.epoch:.1f})")
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        fig.savefig(cm_dir / f"confusion_matrix_{epoch_str}.png", dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_dir / f'confusion_matrix_{epoch_str}.png'}")


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if metric_key_prefix == "eval" and "eval_f1_macro" in metrics:
            self.state.best_metric = metrics["eval_f1_macro"]
            if self.state.best_model_checkpoint is None:
                self.state.best_model_checkpoint = self.args.output_dir
        return metrics

    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = get_layerwise_lr(self.model, base_lr=self.args.learning_rate)
            use_fused = "fused" in getattr(self.args, "optim", "") and torch.cuda.is_available()
            kwargs = {}
            if use_fused:
                try:
                    kwargs["fused"] = True
                except Exception:
                    pass
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                eps=self.args.adam_epsilon,
                **kwargs
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = FocalLoss(alpha=self.class_weights, gamma=1.5)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def _extract_logs(log_history):
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def save_run_artifacts(trainer, run_dir: Path, eval_msa_ds, eval_dial_ds, test_ds=None, test_name="test") -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_logs, eval_logs = _extract_logs(trainer.state.log_history)

    res_msa = trainer.evaluate(eval_msa_ds, metric_key_prefix="msa") if len(eval_msa_ds) > 0 else {}
    res_dial = trainer.evaluate(eval_dial_ds, metric_key_prefix="dialect") if len(eval_dial_ds) > 0 else {}
    eval_results = {"msa": res_msa, "dialect": res_dial}
    
    if test_ds is not None and len(test_ds) > 0:
        res_test = trainer.evaluate(test_ds, metric_key_prefix=test_name)
        eval_results[test_name] = res_test
        print(f"\nFinal {test_name.upper()} Results:")
        print(f"  Accuracy: {res_test.get(f'{test_name}_accuracy', 0):.4f}")
        print(f"  F1-macro: {res_test.get(f'{test_name}_f1_macro', 0):.4f}")
    
    (run_dir / "eval_results.json").write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFinal Eval Primary:", res_msa)
    if res_dial:
        print("Final Eval Secondary:", res_dial)

    if len(eval_msa_ds) > 0:
        predictions = trainer.predict(eval_msa_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        report = classification_report(labels, preds, target_names=[ID2LABEL[i] for i in range(18)], output_dict=True)
        (run_dir / "classification_report_primary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if test_ds is not None and len(test_ds) > 0:
        predictions = trainer.predict(test_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids
        report = classification_report(labels, preds, target_names=[ID2LABEL[i] for i in range(18)], output_dict=True)
        (run_dir / f"classification_report_{test_name}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        
        cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS)))
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(NUM_LABELS))
        ax.set_yticks(np.arange(NUM_LABELS))
        ax.set_xticklabels([ID2LABEL[i] for i in range(NUM_LABELS)], rotation=45, ha='right')
        ax.set_yticklabels([ID2LABEL[i] for i in range(NUM_LABELS)])
        thresh = cm.max() / 2.
        for i in range(NUM_LABELS):
            for j in range(NUM_LABELS):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_title(f'Confusion Matrix — {test_name.upper()} Set')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(run_dir / f"confusion_matrix_{test_name}.png", dpi=150)
        plt.close()

        # Per-class F1
        classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[c]['f1-score'] for c in classes]
        precisions = [report[c]['precision'] for c in classes]
        recalls = [report[c]['recall'] for c in classes]
        x = np.arange(len(classes))
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width, precisions, width, label='Precision')
        ax.bar(x, f1_scores, width, label='F1 Score')
        ax.bar(x + width, recalls, width, label='Recall')
        ax.set_xlabel('Topics')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Performance — {test_name.upper()} Set')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / f"per_class_f1_{test_name}.png", dpi=150)
        plt.close()

    def _steps(logs, key): return [e["step"] for e in logs if key in e]
    def _vals(logs, key): return [e[key] for e in logs if key in e]

    STYLE = dict(linewidth=1.8)

    if train_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"),
                color="#e07b39", label="Train loss", **STYLE)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss — Topic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "train_loss.png", dpi=150)
        plt.close(fig)

    if eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"),
                color="#4a90d9", label="Eval loss", **STYLE)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss — Topic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "eval_loss.png", dpi=150)
        plt.close(fig)

    if eval_logs:
        steps = _steps(eval_logs, "eval_f1_macro")
        f1s = _vals(eval_logs, "eval_f1_macro")
        accs = _vals(eval_logs, "eval_accuracy")
        if steps and f1s:
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax2 = ax1.twinx()
            ax1.plot(steps, f1s, color="#2ecc71", label="F1 macro", **STYLE)
            ax2.plot(steps, accs, color="#9b59b6", label="Accuracy", linestyle="--", **STYLE)
            ax1.set_xlabel("Step")
            ax1.set_ylabel("F1 Macro", color="#2ecc71")
            ax2.set_ylabel("Accuracy", color="#9b59b6")
            ax1.set_title("Eval F1 Macro & Accuracy — Topic")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "eval_f1_accuracy.png", dpi=150)
            plt.close(fig)

    if train_logs and eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"),
                color="#e07b39", label="Train loss", **STYLE)
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"),
                color="#4a90d9", label="Eval loss", **STYLE)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Curve — Train vs Eval Loss (Topic)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=150)
        plt.close(fig)

    print(f"\nRun artifacts saved to: {run_dir}")


def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = OUT_DIR / TASK / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    use_compile = False
    if use_cuda:
        try:
            test_mod = torch.nn.Linear(10, 10).cuda()
            compiled_test = torch.compile(test_mod)
            x = torch.randn(1, 10).cuda()
            compiled_test(x)
            use_compile = True
        except Exception as e:
            print(f"torch.compile test skipped: {e}")

    print(f"Device: {'CUDA' if use_cuda else 'CPU'} | bf16={use_bf16} | fp16={use_fp16} | torch_compile={use_compile}")

    hyperparams = {
        "run_timestamp": run_ts,
        "model_name": MODEL_NAME,
        "task": TASK,
        "data_file": str(DATA_FILE),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_len": MAX_LEN,
        "eval_steps": EVAL_STEPS,
        "seed": SEED,
        "data_subset_ratio": DATA_SUBSET_RATIO,
        "resume_checkpoint": RESUME_CHECKPOINT,
        "warmup_ratio": 0.15,
        "weight_decay": 0.01,
        "class_weighted_loss": True,
        "focal_loss": True,
        "focal_gamma": 1.5,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "torch_compile": use_compile,
        "num_labels": NUM_LABELS,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
    }
    (run_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparams, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df["text"] = df["text"].astype(str)
    df["language"] = df["language"].astype(str).str.lower().str.strip()
    df = df[df["language"].isin(VALID_LANG)].copy()

    df["label"] = df["topic"].apply(parse_topic).astype(int)
    df = df[df["label"] != -1].copy()

    df["variant"] = np.where(df["language"] == "da", "dialect", "msa")
    msa_df = df[df["variant"] == "msa"].copy()
    dial_df = df[df["variant"] == "dialect"].copy()

    train_msa, temp_msa = train_test_split(msa_df, test_size=0.2, random_state=SEED, stratify=msa_df["label"])
    val_msa, test_msa = train_test_split(temp_msa, test_size=0.5, random_state=SEED, stratify=temp_msa["label"])

    if len(dial_df) > 10:
        train_dial, temp_dial = train_test_split(dial_df, test_size=0.2, random_state=SEED, stratify=dial_df["label"])
        val_dial, test_dial = train_test_split(temp_dial, test_size=0.5, random_state=SEED, stratify=temp_dial["label"])
    else:
        train_dial, val_dial, test_dial = dial_df.iloc[0:0], dial_df.iloc[0:0], dial_df.iloc[0:0]

    train_df = pd.concat([train_msa, train_dial], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_mix = pd.concat([val_msa, val_dial], ignore_index=True)
    test_mix = pd.concat([test_msa, test_dial], ignore_index=True)

    if DATA_SUBSET_RATIO < 1.0:
        train_df = train_df.sample(frac=DATA_SUBSET_RATIO, random_state=SEED).reset_index(drop=True)

    eval_mix = val_mix

    print("Rows:", len(df))
    print("Train:", len(train_df), "| Val:", len(val_mix), "| Test:", len(test_mix))

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_mix_ds = Dataset.from_pandas(eval_mix[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_msa_ds = Dataset.from_pandas(val_msa[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])
    eval_dial_ds = Dataset.from_pandas(val_dial[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"]) if len(val_dial) > 0 else Dataset.from_dict({"input_ids": [], "label": []})
    test_ds = Dataset.from_pandas(test_mix[["text", "label"]], preserve_index=False).map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if use_cuda else "cpu")
    class_weights = build_class_weights(train_df, NUM_LABELS, device)

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.15,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        torch_compile=use_compile,
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        logging_steps=100,
        adam_epsilon=1e-8,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_mix_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.001
            ),
            ConfusionMatrixCallback(eval_mix_ds, ID2LABEL, run_dir)
        ],
    )

    if RESUME_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
    else:
        trainer.train()

    trainer.save_model(str(run_dir))
    tok.save_pretrained(str(run_dir))

    save_run_artifacts(trainer, run_dir, eval_msa_ds, eval_dial_ds, test_ds, test_name="test")


if __name__ == "__main__":
    main()
