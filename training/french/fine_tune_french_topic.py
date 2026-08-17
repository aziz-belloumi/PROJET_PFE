"""
french_topic_camembert_large_final.py
-------------------------------------
French Topic Classification - CamemBERT-large
- Standard 512 truncation (single pass)
- Gradient Checkpointing DISABLED
- FOCAL LOSS + CLASS-SPECIFIC GAMMA (2.0–4.0)
- OVERSAMPLED Environment (class 13)
- 100% FIXED EARLY STOPPING
- Lower final LR for fine-tuning
"""

import json
import unicodedata
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ACCELERATION SETTINGS
# ============================================================
TASK       = "topic_french"
DATA_FILE  = "fine_tune_data/global_data_merged.csv"
MODEL_NAME = "camembert/camembert-large"
OUT_DIR = "experiments"

# ============================================================
# TRANSLATION SETUP
# ============================================================
TRANSLATED_DATA_DIR = Path("translated_data")
TRANSLATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATED_CSV_FILE = TRANSLATED_DATA_DIR / "french_topic_translated_data.csv"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-fr"
USE_GPU_FOR_TRANSLATION = True
MAX_TRANSLATED_SAMPLES = None

# ============================================================
# FINAL TRAINING PARAMETERS
# ============================================================
EVAL_PCT = 0.10
EPOCHS = 6                   
LR = 5e-6                     # <-- Halved for final fine-tune (cooling down)
GRAD_ACCUM = 4                
EVAL_STEPS = 3000             
SEED = 42
RESUME_FROM_LAST_CHECKPOINT = True

VALID_LANG = {"fr", "french", "fra"}

MAX_LEN = 512                 
BATCH_SIZE = 4                
DATALOADER_WORKERS = 0

# ============================================================
# CATEGORIES
# ============================================================
CATEGORY_DISPLAY = {
    0:  {"fr": "Politique",    "en": "Politics"},
    1:  {"fr": "Économie",     "en": "Economy"},
    2:  {"fr": "Sécurité",     "en": "Security"},
    3:  {"fr": "Énergie",      "en": "Energy"},
    4:  {"fr": "Conflit",      "en": "Conflict"},
    5:  {"fr": "Élections",    "en": "Elections"},
    6:  {"fr": "Justice",      "en": "Justice"},
    7:  {"fr": "Santé",        "en": "Health"},
    8:  {"fr": "Météo",        "en": "Weather"},
    9:  {"fr": "Sports",       "en": "Sports"},
    10: {"fr": "Culture",      "en": "Culture"},
    11: {"fr": "Éducation",    "en": "Education"},
    12: {"fr": "Technologie",  "en": "Technology"},
    13: {"fr": "Environnement","en": "Environment"},
    14: {"fr": "Diplomatie",   "en": "Diplomacy"},
    15: {"fr": "Religion",     "en": "Religion"},
    16: {"fr": "Migration",    "en": "Migration"},
    17: {"fr": "Général",      "en": "General"},
}

NUM_LABELS = 18
ID2LABEL = {i: CATEGORY_DISPLAY[i]["en"] for i in range(NUM_LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def normalize_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    return x.strip().lower()

TOPIC_MAP: dict = {}
for cid, langs in CATEGORY_DISPLAY.items():
    TOPIC_MAP[normalize_text(langs["fr"])] = cid
    TOPIC_MAP[normalize_text(langs["en"])] = cid
TOPIC_MAP[normalize_text("politique")] = 0
TOPIC_MAP[normalize_text("économie")] = 1
TOPIC_MAP[normalize_text("sécurité")] = 2
TOPIC_MAP[normalize_text("énergie")] = 3
TOPIC_MAP[normalize_text("conflit")] = 4
TOPIC_MAP[normalize_text("élections")] = 5
TOPIC_MAP[normalize_text("justice")] = 6
TOPIC_MAP[normalize_text("santé")] = 7
TOPIC_MAP[normalize_text("météo")] = 8
TOPIC_MAP[normalize_text("sports")] = 9
TOPIC_MAP[normalize_text("culture")] = 10
TOPIC_MAP[normalize_text("éducation")] = 11
TOPIC_MAP[normalize_text("technologie")] = 12
TOPIC_MAP[normalize_text("environnement")] = 13
TOPIC_MAP[normalize_text("diplomatie")] = 14
TOPIC_MAP[normalize_text("religion")] = 15
TOPIC_MAP[normalize_text("migration")] = 16
TOPIC_MAP[normalize_text("général")] = 17

def parse_topic(x) -> int:
    if pd.isna(x):
        return -1
    s = str(x).strip()
    if s.isdigit():
        i = int(s)
        return i if 0 <= i <= 17 else -1
    return TOPIC_MAP.get(normalize_text(s), -1)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }

def balance_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty:
        return train_df
    balanced_frames = []
    for label, group in train_df.groupby("label", sort=False):
        if len(group) == 0:
            continue
        balanced_frames.append(group.sample(frac=1, random_state=SEED))
    if not balanced_frames:
        return train_df
    out = pd.concat(balanced_frames, ignore_index=True)
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)

def build_class_weights(train_df: pd.DataFrame, num_labels: int, device) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.float64)
    for lbl, cnt in train_df["label"].value_counts().items():
        counts[int(lbl)] = cnt
    counts = np.maximum(counts, 1)
    # Soft inverse sqrt weighting
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * num_labels
    weights = np.clip(weights, 0.8, 2.0)
    print("\nClass weights (soft inverse sqrt):")
    for i in range(num_labels):
        if counts[i] > 0:
            print(f"  {i:2d} {ID2LABEL[i]:15s}: count={int(counts[i]):4d}  weight={weights[i]:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

# ============================================================
# DATASET - SINGLE PASS 512
# ============================================================
class SimpleTextDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        print("Tokenizing dataset (single 512 pass)...")
        self.encodings = []
        for text in tqdm(self.texts, desc="Tokenizing"):
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)
            })

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def simple_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ============================================================
# FOCAL LOSS WITH CLASS-SPECIFIC GAMMA
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', class_gammas=None):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.gamma = gamma
        self.class_gammas = class_gammas

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        
        if self.class_gammas is not None:
            gamma_per_sample = self.class_gammas[target]
            focal_loss = ((1 - pt) ** gamma_per_sample) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================
# MODEL - LARGE, FOCAL LOSS, NO GC, CLASS GAMMAS
# ============================================================
class LargeClassifierFocal(nn.Module):
    _keys_to_ignore_on_save = None
    _keys_to_ignore_on_load_missing = None
    _keys_to_ignore_on_load_unexpected = ["class_weights", "class_gammas"]

    def __init__(self, base_model_name, num_labels, id2label, label2id, class_weights=None, dropout=0.3, focal_gamma=2.0, class_gammas=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.focal_gamma = focal_gamma

        self.config = self.bert.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        if class_gammas is not None:
            self.register_buffer("class_gammas", class_gammas)
        else:
            self.class_gammas = None

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = FocalLoss(weight=self.class_weights, gamma=self.focal_gamma, class_gammas=self.class_gammas)
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() for k, v in sd.items()}

def save_model(model, tokenizer, id2label, label2id, base_model_name, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(str(save_dir))
    (save_dir / "model_config.json").write_text(
        json.dumps({
            "base_model_name": base_model_name,
            "num_labels": len(id2label),
            "id2label": id2label,
            "label2id": label2id,
            "architecture": "LargeClassifierFocal",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def _extract_logs(log_history):
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs

def plot_roc_curves(trainer, eval_dataset, plots_dir, split_name="test", num_classes=NUM_LABELS):
    print(f"\n📊 Generating ROC curves for {split_name} set...")
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    y_true_bin = label_binarize(labels, classes=range(num_classes))
    fpr = {}
    tpr = {}
    roc_auc = {}
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        if i in np.unique(labels):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            color = colors[i % len(colors)]
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{ID2LABEL[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {split_name.upper()} Set\nMulti-class Topic Classification', fontsize=14)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f'roc_curves_{split_name}.png', dpi=200)
    plt.close()
    if len(roc_auc) > 0:
        macro_auc = np.mean(list(roc_auc.values()))
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        print(f"  Macro-average AUC: {macro_auc:.4f}")
        print(f"  Micro-average AUC: {roc_auc_micro:.4f}")
        auc_scores = {
            "macro_auc": float(macro_auc),
            "micro_auc": float(roc_auc_micro),
            "per_class_auc": {ID2LABEL[i]: float(roc_auc[i]) for i in roc_auc.keys()}
        }
        with open(plots_dir / f'auc_scores_{split_name}.json', 'w') as f:
            json.dump(auc_scores, f, indent=2)
    return roc_auc

def plot_auc_barchart(roc_auc, plots_dir, split_name):
    class_names = [ID2LABEL[i] for i in sorted(roc_auc.keys())]
    auc_values = [roc_auc[i] for i in sorted(roc_auc.keys())]
    plt.figure(figsize=(14, 6))
    bars = plt.bar(class_names, auc_values, color='skyblue', alpha=0.8)
    for bar, val in zip(bars, auc_values):
        if val >= 0.8:
            bar.set_color('green')
        elif val >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    plt.axhline(y=0.5, color='black', linestyle='--', label='Random (AUC=0.5)')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title(f'AUC Scores per Class - {split_name.upper()} Set', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True, alpha=0.3)
    for bar, val in zip(bars, auc_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / f'auc_barchart_{split_name}.png', dpi=150)
    plt.close()

def plot_confusion_matrix(trainer, eval_dataset, plots_dir, split_name):
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    unique_labels = sorted(np.unique(np.concatenate([labels, preds])))
    class_names = [ID2LABEL[i] for i in unique_labels]
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title(f'Confusion Matrix - {split_name.upper()}\n(Counts)', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title(f'Confusion Matrix - {split_name.upper()}\n(Percentages)', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print(f"\n{split_name.upper()} Set - Per-class Accuracies:")
    for i, acc in enumerate(class_acc):
        print(f"  {ID2LABEL[i]}: {acc:.2%}")
    plt.tight_layout()
    plt.savefig(plots_dir / f'confusion_matrix_{split_name}.png', dpi=150)
    plt.close()
    return class_acc

def plot_training_curves(train_logs, eval_logs, plots_dir):
    STYLE = dict(linewidth=1.8)
    def _steps(logs, key): return [e["step"] for e in logs if key in e]
    def _vals(logs, key):  return [e[key] for e in logs if key in e]
    fig, ax = plt.subplots(figsize=(10, 5))
    if train_logs:
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"),
                color="#e07b39", label="Train loss", **STYLE)
    if eval_logs:
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"),
                color="#4a90d9", label="Validation loss", **STYLE)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    if eval_logs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        steps = _steps(eval_logs, "eval_accuracy")
        accs = _vals(eval_logs, "eval_accuracy")
        f1s = _vals(eval_logs, "eval_f1_macro")
        if steps:
            ax1.plot(steps, accs, color="#2ecc71", label="Accuracy", **STYLE)
            ax1.set_xlabel("Step", fontsize=12)
            ax1.set_ylabel("Accuracy", fontsize=12)
            ax1.set_title("Validation Accuracy", fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
            ax2.plot(steps, f1s, color="#9b59b6", label="F1 Macro", **STYLE)
            ax2.set_xlabel("Step", fontsize=12)
            ax2.set_ylabel("F1 Score", fontsize=12)
            ax2.set_title("Validation F1 Macro", fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(plots_dir / "metrics_curves.png", dpi=150)
            plt.close(fig)

def save_run_artifacts(trainer, run_dir: Path, eval_ds, test_ds) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train_logs, eval_logs = _extract_logs(trainer.state.log_history)
    print("\n📊 Generating evaluation plots...")
    plot_training_curves(train_logs, eval_logs, plots_dir)
    plot_roc_curves(trainer, eval_ds, plots_dir, split_name="validation")
    plot_roc_curves(trainer, test_ds, plots_dir, split_name="test")
    plot_confusion_matrix(trainer, eval_ds, plots_dir, split_name="validation")
    plot_confusion_matrix(trainer, test_ds, plots_dir, split_name="test")
    res_val = trainer.evaluate(eval_ds, metric_key_prefix="val")
    res_test = trainer.evaluate(test_ds, metric_key_prefix="test")
    eval_results = {"validation": res_val, "test": res_test}
    (run_dir / "eval_results.json").write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n📊 Final Results:")
    print(f"  Validation - Accuracy: {res_val.get('val_accuracy', 0):.4f}")
    print(f"  Validation - F1 Macro: {res_val.get('val_f1_macro', 0):.4f}")
    print(f"  Test - Accuracy: {res_test.get('test_accuracy', 0):.4f}")
    print(f"  Test - F1 Macro: {res_test.get('test_f1_macro', 0):.4f}")
    print(f"\n✅ All plots saved to: {plots_dir}")
    return eval_results

# ============================================================
# EXPERIMENT MANAGEMENT
# ============================================================
def is_experiment_finished(run_dir: Path) -> bool:
    if (run_dir / "eval_results.json").exists() or (run_dir / "model_config.json").exists():
        return True
    checkpoint_dirs = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
    if not checkpoint_dirs:
        return False
    def get_step(d):
        try:
            return int(d.name.split("-")[-1])
        except ValueError:
            return -1
    latest_ckpt = max(checkpoint_dirs, key=get_step)
    trainer_state_file = latest_ckpt / "trainer_state.json"
    if trainer_state_file.exists():
        try:
            state = json.loads(trainer_state_file.read_text(encoding="utf-8"))
            max_steps = state.get("max_steps", 0)
            global_step = state.get("global_step", 0)
            if max_steps > 0 and global_step >= max_steps:
                return True
        except Exception:
            pass
    return False

def get_experiment_target(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = [d for d in sorted(output_root.glob("*")) if d.is_dir()]
    if not run_dirs:
        return None, None
    latest_run_dir = run_dirs[-1]
    if is_experiment_finished(latest_run_dir):
        print(f"Latest experiment '{latest_run_dir.name}' is completed. Starting a new experiment.")
        return None, None
    else:
        checkpoint_dirs = [d for d in latest_run_dir.glob("checkpoint-*") if d.is_dir()]
        def get_step(d):
            try:
                return int(d.name.split("-")[-1])
            except ValueError:
                return -1
        valid_ckpts = [d for d in checkpoint_dirs if get_step(d) >= 0]
        if valid_ckpts:
            latest_ckpt = max(valid_ckpts, key=get_step)
            print(f"Latest experiment '{latest_run_dir.name}' is UNFINISHED. Resuming from checkpoint: {latest_ckpt}")
            return latest_run_dir, latest_ckpt
        else:
            print(f"Latest experiment '{latest_run_dir.name}' is unfinished (no checkpoints yet). Resuming experiment folder: {latest_run_dir}")
            return latest_run_dir, None

# ============================================================
# TRANSLATION FUNCTIONS
# ============================================================
def translate_texts(texts, batch_size=128):
    if len(texts) == 0:
        return []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(device)
    tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL, use_fast=False)
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        translated_tokens = model.generate(**inputs, max_length=128, num_beams=1)
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated_texts.extend([t.strip() for t in decoded])
    return translated_texts

def load_or_create_translated_data():
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        print(f"\n⚠️ Missing input data file: {data_path}")
        return None
    if TRANSLATED_CSV_FILE.exists():
        try:
            df = pd.read_csv(TRANSLATED_CSV_FILE)
            if {"text_french", "topic"}.issubset(df.columns):
                print(f"\nLoaded translated data from {TRANSLATED_CSV_FILE}")
                return df
            print(f"\n⚠️ Translated file {TRANSLATED_CSV_FILE} is missing required columns. Rebuilding from source data.")
        except Exception as exc:
            print(f"\n⚠️ Failed to load translated CSV: {exc}. Rebuilding from source data.")
    df = pd.read_csv(data_path)
    if "topic" not in df.columns or "text" not in df.columns:
        print(f"\n⚠️ Input data file must contain 'text' and 'topic' columns.")
        return None
    language = df.get("language", pd.Series(["unknown"] * len(df))).astype(str).str.lower().fillna("unknown")
    french_mask = language.isin({"fr", "fra", "french"})
    english_mask = language.isin({"en", "eng", "english"})
    df["text_french"] = df["text"].where(french_mask, None)
    df["is_translated"] = False
    df.loc[english_mask, "is_translated"] = True
    english_idx = df.loc[english_mask].index
    if MAX_TRANSLATED_SAMPLES is not None:
        english_idx = english_idx[:MAX_TRANSLATED_SAMPLES]
    if len(english_idx) > 0:
        print(f"\nTranslating {len(english_idx)} English samples to French...")
        english_texts = df.loc[english_idx, "text"].astype(str).tolist()
        translated_texts = translate_texts(english_texts, batch_size=128)
        df.loc[english_idx, "text_french"] = translated_texts
    df["text_french"] = df["text_french"].fillna(df["text"])
    df.loc[~french_mask & ~english_mask, "is_translated"] = False
    try:
        df.to_csv(TRANSLATED_CSV_FILE, index=False)
        print(f"\nSaved translated data to {TRANSLATED_CSV_FILE}")
    except Exception as exc:
        print(f"\n⚠️ Failed to save translated data: {exc}")
    return df

def create_synthetic_topics(df):
    if "topic" in df.columns:
        return df
    df["topic"] = "général"
    return df

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    base = Path(__file__).resolve().parent
    task_out_dir = base / OUT_DIR / TASK
    task_out_dir.mkdir(parents=True, exist_ok=True)

    resume_run_dir = None
    resume_checkpoint = None

    if RESUME_FROM_LAST_CHECKPOINT:
        resume_run_dir, resume_checkpoint = get_experiment_target(task_out_dir)

    if RESUME_FROM_LAST_CHECKPOINT and resume_run_dir is not None and resume_checkpoint is not None:
        run_dir = resume_run_dir
        run_ts = run_dir.name
        print(f"Resuming experiment '{run_dir.name}' from checkpoint: {resume_checkpoint}")
    elif RESUME_FROM_LAST_CHECKPOINT and resume_run_dir is not None:
        run_dir = resume_run_dir
        run_ts = run_dir.name
        print(f"Resuming experiment '{run_dir.name}' without a checkpoint")
    else:
        run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = task_out_dir / run_ts
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created new experiment folder: {run_dir}")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    print(f"Device: {'CUDA' if use_cuda else 'CPU'} | bf16={use_bf16} | fp16={use_fp16}")
    print(f"Evaluation percentage: {EVAL_PCT*100:.1f}%")

    hyperparams = {
        "run_timestamp": run_ts,
        "resume_from_last_checkpoint": RESUME_FROM_LAST_CHECKPOINT,
        "resumed_from_checkpoint": str(resume_checkpoint) if RESUME_FROM_LAST_CHECKPOINT and resume_checkpoint is not None else None,
        "model_name": MODEL_NAME,
        "task": TASK,
        "data_file": DATA_FILE,
        "translated_csv": str(TRANSLATED_CSV_FILE),
        "eval_percentage": EVAL_PCT,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_length": MAX_LEN,
        "pooling_strategy": "cls_token",
        "loss_function": "focal_loss_gamma_2.0_to_4.0",
        "eval_steps": EVAL_STEPS,
        "seed": SEED,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "class_weighted_loss": True,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "num_labels": NUM_LABELS,
        "dropout": 0.3,
        "max_grad_norm": 0.3,
        "early_stopping_patience": 5,
        "early_stopping_threshold": 0.001,
        "translation_model": TRANSLATION_MODEL,
        "max_translated_samples": MAX_TRANSLATED_SAMPLES,
        "gradient_checkpointing": False,
        "dataloader_workers": DATALOADER_WORKERS,
    }
    (run_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparams, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Run directory: {run_dir}")

    print("\n" + "=" * 80)
    print("🇫🇷 FRENCH TOPIC CLASSIFICATION - CamemBERT-large (FINAL)")
    print("=" * 80)
    
    df_all = load_or_create_translated_data()
    if df_all is None or len(df_all) == 0:
        print("\n⚠️ No data available. Please check your data file.")
        return

    print(f"\n📊 Dataset composition:")
    print(f"  Total samples: {len(df_all):,}")
    print(f"  French originals: {len(df_all[df_all['is_translated'] == False]):,}")
    print(f"  Translated: {len(df_all[df_all['is_translated'] == True]):,}")
    
    df_all["text"] = df_all["text_french"]
    df_all["label"] = df_all["topic"].apply(parse_topic).astype(int)
    df_all = df_all[df_all["label"] != -1].copy()
    
    if len(df_all) == 0:
        print("❌ No valid topic data after filtering!")
        return

    # ============================================================
    # OVERSAMPLE ENVIRONMENT (CLASS 13)
    # ============================================================
    env_mask = df_all['label'] == 13
    if env_mask.sum() < 500:
        env_samples = df_all[env_mask]
        repeats = int(np.ceil(500 / max(1, len(env_samples))))
        if repeats > 0:
            df_all = pd.concat([df_all] + [env_samples] * repeats, ignore_index=True)
            print(f"\n🌱 Oversampled Environment to {len(df_all[df_all['label'] == 13])} samples")

    print(f"\n📊 Final data distribution ({len(df_all)} samples):")
    for label, count in df_all["label"].value_counts().items():
        print(f"  {ID2LABEL[label]}: {count}")

    train_df, temp_df = train_test_split(
        df_all,
        test_size=0.2,
        stratify=df_all["label"],
        random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=SEED
    )
    print(f"\n📊 Data split:")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_all)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df_all)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ({len(test_df)/len(df_all)*100:.1f}%)")

    train_df = balance_training_frame(train_df)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print("\n📝 Creating datasets (no chunking, max_len=512)...")
    train_ds = SimpleTextDataset(train_df, tok, MAX_LEN)
    val_ds = SimpleTextDataset(val_df, tok, MAX_LEN)
    test_ds = SimpleTextDataset(test_df, tok, MAX_LEN)

    device = torch.device("cuda" if use_cuda else "cpu")
    class_weights = build_class_weights(train_df, NUM_LABELS, device)

    # ============================================================
    # BUILD CLASS-SPECIFIC GAMMAS
    # ============================================================
    counts = train_df["label"].value_counts().sort_index().values
    counts = np.maximum(counts, 1)
    class_gammas = 2.0 + (1.0 / np.sqrt(counts)) * 2.0
    class_gammas = np.clip(class_gammas, 2.0, 4.0)
    class_gammas = torch.tensor(class_gammas, dtype=torch.float32).to(device)
    
    print("\n🔬 Class-specific Focal Gammas:")
    for i in range(NUM_LABELS):
        print(f"  {ID2LABEL[i]:15s}: gamma = {class_gammas[i]:.2f}")

    # MODEL WITH FOCAL LOSS + CLASS GAMMAS
    model = LargeClassifierFocal(
        base_model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        class_weights=class_weights,
        dropout=0.3,
        focal_gamma=2.0,
        class_gammas=class_gammas
    )

    if RESUME_FROM_LAST_CHECKPOINT and resume_checkpoint is not None:
        sf_path = resume_checkpoint / "model.safetensors"
        bin_path = resume_checkpoint / "pytorch_model.bin"
        if sf_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(sf_path))
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model state from safetensors: {sf_path}")
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model state from: {bin_path}")

    total_steps = int(len(train_ds) * EPOCHS / BATCH_SIZE / GRAD_ACCUM)
    warmup_steps = int(0.1 * total_steps)

    print(f"\n📊 Training plan (FOCAL LOSS, NO GC, NO CHUNKING):")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",   # <-- 100% FIXED
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
        adam_epsilon=1e-8,
        max_grad_norm=0.3,
        dataloader_num_workers=DATALOADER_WORKERS,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=simple_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
    )

    trainer.train(
        resume_from_checkpoint=str(resume_checkpoint)
        if RESUME_FROM_LAST_CHECKPOINT and resume_checkpoint is not None
        else None
    )

    save_model(model, tok, ID2LABEL, LABEL2ID, MODEL_NAME, run_dir)
    save_run_artifacts(trainer, run_dir, val_ds, test_ds)

    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint:
        print(f"\nBest model checkpoint: {best_checkpoint}")
        print(f"Best eval F1 macro: {trainer.state.best_metric:.4f}")

    print(f"\nTraining Summary (ACCELERATED):")
    print(f"  - Total steps planned: {total_steps}")
    print(f"  - Steps completed: {trainer.state.global_step}")
    print(f"  - Epochs completed: {trainer.state.epoch:.2f} out of {EPOCHS}")
    print(f"  - Best F1: {trainer.state.best_metric:.4f}")
    print(f"  - Translated data saved to: {TRANSLATED_CSV_FILE}")

if __name__ == "__main__":
    main()