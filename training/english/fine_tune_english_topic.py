"""
fine_tune_english_topic.py
--------------------------
Base model : roberta-base (optimized)
Task       : English Topic Classification (18 categories)
Data source: data/global_data_libelised.csv
Output     : experiments/english_topic_optimized/run_<timestamp>/
"""
import os
import gc
import warnings
from pathlib import Path
import json
import random
import pickle
from datetime import datetime
import zipfile
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# ============================================================
# PATH CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "global_data_libelised.csv"
WORKING_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_BASE = WORKING_DIR / "english_topic"
EXPERIMENTS_BASE.mkdir(parents=True, exist_ok=True)


def find_latest_experiment():
    experiments = list(EXPERIMENTS_BASE.glob("run_*"))
    if not experiments:
        return None
    experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return experiments[0]


def is_experiment_complete(experiment_dir):
    summary_file = experiment_dir / "summary.json"
    if not summary_file.exists():
        return False
    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        return summary.get("status") == "completed"
    except Exception:
        return False


def get_latest_checkpoint(experiment_dir):
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    checkpoints = list(checkpoints_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    def get_step(path):
        try:
            return int(path.name.split("-")[1])
        except Exception:
            return 0
    latest = max(checkpoints, key=get_step)
    return str(latest)


def find_resume_point():
    latest_exp = find_latest_experiment()
    if latest_exp is None:
        print("\n🆕 No existing experiment found. Starting new experiment.")
        return None, None, False
    print(f"\n📁 Found existing experiment: {latest_exp.name}")
    if is_experiment_complete(latest_exp):
        print("✅ Experiment complete. Starting new experiment.")
        return None, None, False
    print("⏳ Experiment incomplete. Checking for checkpoints...")
    checkpoint = get_latest_checkpoint(latest_exp)
    if checkpoint is None:
        print("⚠️ No checkpoints found. Starting new experiment.")
        return None, None, False
    print(f"✅ Found checkpoint: {checkpoint}")
    return latest_exp, checkpoint, True


experiment_dir, resume_checkpoint, is_resuming = find_resume_point()

FORCE_NEW_EXPERIMENT = True
RESUME_FROM_CHECKPOINT = None

if FORCE_NEW_EXPERIMENT:
    is_resuming = False
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPERIMENT_DIR = EXPERIMENTS_BASE / f"run_{timestamp}"
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🆕 FORCED NEW EXPERIMENT: {EXPERIMENT_DIR.name}")
elif is_resuming:
    EXPERIMENT_DIR = experiment_dir
    RESUME_FROM_CHECKPOINT = resume_checkpoint
    print(f"\n🔄 RESUMING FROM: {RESUME_FROM_CHECKPOINT}")
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPERIMENT_DIR = EXPERIMENTS_BASE / f"run_{timestamp}"
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🆕 NEW EXPERIMENT: {EXPERIMENT_DIR.name}")

CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
BEST_MODEL_DIR = EXPERIMENT_DIR / "best_model"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
LOGS_DIR = EXPERIMENT_DIR / "logs"

for dir_path in [CHECKPOINTS_DIR, BEST_MODEL_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"📁 EXPERIMENT DIRECTORY: {EXPERIMENT_DIR}")
print("=" * 80)


def load_and_prepare_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    df_english = df[df["language"].str.lower().str.strip() == "en"].copy()
    df_english = df_english[["text", "topic"]].dropna()
    df_english["text"] = df_english["text"].astype(str).str.strip()
    df_english = df_english[df_english["text"].str.len() > 10]
    df_english = df_english[df_english["text"].str.len() < 3000]
    df_english = df_english[df_english["text"].str.len() > 0]

    label_encoder = LabelEncoder()
    df_english["label"] = label_encoder.fit_transform(df_english["topic"])
    num_labels = len(label_encoder.classes_)

    train_df, temp_df = train_test_split(
        df_english, 
        test_size=0.2, 
        stratify=df_english["label"], 
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df["label"], 
        random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return df_english, train_df, val_df, test_df, label_encoder, num_labels


data_info_file = RESULTS_DIR / "data_info.json"
if is_resuming and data_info_file.exists():
    print("\n📊 Loading saved data info...")
    with open(data_info_file, 'r') as f:
        data_info = json.load(f)
    df_english, train_df, val_df, test_df, label_encoder, NUM_LABELS = load_and_prepare_data()
    print("✅ Data loaded (using saved configuration)")
else:
    print("\n📊 Loading and preparing data...")
    df_english, train_df, val_df, test_df, label_encoder, NUM_LABELS = load_and_prepare_data()
    print("✅ Data loaded")

print(f"\nTopic mapping: {dict(enumerate(label_encoder.classes_))}")
print(f"Number of classes: {NUM_LABELS}")
print(f"Train samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")
print(f"Test samples: {len(test_df):,}")

if not is_resuming or not data_info_file.exists():
    data_info = {
        "total_samples": len(df_english),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "class_distribution": df_english["topic"].value_counts().to_dict(),
        "class_mapping": dict(enumerate(label_encoder.classes_)),
        "num_labels": NUM_LABELS,
    }
    with open(RESULTS_DIR / "data_info.json", "w") as f:
        json.dump(data_info, f, indent=2)


class RobertaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"⏳ Tokenizing {len(self.texts)} samples...")
        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        print(f"✅ Tokenization complete")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    f1_w = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )[2]
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_weighted": f1_w,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    
    fig1, ax1 = plt.subplots(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8}, ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title(f'{title} - Raw Counts', fontsize=14)
    ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(class_names, rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_raw_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(16, 12))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 8}, ax=ax2)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title(f'{title} - Normalized', fontsize=14)
    ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(class_names, rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_norm_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_performance_metrics(y_true, y_pred, class_names, title, save_dir):
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   zero_division=0, output_dict=True)
    
    classes = [c for c in class_names if c in report]
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]
    
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(classes)))[::-1]
    bars = ax1.barh(classes, precision, color=colors)
    ax1.set_xlabel('Precision', fontsize=12)
    ax1.set_title(f'{title} - Precision per Class', fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)
    for bar, val in zip(bars, precision):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / f'precision_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(classes)))[::-1]
    bars = ax2.barh(classes, recall, color=colors)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_title(f'{title} - Recall per Class', fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars, recall):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / f'recall_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(classes)))[::-1]
    bars = ax3.barh(classes, f1, color=colors)
    ax3.set_xlabel('F1 Score', fontsize=12)
    ax3.set_title(f'{title} - F1 per Class', fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Target (0.8)')
    ax3.legend()
    for bar, val in zip(bars, f1):
        ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / f'f1_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(classes)))[::-1]
    bars = ax4.barh(classes, support, color=colors)
    ax4.set_xlabel('Number of Samples', fontsize=12)
    ax4.set_title(f'{title} - Support per Class', fontsize=14)
    ax4.grid(True, alpha=0.3)
    for bar, val in zip(bars, support):
        ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / f'support_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return report


def plot_training_history(history, save_dir):
    steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    eval_accs = []
    eval_f1s = []
    learning_rates = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'learning_rate' in entry and 'step' in entry:
            learning_rates.append((entry['step'], entry['learning_rate']))
        if 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
            if 'eval_accuracy' in entry:
                eval_accs.append((entry['step'], entry['eval_accuracy']))
            if 'eval_f1' in entry:
                eval_f1s.append((entry['step'], entry['eval_f1']))
    
    if train_losses:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(steps, train_losses, label='Train Loss', color='blue', linewidth=2)
        ax1.set_xlabel('Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Over Steps', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'train_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if eval_losses:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(eval_steps, eval_losses, label='Validation Loss', color='orange', linewidth=2, marker='o')
        ax2.set_xlabel('Steps', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Validation Loss Over Steps', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'val_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if train_losses:
        ax3.plot(steps, train_losses, label='Train Loss', color='blue', linewidth=2, alpha=0.7)
    if eval_losses:
        ax3.plot(eval_steps, eval_losses, label='Validation Loss', color='orange', linewidth=2)
    ax3.set_xlabel('Steps', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Training vs Validation Loss', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'train_val_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if eval_accs:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        eval_steps_acc, eval_acc_values = zip(*eval_accs)
        ax4.plot(eval_steps_acc, eval_acc_values, label='Accuracy', color='green', linewidth=2, marker='o')
        ax4.set_xlabel('Steps', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Validation Accuracy Over Steps', fontsize=14)
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'val_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if eval_f1s:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        eval_steps_f1, eval_f1_values = zip(*eval_f1s)
        ax5.plot(eval_steps_f1, eval_f1_values, label='F1 Macro', color='red', linewidth=2, marker='s')
        ax5.set_xlabel('Steps', fontsize=12)
        ax5.set_ylabel('F1 Score', fontsize=12)
        ax5.set_title('Validation F1 (Macro) Over Steps', fontsize=14)
        ax5.set_ylim(0, 1)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'val_f1.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if eval_accs and eval_f1s:
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        eval_steps_acc, eval_acc_values = zip(*eval_accs)
        eval_steps_f1, eval_f1_values = zip(*eval_f1s)
        ax6.plot(eval_steps_acc, eval_acc_values, label='Accuracy', color='green', linewidth=2, marker='o')
        ax6.plot(eval_steps_f1, eval_f1_values, label='F1 Macro', color='red', linewidth=2, marker='s')
        ax6.set_xlabel('Steps', fontsize=12)
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Validation Accuracy & F1 Over Steps', fontsize=14)
        ax6.set_ylim(0, 1)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'val_accuracy_f1.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if learning_rates:
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        lr_steps, lr_values = zip(*learning_rates)
        ax7.plot(lr_steps, lr_values, label='Learning Rate', color='purple', linewidth=2)
        ax7.set_xlabel('Steps', fontsize=12)
        ax7.set_ylabel('Learning Rate', fontsize=12)
        ax7.set_title('Learning Rate Schedule', fontsize=14)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'learning_rate.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_class_distribution(labels, class_names, title, save_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    unique, counts = np.unique(labels, return_counts=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(unique)))
    bars = ax.bar([class_names[i] for i in unique], counts, color=colors)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'{title} - Class Distribution', fontsize=14)
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'class_distribution_{title.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(train_metrics, val_metrics, test_metrics, save_dir):
    metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    train_values = [train_metrics[m] for m in metrics]
    val_values = [val_metrics[m] for m in metrics]
    test_values = [test_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, train_values, width, label='Train', color='blue', alpha=0.7)
    bars2 = ax.bar(x, val_values, width, label='Validation', color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, test_values, width, label='Test', color='green', alpha=0.7)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison Across Datasets', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1 Macro', 'F1 Weighted'])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_and_plot(model, dataset, class_names, split_name, save_dir):
    print(f"\n📊 Evaluating on {split_name} set...")
    
    predictions = model.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='macro', zero_division=0
    )
    f1_weighted = precision_recall_fscore_support(
        true_labels, preds, average='weighted', zero_division=0
    )[2]
    
    print(f"{split_name} - Acc: {accuracy:.4f}, F1 Macro: {f1:.4f}, F1 Weighted: {f1_weighted:.4f}")
    
    plot_confusion_matrix(true_labels, preds, class_names, split_name, save_dir)
    report = plot_performance_metrics(true_labels, preds, class_names, split_name, save_dir)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'per_class': report,
    }


MODEL_NAME = "roberta-base"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 6
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.08
MAX_GRAD_NORM = 0.3
LABEL_SMOOTHING = 0.25
DROPOUT_RATE = 0.35
GRADIENT_ACCUMULATION_STEPS = 4

print("\n" + "="*60)
print("🚀 PREPARING DATA FOR ROBERTA-BASE (OPTIMIZED)")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = RobertaDataset(
    train_df["text"].values, 
    train_df["label"].values, 
    tokenizer,
    max_length=MAX_LEN
)
val_dataset = RobertaDataset(
    val_df["text"].values, 
    val_df["label"].values, 
    tokenizer,
    max_length=MAX_LEN
)
test_dataset = RobertaDataset(
    test_df["text"].values, 
    test_df["label"].values, 
    tokenizer,
    max_length=MAX_LEN
)

plot_class_distribution(
    train_df["label"].values, 
    label_encoder.classes_, 
    'Train', 
    PLOTS_DIR
)
plot_class_distribution(
    val_df["label"].values, 
    label_encoder.classes_, 
    'Validation', 
    PLOTS_DIR
)
plot_class_distribution(
    test_df["label"].values, 
    label_encoder.classes_, 
    'Test', 
    PLOTS_DIR
)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"].values),
    y=train_df["label"].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("\n" + "="*60)
print("🚀 LOADING ROBERTA-BASE MODEL (OPTIMIZED)")
print("="*60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)

model.config.classifier_dropout = DROPOUT_RATE
model.config.hidden_dropout_prob = DROPOUT_RATE
model.config.attention_probs_dropout_prob = DROPOUT_RATE

print(f"\n✅ Model loaded! Parameters: {model.num_parameters():,}")
model = model.to(device)

run_name = EXPERIMENT_DIR.name
run_timestamp = run_name

hyperparams_file = RESULTS_DIR / "hyperparameters.json"
if not hyperparams_file.exists():
    hyperparams = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "label_smoothing": LABEL_SMOOTHING,
        "dropout_rate": DROPOUT_RATE,
        "seed": 42,
        "num_labels": NUM_LABELS,
        "classes": list(label_encoder.classes_),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "is_resuming": is_resuming,
        "resumed_from": RESUME_FROM_CHECKPOINT if is_resuming else None,
    }
    with open(hyperparams_file, "w") as f:
        json.dump(hyperparams, f, indent=2)

training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=False,
    optim="adamw_torch",
    dataloader_num_workers=0,
    report_to="none",
    seed=42,
    eval_strategy="epoch",
    logging_steps=50,
    max_grad_norm=MAX_GRAD_NORM,
    remove_unused_columns=True,
    logging_first_step=True,
    run_name=f"optimized_roberta_{run_name}",
    disable_tqdm=False,
    dataloader_drop_last=True,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EnhancedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, use_focal=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.scaler = GradScaler()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        with autocast():
            outputs = model(**inputs)
            logits = outputs.logits
            
            if self.use_focal and self.state.global_step > 100:
                if self.class_weights is not None:
                    weight = self.class_weights[labels]
                    loss = (self.focal_loss(logits, labels) * weight).mean()
                else:
                    loss = self.focal_loss(logits, labels)
            else:
                if self.class_weights is not None:
                    loss = F.cross_entropy(
                        logits, 
                        labels, 
                        weight=self.class_weights,
                        label_smoothing=self.label_smoothing
                    )
                else:
                    loss = F.cross_entropy(
                        logits, 
                        labels, 
                        label_smoothing=self.label_smoothing
                    )
        
        return (loss, outputs) if return_outputs else loss


trainer = EnhancedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    label_smoothing=LABEL_SMOOTHING,
    use_focal=True,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.005
        )
    ],
)

print("\n" + "="*60)
print("🚀 STARTING OPTIMIZED ROBERTA-BASE TRAINING")
print("="*60)

try:
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
except Exception as e:
    print(f"\n⚠️ Training error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n💾 Saving model...")

trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))
joblib.dump(label_encoder, BEST_MODEL_DIR / 'label_encoder.pkl')

model_info = {
    'model_name': MODEL_NAME,
    'num_labels': NUM_LABELS,
    'class_names': label_encoder.classes_.tolist(),
    'run_timestamp': run_timestamp,
    'enhanced': True,
    'version': 'optimized',
    'use_focal_loss': True,
    'label_smoothing': LABEL_SMOOTHING,
    'dropout_rate': DROPOUT_RATE,
    'weight_decay': WEIGHT_DECAY,
    'max_grad_norm': MAX_GRAD_NORM,
    'gradient_checkpointing': False,
}

with open(BEST_MODEL_DIR / 'model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

plot_training_history(trainer.state.log_history, PLOTS_DIR)

log_history = trainer.state.log_history
with open(LOGS_DIR / "training_logs.json", "w") as f:
    json.dump(log_history, f, indent=2)

print("\n" + "="*60)
print("📊 EVALUATING MODEL")
print("="*60)

class_names = label_encoder.classes_.tolist()

train_metrics = evaluate_and_plot(
    trainer, train_dataset, class_names, 'Train', PLOTS_DIR
)

val_metrics = evaluate_and_plot(
    trainer, val_dataset, class_names, 'Validation', PLOTS_DIR
)

test_metrics = evaluate_and_plot(
    trainer, test_dataset, class_names, 'Test', PLOTS_DIR
)

plot_metrics_comparison(train_metrics, val_metrics, test_metrics, PLOTS_DIR)

all_metrics = {
    'run_id': run_timestamp,
    'timestamp': datetime.now().isoformat(),
    'model': MODEL_NAME,
    'model_type': 'roberta-base-optimized',
    'num_labels': NUM_LABELS,
    'data_split': {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
    },
    'train_metrics': train_metrics,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'hyperparameters': {
        'batch_size': TRAIN_BATCH_SIZE,
        'effective_batch_size': TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'max_length': MAX_LEN,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'max_grad_norm': MAX_GRAD_NORM,
        'dropout_rate': DROPOUT_RATE,
        'fp16': training_args.fp16,
        'gradient_checkpointing': training_args.gradient_checkpointing,
        'optimizer': 'adamw_torch',
        'scheduler': 'cosine',
        'use_focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'early_stopping_patience': 2,
        'early_stopping_threshold': 0.005,
    }
}

with open(RESULTS_DIR / 'all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=-1)
test_true = test_predictions.label_ids

n_preds = len(test_true)
results_df = pd.DataFrame({
    'text': test_df['text'].values[:n_preds],
    'true_label': test_true,
    'true_topic': label_encoder.inverse_transform(test_true),
    'pred_label': test_preds,
    'pred_topic': label_encoder.inverse_transform(test_preds),
})
results_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

class_report = classification_report(
    test_true, 
    test_preds, 
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)
with open(RESULTS_DIR / "classification_report.json", "w") as f:
    json.dump(class_report, f, indent=2)

cm = confusion_matrix(test_true, test_preds)
cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")

summary = {
    "experiment_name": EXPERIMENT_DIR.name,
    "model_name": MODEL_NAME,
    "model_type": "optimized",
    "status": "completed",
    "completed_at": datetime.now().isoformat(),
    "metrics": {
        "train": {
            "accuracy": float(train_metrics["accuracy"]),
            "f1_macro": float(train_metrics["f1_macro"]),
            "f1_weighted": float(train_metrics["f1_weighted"]),
        },
        "validation": {
            "accuracy": float(val_metrics["accuracy"]),
            "f1_macro": float(val_metrics["f1_macro"]),
            "f1_weighted": float(val_metrics["f1_weighted"]),
        },
        "test": {
            "accuracy": float(test_metrics["accuracy"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "f1_weighted": float(test_metrics["f1_weighted"]),
        }
    },
    "resumed_from": RESUME_FROM_CHECKPOINT if is_resuming else None,
    "files": {
        "hyperparameters": str(RESULTS_DIR / "hyperparameters.json"),
        "metrics": str(RESULTS_DIR / "all_metrics.json"),
        "classification_report": str(RESULTS_DIR / "classification_report.json"),
        "confusion_matrix_csv": str(RESULTS_DIR / "confusion_matrix.csv"),
        "confusion_matrix_plots": {
            "train_raw": str(PLOTS_DIR / "confusion_matrix_raw_train.png"),
            "train_norm": str(PLOTS_DIR / "confusion_matrix_norm_train.png"),
            "validation_raw": str(PLOTS_DIR / "confusion_matrix_raw_validation.png"),
            "validation_norm": str(PLOTS_DIR / "confusion_matrix_norm_validation.png"),
            "test_raw": str(PLOTS_DIR / "confusion_matrix_raw_test.png"),
            "test_norm": str(PLOTS_DIR / "confusion_matrix_norm_test.png"),
        },
        "predictions": str(RESULTS_DIR / "predictions.csv"),
        "training_curves": str(PLOTS_DIR),
        "best_model": str(BEST_MODEL_DIR),
        "checkpoints": str(CHECKPOINTS_DIR),
        "logs": str(LOGS_DIR),
    }
}

with open(EXPERIMENT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("📊 FINAL RESULTS")
print("="*60)

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n✅ Done!")
