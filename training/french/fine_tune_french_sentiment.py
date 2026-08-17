"""
french_sentiment_camembert_large_final.py
----------------------------------------
French Sentiment Analysis - CamemBERT-large FINAL
- Target: >80% Accuracy
- Optimized for RTX 3050 (4GB VRAM)
- ALL plots included (Test set only)
- 80/10/10 data split
"""

import gc
import json
import os
import warnings
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# GPU SETUP
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# PATHS
# ============================================================
DATA_FILE = "fine_tune_data/global_data_merged.csv"
BASE_DIR = Path(__file__).resolve().parent
TRANSLATED_DATA_DIR = BASE_DIR / "translated_data"
TRANSLATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "experiments" / "french_sentiment_camembert_large_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FINAL HYPERPARAMETERS - TARGET >80%
# ============================================================
MODEL_NAME = "camembert/camembert-large"
MAX_LEN = 384

# Batch sizes
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch: 32

# FINAL tuned parameters
LEARNING_RATE = 1e-5
NUM_EPOCHS = 15
WARMUP_RATIO = 0.2
WEIGHT_DECAY = 0.05

# Focal Loss - stronger for POSITIVE
FOCAL_ALPHA = 0.65
FOCAL_GAMMA = 4.0

# Evaluation
EVAL_STEPS = 1000
SAVE_STEPS = 1000
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_THRESHOLD = 0.003
SEED = 42
SAMPLE_RATIO = 1.0

# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_positive_class(df, augment_factor=0.5):
    """Smart augmentation for POSITIVE class"""
    positive_df = df[df['label'] == 2].copy()
    if len(positive_df) == 0:
        return df
    
    n_augment = int(len(positive_df) * augment_factor)
    if n_augment == 0:
        return df
    
    print(f"\n📈 Augmenting POSITIVE class:")
    print(f"  Original POSITIVE: {len(positive_df):,}")
    print(f"  Adding: {n_augment:,} augmented samples")
    
    intensifiers = [
        " très", " vraiment", " extrêmement", " absolument", 
        " totalement", " complètement", " particulièrement",
        " tellement", " incroyablement", " formidablement",
        " fantastiquement", " merveilleusement"
    ]
    
    augmented_parts = []
    
    for _ in range(n_augment):
        row = positive_df.sample(1, random_state=SEED)
        text = row['text'].values[0]
        if np.random.random() > 0.5:
            text = np.random.choice(intensifiers) + " " + text
        else:
            text = text + np.random.choice(intensifiers)
        row['text'] = text
        augmented_parts.append(row)
    
    result = pd.concat([df] + augmented_parts, ignore_index=True)
    final_pos = len(result[result['label'] == 2])
    print(f"  Final POSITIVE: {final_pos:,}")
    return result

# ============================================================
# DATA LOADING
# ============================================================

def load_cached_data():
    translated_file = TRANSLATED_DATA_DIR / "translated_french_data_all.parquet"
    real_french_file = TRANSLATED_DATA_DIR / "real_french_data.parquet"
    metadata_file = TRANSLATED_DATA_DIR / "translation_metadata.json"
    
    if not translated_file.exists() or not real_french_file.exists():
        print("❌ Translated data not found!")
        return None, None
    
    print(f"\n📁 Loading cached translated data...")
    df_translated = pd.read_parquet(translated_file)
    df_french_real = pd.read_parquet(real_french_file)
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"📊 Translation info:")
            print(f"  - Real French: {metadata['real_french_count']:,}")
            print(f"  - Translated: {metadata['translated_count']:,}")
            print(f"  - Date: {metadata['translation_date']}")
    
    if SAMPLE_RATIO < 1.0:
        df_translated = df_translated.sample(frac=SAMPLE_RATIO, random_state=SEED)
        print(f"📊 Sampled {len(df_translated):,} translated samples ({SAMPLE_RATIO*100:.0f}%)")
    
    return df_french_real, df_translated

print("\n" + "=" * 80)
print("🇫🇷 FRENCH SENTIMENT ANALYSIS - CamemBERT-large FINAL")
print("=" * 80)

df_french_real, df_translated = load_cached_data()

if df_french_real is None or df_translated is None:
    print("\n⚠️ No cached data found. Please run the translation script first.")
    exit()

# Combine datasets
print("\n📊 Combining datasets...")
df_combined = pd.concat([df_french_real, df_translated], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['text'])

print(f"\n📊 Final dataset composition:")
print(f"  Real French samples: {len(df_french_real):,}")
print(f"  Translated samples: {len(df_translated):,}")
print(f"  Total combined: {len(df_combined):,}")

print("\n📊 Combined sentiment distribution:")
for sentiment, count in df_combined['sentiment'].value_counts().items():
    print(f"  {sentiment}: {count:,} ({count/len(df_combined)*100:.1f}%)")

# ============================================================
# PREPARE DATASET - 80/10/10 SPLIT
# ============================================================
print("\n" + "=" * 80)
print("🔧 PREPARING DATASET FOR TRAINING")
print("=" * 80)

label_encoder = LabelEncoder()
df_combined["label"] = label_encoder.fit_transform(df_combined["sentiment"])
NUM_LABELS = len(label_encoder.classes_)

id2label = {int(i): str(label) for i, label in enumerate(label_encoder.classes_)}
label2id = {str(label): int(i) for i, label in enumerate(label_encoder.classes_)}

with open(OUTPUT_DIR / "label_mapping.json", "w", encoding='utf-8') as f:
    json.dump(id2label, f, indent=2, ensure_ascii=False)

print("\n📊 Splitting data (80/10/10)...")
train_df, temp_df = train_test_split(
    df_combined,
    test_size=0.2,
    stratify=df_combined["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED
)

print(f"\n📊 Data split:")
print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_combined)*100:.1f}%)")
print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df_combined)*100:.1f}%)")
print(f"  Test: {len(test_df):,} ({len(test_df)/len(df_combined)*100:.1f}%)")

# ============================================================
# AUGMENT POSITIVE CLASS - INCREASED TO 50%
# ============================================================
train_df = augment_positive_class(train_df, augment_factor=0.5)

# ============================================================
# DATASET CLASSES
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": str(self.texts[idx]), "labels": int(self.labels[idx])}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = SentimentDataset(train_df["text"].values, train_df["label"].values)
val_dataset = SentimentDataset(val_df["text"].values, val_df["label"].values)
test_dataset = SentimentDataset(test_df["text"].values, test_df["label"].values)

def collate_fn(batch):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt",
    )
    encodings["labels"] = labels
    return encodings

# ============================================================
# CLASS WEIGHTS - STRONG POSITIVE FOCUS
# ============================================================
class_counts = np.bincount(train_df["label"].values)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_LABELS

# Strong POSITIVE focus
class_weights[2] = class_weights[2] * 2.5
class_weights[1] = class_weights[1] * 0.7
class_weights[0] = class_weights[0] * 1.1

class_weights = np.clip(class_weights, 0.5, 5.0)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print(f"\n📊 Enhanced Class weights:")
for i, weight in enumerate(class_weights_tensor):
    print(f"  {label_encoder.classes_[i]}: {weight:.4f}")

# ============================================================
# CUSTOM TRAINER
# ============================================================
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, alpha=0.65, gamma=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.training_losses = []
        self.best_f1 = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            weight = weights[labels.view(-1)]
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
            focal_loss = focal_loss * weight
        else:
            focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        
        loss = focal_loss.mean()
        self.training_losses.append(loss.item())
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if metric_key_prefix == "eval" and "eval_f1_macro" in metrics:
            current_f1 = metrics["eval_f1_macro"]
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                print(f"\n🏆 New best F1: {self.best_f1:.4f}")
        
        return metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    report = classification_report(labels, preds, 
                                   target_names=['NEGATIVE', 'NEUTRAL', 'POSITIVE'],
                                   output_dict=True, zero_division=0)
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": p,
        "recall_macro": r,
        "f1_NEGATIVE": report['NEGATIVE']['f1-score'],
        "f1_NEUTRAL": report['NEUTRAL']['f1-score'],
        "f1_POSITIVE": report['POSITIVE']['f1-score'],
        "recall_NEGATIVE": report['NEGATIVE']['recall'],
        "recall_NEUTRAL": report['NEUTRAL']['recall'],
        "recall_POSITIVE": report['POSITIVE']['recall'],
    }
    return metrics

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def _extract_logs(log_history):
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs

def plot_confusion_matrix(trainer, eval_dataset, run_dir, split_name):
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    cm = confusion_matrix(labels, preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], 
                yticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], 
                annot_kws={'size': 12})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {split_name.upper()}\n'
                 f'NEG={class_acc[0]:.1%} | NEU={class_acc[1]:.1%} | POS={class_acc[2]:.1%}', 
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / f'confusion_matrix_{split_name}.png', dpi=150)
    plt.close(fig)
    return class_acc

def save_run_artifacts(trainer, run_dir, eval_dataset, split_name="test"):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    train_logs, eval_logs = _extract_logs(trainer.state.log_history)
    
    res = trainer.evaluate(eval_dataset, metric_key_prefix=split_name)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\n{split_name.upper()}:")
    for key in [f'{split_name}_accuracy', f'{split_name}_f1_macro', 
                f'{split_name}_f1_NEGATIVE', f'{split_name}_f1_NEUTRAL', 
                f'{split_name}_f1_POSITIVE']:
        if key in res:
            print(f"  {key}: {res[key]:.4f}")
    
    plot_confusion_matrix(trainer, eval_dataset, plots_dir, split_name)
    
    def _steps(logs, key): return [e["step"] for e in logs if key in e]
    def _vals(logs, key): return [e[key] for e in logs if key in e]
    STYLE = dict(linewidth=2)
    
    if train_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"), 
                color="#e07b39", label="Train loss", **STYLE)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "train_loss.png", dpi=150)
        plt.close(fig)
    
    if eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"), 
                color="#4a90d9", label="Eval loss", **STYLE)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Evaluation Loss", fontsize=14)
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
            ax2.plot(steps, accs, color="#9b59b6", label="Accuracy", 
                     linestyle="--", **STYLE)
            ax1.set_xlabel("Step", fontsize=12)
            ax1.set_ylabel("F1 Macro", color="#2ecc71", fontsize=12)
            ax2.set_ylabel("Accuracy", color="#9b59b6", fontsize=12)
            ax1.set_title("Eval F1 Macro & Accuracy", fontsize=14)
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
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Learning Curve — Train vs Eval Loss", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_curve.png", dpi=150)
        plt.close(fig)
    
    print(f"\n✅ All plots saved to: {plots_dir}")
    return res

# ============================================================
# LOAD MODEL
# ============================================================
print("\n🚀 Loading CamemBERT-large model...")
print("⚠️ Model size: 435M parameters (~2.5GB VRAM)")
print("⚠️ Gradient checkpointing: DISABLED for speed")
print(f"📊 Training with {len(train_dataset):,} samples")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# ============================================================
# TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    max_grad_norm=0.5,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=True,
    gradient_checkpointing=False,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    seed=SEED,
    logging_steps=50,
    dataloader_num_workers=0,
    save_only_model=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,
    alpha=FOCAL_ALPHA,
    gamma=FOCAL_GAMMA,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE, 
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )
    ],
)

# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 80)
print("🚀 STARTING FINAL TRAINING - CamemBERT-large")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Parameters: 435M")
print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Max length: {MAX_LEN}")
print(f"Batch size: {TRAIN_BATCH_SIZE} (effective: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Num epochs: {NUM_EPOCHS}")
print(f"Focal Alpha: {FOCAL_ALPHA}")
print(f"Focal Gamma: {FOCAL_GAMMA}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Warmup Ratio: {WARMUP_RATIO}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"Gradient Checkpointing: DISABLED ⚡")
print(f"POSITIVE Augmentation: 50%")
print(f"Target: >80% Accuracy")
print("=" * 80)

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================
best_model_dir = OUTPUT_DIR / "best_model"
trainer.save_model(str(best_model_dir))
tokenizer.save_pretrained(str(best_model_dir))
print(f"\n✅ Model saved to: {best_model_dir}")

# ============================================================
# GENERATE ALL PLOTS - TEST SET ONLY
# ============================================================
print("\n" + "=" * 80)
print("📊 GENERATING ALL PLOTS")
print("=" * 80)

save_run_artifacts(trainer, OUTPUT_DIR, test_dataset, "test")

# ============================================================
# SAVE PREDICTIONS
# ============================================================
print("\n📊 Saving predictions...")
eval_results = trainer.predict(test_dataset)
predictions = np.argmax(eval_results.predictions, axis=-1)
true_labels = eval_results.label_ids

results_df = pd.DataFrame({
    'text': test_df['text'].values,
    'true_sentiment': label_encoder.inverse_transform(true_labels),
    'pred_sentiment': label_encoder.inverse_transform(predictions),
    'true_label': true_labels,
    'pred_label': predictions,
})
results_df.to_csv(OUTPUT_DIR / 'test_predictions.csv', index=False)

# ============================================================
# SUMMARY
# ============================================================
summary = {
    "experiment_name": "camembert_large_french_sentiment_final",
    "completed_at": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "model_size": "435M parameters (large)",
    "gradient_checkpointing": False,
    "focal_alpha": FOCAL_ALPHA,
    "focal_gamma": FOCAL_GAMMA,
    "augmentation": "POSITIVE class augmented (50%)",
    "data_split": "80/10/10 (train/val/test)",
    "total_samples": len(df_combined),
    "real_french_samples": len(df_french_real),
    "translated_samples": len(df_translated),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "classes": list(label_encoder.classes_),
    "class_distribution": df_combined['sentiment'].value_counts().to_dict(),
    "translated_data_location": str(TRANSLATED_DATA_DIR),
    "plots_location": str(PLOTS_DIR),
    "hyperparameters": {
        "max_len": MAX_LEN,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
        "gradient_checkpointing": False,
        "fp16": True,
        "eval_steps": EVAL_STEPS,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
    },
}

with open(OUTPUT_DIR / "summary.json", "w", encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("✅ FINAL EXPERIMENT COMPLETE!")
print("=" * 80)
print(f"\n📁 Output directory: {OUTPUT_DIR}")
print(f"📁 Plots directory: {PLOTS_DIR}")
print(f"\n📊 Data Split (80/10/10):")
print(f"  ✓ Train:      {len(train_df):,} samples")
print(f"  ✓ Validation: {len(val_df):,} samples")
print(f"  ✓ Test:       {len(test_df):,} samples")
print(f"\n📊 FINAL Parameters:")
print(f"  ✓ Max Length: {MAX_LEN}")
print(f"  ✓ Learning Rate: {LEARNING_RATE}")
print(f"  ✓ Focal Alpha: {FOCAL_ALPHA}")
print(f"  ✓ Focal Gamma: {FOCAL_GAMMA}")
print(f"  ✓ Weight Decay: {WEIGHT_DECAY}")
print(f"  ✓ Warmup Ratio: {WARMUP_RATIO}")
print(f"  ✓ POSITIVE Augmentation: 50%")
print(f"  ✓ POSITIVE Weight: {class_weights[2]:.4f}")
print(f"\n📊 Plots generated:")
print(f"  - confusion_matrix_test.png")
print(f"  - train_loss.png")
print(f"  - eval_loss.png")
print(f"  - eval_f1_accuracy.png")
print(f"  - learning_curve.png")

print(f"\n✅ Done!")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()