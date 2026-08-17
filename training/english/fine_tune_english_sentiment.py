import os
import sys
import gc
import json
import warnings
import shutil
from pathlib import Path
from datetime import datetime
import random
import re

# Enable UTF-8 output for Windows console emoji support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim

warnings.filterwarnings("ignore")

# ============================================================
# OPTIMIZATION FOR RTX 3050 6GB VRAM
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# ENHANCED PATHS AND HYPERPARAMETERS
# ============================================================
DATA_FILE = "fine_tune_data/global_data_merged.csv"
WORKING_DIR = Path("./sentiment_experiments")
EXPERIMENTS_BASE = WORKING_DIR / "english_sentiment_enhanced"
EXPERIMENTS_BASE.mkdir(parents=True, exist_ok=True)

# Enhanced hyperparameters
MODEL_NAME = "cardiffnlp/twitter-roberta-large-topic-sentiment-latest"
MAX_LEN = 160  # Slightly increased for better context

# Optimized batch sizes for 6GB VRAM
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

# Enhanced learning parameters
LEARNING_RATE = 5e-6  # Slightly lower for stability
NUM_EPOCHS = 12  # Increased epochs
WARMUP_RATIO = 0.1  # Warmup ratio instead of steps
EVAL_STEPS = 1000  # More frequent evaluation
SAVE_STEPS = 1000
SEED = 42
WEIGHT_DECAY = 0.01  # Reduced for better convergence
EARLY_STOPPING_PATIENCE = 10  # Increased patience
EARLY_STOPPING_THRESHOLD = 0.001

# Data augmentation settings
AUGMENT_PROB = 0.3
MAX_AUGMENTATIONS = 2

# ============================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================
class TextAugmenter:
    """Enhanced text augmentation techniques"""
    
    def __init__(self, augment_prob=0.3):
        self.augment_prob = augment_prob
        # Common synonym pairs for sentiment-conserving augmentation
        self.synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice', 'positive'],
            'bad': ['poor', 'terrible', 'awful', 'horrible', 'negative'],
            'happy': ['joyful', 'delighted', 'pleased', 'glad', 'cheerful'],
            'sad': ['unhappy', 'depressed', 'sorrowful', 'melancholy', 'down'],
            'big': ['large', 'huge', 'enormous', 'massive', 'great'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'modest'],
            'amazing': ['incredible', 'fantastic', 'wonderful', 'remarkable', 'extraordinary'],
            'terrible': ['horrible', 'awful', 'dreadful', 'atrocious', 'abysmal'],
            'love': ['adore', 'cherish', 'treasure', 'appreciate', 'enjoy'],
            'hate': ['despise', 'loathe', 'detest', 'abhor', 'dislike'],
        }
        
    def synonym_replacement(self, text):
        """Replace words with synonyms while preserving sentiment"""
        words = text.split()
        augmented_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.synonyms and random.random() < 0.15:
                synonym = random.choice(self.synonyms[word_lower])
                # Preserve capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def random_deletion(self, text):
        """Randomly delete words with low probability"""
        words = text.split()
        if len(words) < 3:
            return text
        
        keep_prob = 0.85
        augmented_words = [w for w in words if random.random() < keep_prob]
        
        if len(augmented_words) < 2:
            return text
        
        return ' '.join(augmented_words)
    
    def random_swap(self, text):
        """Swap adjacent words"""
        words = text.split()
        if len(words) < 3:
            return text
        
        for _ in range(1):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def augment_text(self, text):
        """Apply random augmentations to text"""
        if random.random() > self.augment_prob:
            return text
        
        augmented_text = text
        num_augmentations = random.randint(1, MAX_AUGMENTATIONS)
        
        for _ in range(num_augmentations):
            aug_type = random.choice([
                self.synonym_replacement,
                self.random_deletion,
                self.random_swap
            ])
            augmented_text = aug_type(augmented_text)
        
        return augmented_text

# ============================================================
# ENHANCED DATASET WITH AUGMENTATION
# ============================================================
class EnhancedDataset(Dataset):
    """Memory efficient dataset with augmentation"""
    def __init__(self, texts, labels, augment=False, augment_prob=0.3):
        self.texts = texts
        self.labels = labels
        self.augment = augment
        self.augmenter = TextAugmenter(augment_prob) if augment else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Apply augmentation
        if self.augment and self.augmenter:
            text = self.augmenter.augment_text(text)
        
        return {"text": text, "labels": label}

# ============================================================
# ENHANCED COLLATE FUNCTION
# ============================================================
def enhanced_collate_fn(batch, tokenizer, max_len):
    """Enhanced collate with better handling"""
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    
    # Tokenize with attention mask
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    encodings["labels"] = labels
    return encodings

# ============================================================
# ENHANCED FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================
class EnhancedFocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, focal_gamma=2.0, 
                 label_smoothing=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = logits.size(-1)
            smooth_labels = torch.full_like(logits, self.label_smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            labels = smooth_labels
        
        # Weighted focal loss
        if self.class_weights is not None:
            device = logits.device
            class_weights = self.class_weights.to(device)
        else:
            class_weights = None
        
        # Compute focal loss with label smoothing
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

# ============================================================
# ENHANCED METRICS
# ============================================================
def enhanced_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Calculate multiple metrics
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
        "per_class_f1": f1_class.tolist() if isinstance(f1_class, np.ndarray) else f1_class,
    }

# ============================================================
# ENHANCED DATA LOADING WITH STRATIFICATION
# ============================================================
def load_and_prepare_data_enhanced():
    """Load and prepare data with enhanced preprocessing"""
    print("\n📊 Loading and preparing data...")
    df = pd.read_csv(DATA_FILE)
    
    # Enhanced preprocessing
    df_english = df[df["language"].str.lower().str.strip() == "en"].copy()
    df_english = df_english[df_english["sentiment"] != "UNK"].copy()
    df_english = df_english[["text", "sentiment"]].dropna()
    
    # Clean text
    df_english["text"] = df_english["text"].apply(clean_text)
    
    # Sample if needed for memory
    if len(df_english) > 100000:  # Increased sampling threshold
        df_english = df_english.sample(100000, random_state=SEED)
        print(f"📊 Sampled to 100,000 examples")
    
    label_encoder = LabelEncoder()
    df_english["label"] = label_encoder.fit_transform(df_english["sentiment"])
    NUM_LABELS = len(label_encoder.classes_)
    
    # Stratified split with 70-15-15
    train_df, temp_df = train_test_split(
        df_english,
        test_size=0.3,
        stratify=df_english["label"],
        random_state=SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=SEED
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print("\n📊 Data Split (Enhanced):")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df_english)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} ({len(val_df)/len(df_english)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} ({len(test_df)/len(df_english)*100:.1f}%)")
    
    return df_english, train_df, val_df, test_df, label_encoder, NUM_LABELS

def clean_text(text):
    """Enhanced text cleaning"""
    if not isinstance(text, str):
        return str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters but keep sentiment-relevant ones
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle negation (basic)
    text = re.sub(r'\b(not|no|never|none)\s+', r'not_', text)
    
    return text

# ============================================================
# ENHANCED TRAINING LOOP WITH CROSS-VALIDATION
# ============================================================
def get_class_weights_enhanced(train_df, num_labels):
    """Enhanced class weight calculation with smoothing"""
    class_counts = np.bincount(train_df["label"].values)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_labels
    # Add smoothing to prevent extreme weights
    weights = weights * 0.9 + 0.1
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================
# ENHANCED CALLBACKS
# ============================================================
from transformers import TrainerCallback



class EarlyStoppingWithWarmup(EarlyStoppingCallback):
    """Enhanced early stopping with warmup period"""
    def __init__(self, early_stopping_patience=4, early_stopping_threshold=0.001, 
                 warmup_steps=1000):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.warmup_steps = warmup_steps
        
    def check_metric_value(self, args, state, control, metric_value):
        # Don't stop during warmup
        if state.global_step < self.warmup_steps:
            return False
        return super().check_metric_value(args, state, control, metric_value)

# ============================================================
# MEMORY OPTIMIZATION FUNCTIONS
# ============================================================
def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
    except:
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
        except:
            return 0
    latest = max(checkpoints, key=get_step)
    return str(latest)

def find_resume_point():
    latest_exp = find_latest_experiment()
    if latest_exp is None:
        print("\n🆕 Starting new enhanced experiment.")
        return None, None, False
    print(f"\n📁 Found existing experiment: {latest_exp.name}")
    if is_experiment_complete(latest_exp):
        print("✅ Experiment complete. Starting new enhanced experiment.")
        return None, None, False
    print("⏳ Experiment incomplete. Checking for checkpoints...")
    checkpoint = get_latest_checkpoint(latest_exp)
    if checkpoint is None:
        print("⚠️ No checkpoints found. Starting new enhanced experiment.")
        return None, None, False
    print(f"✅ Found checkpoint: {checkpoint}")
    return latest_exp, checkpoint, True

# ============================================================
# MAIN EXECUTION
# ============================================================
# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

experiment_dir, resume_checkpoint, is_resuming = find_resume_point()

if is_resuming:
    EXPERIMENT_DIR = experiment_dir
    RESUME_FROM_CHECKPOINT = resume_checkpoint
    print(f"\n🔄 RESUMING FROM: {RESUME_FROM_CHECKPOINT}")
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPERIMENT_DIR = EXPERIMENTS_BASE / f"run_{timestamp}"
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_FROM_CHECKPOINT = None
    print(f"\n🆕 NEW ENHANCED EXPERIMENT: {EXPERIMENT_DIR.name}")

CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"
BEST_MODEL_DIR = EXPERIMENT_DIR / "best_model"
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
LOGS_DIR = EXPERIMENT_DIR / "logs"

for dir_path in [CHECKPOINTS_DIR, BEST_MODEL_DIR, RESULTS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"📁 ENHANCED EXPERIMENT DIRECTORY: {EXPERIMENT_DIR}")
print("=" * 80)

# Load data
df_english, train_df, val_df, test_df, label_encoder, NUM_LABELS = load_and_prepare_data_enhanced()

print(f"\nSentiment mapping: {dict(enumerate(label_encoder.classes_))}")
print(f"Number of classes: {NUM_LABELS}")
print(f"Train samples: {len(train_df):,}")
print(f"Validation samples: {len(val_df):,}")
print(f"Test samples: {len(test_df):,}")

# Save data info
data_info = {
    "total_samples": len(df_english),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "class_distribution": df_english["sentiment"].value_counts().to_dict(),
    "class_mapping": dict(enumerate(label_encoder.classes_)),
    "num_labels": NUM_LABELS,
}
with open(RESULTS_DIR / "data_info.json", "w") as f:
    json.dump(data_info, f, indent=2)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create enhanced datasets
train_dataset = EnhancedDataset(
    train_df["text"].values, 
    train_df["label"].values,
    augment=True,
    augment_prob=AUGMENT_PROB
)
val_dataset = EnhancedDataset(
    val_df["text"].values, 
    val_df["label"].values,
    augment=False
)
test_dataset = EnhancedDataset(
    test_df["text"].values, 
    test_df["label"].values,
    augment=False
)

# Create collate function with tokenizer
def collate_fn(batch):
    return enhanced_collate_fn(batch, tokenizer, MAX_LEN)

# Calculate class weights
class_weights_tensor = get_class_weights_enhanced(train_df, NUM_LABELS)
print(f"\nEnhanced Class weights: {class_weights_tensor}")

# Load model
print("\n🔄 Loading enhanced model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)

model = model.to(device)
print("✅ Model loaded.")

# Save hyperparameters
hyperparams = {
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "warmup_ratio": WARMUP_RATIO,
    "eval_steps": EVAL_STEPS,
    "save_steps": SAVE_STEPS,
    "seed": SEED,
    "weight_decay": WEIGHT_DECAY,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
    "num_labels": NUM_LABELS,
    "classes": list(label_encoder.classes_),
    "class_weights": class_weights_tensor.tolist(),
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "test_samples": len(test_df),
    "focal_loss_gamma": 2.0,
    "label_smoothing": 0.05,
    "augment_prob": AUGMENT_PROB,
    "gpu": "RTX 3050 6GB",
    "enhancements": [
        "Data augmentation with synonym replacement",
        "Label smoothing",
        "Enhanced text cleaning",
        "Increased validation frequency",
        "Improved class weight calculation",
        "Longer warmup period",
        "Early stopping with warmup",
        "Memory monitoring callback",
        "Stratified data splits",
        "Increased epoch count",
    ]
}
with open(RESULTS_DIR / "hyperparameters.json", "w") as f:
    json.dump(hyperparams, f, indent=2)

# Enhanced training arguments
training_args = TrainingArguments(
    output_dir=str(CHECKPOINTS_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,  # Use ratio instead of steps
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    dataloader_num_workers=0,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED,
    eval_strategy="steps",
    save_strategy="steps",
    remove_unused_columns=False,
    logging_steps=50,  # More frequent logging
    max_grad_norm=0.5,
    adam_epsilon=1e-8,
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,
)

# Create enhanced trainer
trainer = EnhancedFocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=enhanced_compute_metrics,
    class_weights=class_weights_tensor,
    focal_gamma=2.0,
    label_smoothing=0.05,
    callbacks=[
        EarlyStoppingWithWarmup(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            warmup_steps=1000
        ),
    ],
)

print("\n" + "=" * 80)
print("🚀 STARTING ENHANCED TRAINING (RTX 3050 OPTIMIZED)")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Classes: {NUM_LABELS} ({', '.join(label_encoder.classes_)})")
print(f"Train samples: {len(train_dataset):,}")
print(f"Val samples: {len(val_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Max length: {MAX_LEN}")
print(f"Batch size: {TRAIN_BATCH_SIZE}")
print(f"Effective batch: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"Loss function: Focal Loss (gamma=2.0) + Label Smoothing (0.05)")
print(f"Data Augmentation: Enabled (prob={AUGMENT_PROB})")
print(f"FP16: Enabled")
print(f"Gradient Checkpointing: Enabled")
print(f"Checkpoint limit: {training_args.save_total_limit}")
print("=" * 80)

# Initial memory check
clear_memory()

# Train with enhanced monitoring
print("\n🚀 Starting enhanced training...")
max_retries = 3
for attempt in range(max_retries):
    try:
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️ OOM Error (attempt {attempt+1}/{max_retries})")
            clear_memory()
            if attempt == max_retries - 1:
                print("❌ Out of memory. Consider reducing MAX_LEN or batch size.")
                raise
            training_args.per_device_train_batch_size = max(2, training_args.per_device_train_batch_size // 2)
            print(f"Reduced batch size to: {training_args.per_device_train_batch_size}")
        else:
            raise

# Save model
trainer.save_model(str(BEST_MODEL_DIR))
tokenizer.save_pretrained(str(BEST_MODEL_DIR))
print(f"\n✅ Best model saved to: {BEST_MODEL_DIR}")

# Save training logs
log_history = trainer.state.log_history
with open(LOGS_DIR / "training_logs.json", "w") as f:
    json.dump(log_history, f, indent=2)

# Enhanced plotting
def plot_separated_metrics(log_history, plots_dir):
    """Generate individual separated metric plots"""
    train_losses = [entry for entry in log_history if "loss" in entry and "eval_loss" not in entry]
    eval_metrics = [entry for entry in log_history if "eval_f1" in entry or "eval_loss" in entry]
    
    # 1. Train Loss Plot
    if train_losses:
        steps = [entry["step"] for entry in train_losses]
        losses = [entry["loss"] for entry in train_losses]
        plt.figure(figsize=(9, 5))
        plt.plot(steps, losses, label="Train Loss", color="#1f77b4", linewidth=2)
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("Training Loss Curve", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(plots_dir / "train_loss.png", dpi=150)
        plt.close()
        print(f"  ✅ Saved: {plots_dir / 'train_loss.png'}")

    # 2. Eval Loss Plot
    if eval_metrics and any("eval_loss" in e for e in eval_metrics):
        eval_steps = [e["step"] for e in eval_metrics if "eval_loss" in e]
        eval_losses = [e["eval_loss"] for e in eval_metrics if "eval_loss" in e]
        plt.figure(figsize=(9, 5))
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ff7f0e", linewidth=2, marker='o')
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("Validation Loss Curve", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(plots_dir / "eval_loss.png", dpi=150)
        plt.close()
        print(f"  ✅ Saved: {plots_dir / 'eval_loss.png'}")

    # 3. Combined Learning Curve (Train Loss vs Eval Loss)
    if train_losses and eval_metrics:
        steps = [entry["step"] for entry in train_losses]
        losses = [entry["loss"] for entry in train_losses]
        eval_steps = [e["step"] for e in eval_metrics if "eval_loss" in e]
        eval_losses = [e["eval_loss"] for e in eval_metrics if "eval_loss" in e]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label="Train Loss", color="#1f77b4", linewidth=2, alpha=0.8)
        if eval_steps:
            plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker='s')
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Loss", fontsize=11)
        plt.title("Learning Curve (Train vs Validation Loss)", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "learning_curve.png", dpi=150)
        plt.close()
        print(f"  ✅ Saved: {plots_dir / 'learning_curve.png'}")

    # 4. Validation F1 & Accuracy Plot
    if eval_metrics and any("eval_f1" in e for e in eval_metrics):
        eval_steps = [e["step"] for e in eval_metrics if "eval_f1" in e]
        f1_scores = [e["eval_f1"] for e in eval_metrics if "eval_f1" in e]
        accuracy_scores = [e.get("eval_accuracy", 0) for e in eval_metrics if "eval_f1" in e]

        plt.figure(figsize=(10, 6))
        plt.plot(eval_steps, f1_scores, label="Val F1-Macro", color="#2ca02c", linewidth=2.5, marker='o')
        plt.plot(eval_steps, accuracy_scores, label="Val Accuracy", color="#9467bd", linewidth=2, linestyle='--', marker='^')
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.title("Validation F1 & Accuracy Curves", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(plots_dir / "eval_f1_accuracy.png", dpi=150)
        plt.close()
        print(f"  ✅ Saved: {plots_dir / 'eval_f1_accuracy.png'}")

# Generate training/validation step plots
print("\n📊 Generating individual metric plots...")
plot_separated_metrics(log_history, PLOTS_DIR)

# ============================================================
# TEST SET EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("📊 ENHANCED EVALUATION ON TEST SET")
print("=" * 80)

eval_results = trainer.predict(test_dataset)
predictions_array = np.argmax(eval_results.predictions, axis=-1)
true_labels_array = eval_results.label_ids
probabilities = torch.nn.functional.softmax(
    torch.tensor(eval_results.predictions), dim=-1
).numpy()

# Calculate overall metrics
acc = accuracy_score(true_labels_array, predictions_array)
p, r, f1, _ = precision_recall_fscore_support(
    true_labels_array, predictions_array, average="macro", zero_division=0
)
p_w, r_w, f1_w, _ = precision_recall_fscore_support(
    true_labels_array, predictions_array, average="weighted", zero_division=0
)

metrics = {
    "accuracy": float(acc),
    "precision_macro": float(p),
    "recall_macro": float(r),
    "f1_macro": float(f1),
    "precision_weighted": float(p_w),
    "recall_weighted": float(r_w),
    "f1_weighted": float(f1_w),
}

with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nOverall Test Results:")
print(f"  Accuracy:          {acc:.4f}")
print(f"  Precision (macro): {p:.4f}")
print(f"  Recall (macro):    {r:.4f}")
print(f"  F1-Macro:          {f1:.4f}")
print(f"  F1-Weighted:       {f1_w:.4f}")

# Detailed per-class classification report
class_report = classification_report(
    true_labels_array,
    predictions_array,
    target_names=label_encoder.classes_,
    zero_division=0,
    output_dict=True
)

with open(RESULTS_DIR / "classification_report.json", "w") as f:
    json.dump(class_report, f, indent=2)

print("\nPer-Class Results:")
print(classification_report(
    true_labels_array,
    predictions_array,
    target_names=label_encoder.classes_,
    zero_division=0
))

# 5. Test Confusion Matrix (Counts & Percentages)
cm_test = confusion_matrix(true_labels_array, predictions_array)
cm_test_percent = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100

cm_test_df = pd.DataFrame(
    cm_test,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
cm_test_df.to_csv(RESULTS_DIR / "confusion_matrix_test.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            annot_kws={'size': 11}, ax=axes[0])
axes[0].set_title('Test Confusion Matrix - Counts', fontsize=12, fontweight="bold")
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)

sns.heatmap(cm_test_percent, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            annot_kws={'size': 11}, ax=axes[1])
axes[1].set_title('Test Confusion Matrix - Percentages (%)', fontsize=12, fontweight="bold")
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'test_confusion_matrix.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'test_confusion_matrix.png'}")

# ============================================================
# VALIDATION SET EVALUATION & CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 80)
print("📊 EVALUATION ON VALIDATION SET")
print("=" * 80)

val_results = trainer.predict(val_dataset)
val_predictions = np.argmax(val_results.predictions, axis=-1)
val_true = val_results.label_ids
val_probabilities = torch.nn.functional.softmax(
    torch.tensor(val_results.predictions), dim=-1
).numpy()

val_acc = accuracy_score(val_true, val_predictions)
val_p, val_r, val_f1, _ = precision_recall_fscore_support(
    val_true, val_predictions, average="macro", zero_division=0
)

# 6. Validation Confusion Matrix (Counts & Percentages)
cm_val = confusion_matrix(val_true, val_predictions)
cm_val_percent = cm_val.astype('float') / cm_val.sum(axis=1)[:, np.newaxis] * 100

cm_val_df = pd.DataFrame(
    cm_val,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)
cm_val_df.to_csv(RESULTS_DIR / "confusion_matrix_val.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            annot_kws={'size': 11}, ax=axes[0])
axes[0].set_title('Val Confusion Matrix - Counts', fontsize=12, fontweight="bold")
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('True', fontsize=11)

sns.heatmap(cm_val_percent, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            annot_kws={'size': 11}, ax=axes[1])
axes[1].set_title('Val Confusion Matrix - Percentages (%)', fontsize=12, fontweight="bold")
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('True', fontsize=11)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'val_confusion_matrix.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'val_confusion_matrix.png'}")

print(f"\nValidation Results:")
print(f"  Validation Accuracy: {val_acc:.4f}")
print(f"  Validation F1-Macro: {val_f1:.4f}")
print(f"  Test Accuracy:       {acc:.4f}")
print(f"  Test F1-Macro:       {f1:.4f}")
print(f"  Gap (Test - Val):    {acc - val_acc:.4f}")

# ============================================================
# RECOMMENDED ADDITIONAL METRICS PLOTS (ROC, PR, Per-Class F1, Confidence)
# ============================================================
print("\n📊 Generating recommended evaluation plots (ROC, PR Curves, Per-Class F1, Confidence)...")

# 7. ROC Curves (One-vs-Rest for each class)
plt.figure(figsize=(9, 6))
colors = ['#d62728', '#7f7f7f', '#2ca02c']
for i, class_name in enumerate(label_encoder.classes_):
    y_true_binary = (true_labels_array == i).astype(int)
    y_score = probabilities[:, i]
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
             label=f'ROC {class_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Chance (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'roc_curves.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'roc_curves.png'}")

# 8. Precision-Recall Curves (for each class)
plt.figure(figsize=(9, 6))
for i, class_name in enumerate(label_encoder.classes_):
    y_true_binary = (true_labels_array == i).astype(int)
    y_score = probabilities[:, i]
    prec_curve, rec_curve, _ = precision_recall_curve(y_true_binary, y_score)
    ap_score = average_precision_score(y_true_binary, y_score)
    plt.plot(rec_curve, prec_curve, color=colors[i % len(colors)], linewidth=2,
             label=f'PR {class_name} (AP = {ap_score:.4f})')

plt.xlabel('Recall', fontsize=11)
plt.ylabel('Precision', fontsize=11)
plt.title('Multi-Class Precision-Recall Curves', fontsize=13, fontweight="bold")
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'precision_recall_curves.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'precision_recall_curves.png'}")

# 9. Per-Class F1 Comparison Bar Chart (Validation vs Test)
val_p_class, val_r_class, val_f1_class, _ = precision_recall_fscore_support(
    val_true, val_predictions, average=None, zero_division=0
)
test_p_class, test_r_class, test_f1_class, _ = precision_recall_fscore_support(
    true_labels_array, predictions_array, average=None, zero_division=0
)

x = np.arange(len(label_encoder.classes_))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width/2, val_f1_class, width, label='Validation F1', color='#2ca02c', alpha=0.85)
plt.bar(x + width/2, test_f1_class, width, label='Test F1', color='#1f77b4', alpha=0.85)

for i in range(len(label_encoder.classes_)):
    plt.text(x[i] - width/2, val_f1_class[i] + 0.01, f"{val_f1_class[i]:.3f}", ha='center', fontsize=9)
    plt.text(x[i] + width/2, test_f1_class[i] + 0.01, f"{test_f1_class[i]:.3f}", ha='center', fontsize=9)

plt.xlabel('Sentiment Class', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('Per-Class F1 Score Comparison (Val vs Test)', fontsize=13, fontweight="bold")
plt.xticks(x, label_encoder.classes_, fontsize=10)
plt.ylim([0, 1.08])
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'per_class_f1_bar.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'per_class_f1_bar.png'}")

# 10. Confidence Analysis
confidence_scores = probabilities.max(axis=1)
predictions_correct = (predictions_array == true_labels_array)
correct_confidence = confidence_scores[predictions_correct]
incorrect_confidence = confidence_scores[~predictions_correct]

plt.figure(figsize=(9, 6))
plt.hist(correct_confidence, bins=20, alpha=0.7, label='Correct Predictions', color='#2ca02c')
plt.hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect Predictions', color='#d62728')
plt.xlabel('Confidence Score', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Prediction Confidence Distribution (Correct vs Incorrect)', fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'confidence_analysis.png', dpi=150)
plt.close()
print(f"  ✅ Saved: {PLOTS_DIR / 'confidence_analysis.png'}")

# Save detailed prediction results CSV
results_df = pd.DataFrame({
    'text': test_df['text'].values,
    'true_label': true_labels_array,
    'true_sentiment': label_encoder.inverse_transform(true_labels_array),
    'pred_label': predictions_array,
    'pred_sentiment': label_encoder.inverse_transform(predictions_array),
    'confidence': confidence_scores,
    'correct': predictions_correct,
    'prob_NEGATIVE': probabilities[:, 0] if NUM_LABELS >= 1 else 0,
    'prob_NEUTRAL': probabilities[:, 1] if NUM_LABELS >= 2 else 0,
    'prob_POSITIVE': probabilities[:, 2] if NUM_LABELS >= 3 else 0,
})
results_df.to_csv(RESULTS_DIR / "predictions_with_confidence.csv", index=False)
print(f"  ✅ Saved: {RESULTS_DIR / 'predictions_with_confidence.csv'}")

# Summary
summary = {
    "experiment_name": EXPERIMENT_DIR.name,
    "model_name": MODEL_NAME,
    "status": "completed",
    "completed_at": datetime.now().isoformat(),
    "metrics_test": metrics,
    "metrics_val": {
        "accuracy": float(val_acc),
        "precision_macro": float(val_p),
        "recall_macro": float(val_r),
        "f1_macro": float(val_f1),
    },
    "enhancements_applied": hyperparams["enhancements"],
    "gpu": "RTX 3050 6GB",
    "files": {
        "hyperparameters": str(RESULTS_DIR / "hyperparameters.json"),
        "metrics_test": str(RESULTS_DIR / "metrics.json"),
        "classification_report": str(RESULTS_DIR / "classification_report.json"),
        "confusion_matrix_test_csv": str(RESULTS_DIR / "confusion_matrix_test.csv"),
        "confusion_matrix_val_csv": str(RESULTS_DIR / "confusion_matrix_val.csv"),
        "predictions": str(RESULTS_DIR / "predictions_with_confidence.csv"),
        "plots": {
            "train_loss": str(PLOTS_DIR / "train_loss.png"),
            "eval_loss": str(PLOTS_DIR / "eval_loss.png"),
            "learning_curve": str(PLOTS_DIR / "learning_curve.png"),
            "eval_f1_accuracy": str(PLOTS_DIR / "eval_f1_accuracy.png"),
            "val_confusion_matrix": str(PLOTS_DIR / "val_confusion_matrix.png"),
            "test_confusion_matrix": str(PLOTS_DIR / "test_confusion_matrix.png"),
            "roc_curves": str(PLOTS_DIR / "roc_curves.png"),
            "precision_recall_curves": str(PLOTS_DIR / "precision_recall_curves.png"),
            "per_class_f1_bar": str(PLOTS_DIR / "per_class_f1_bar.png"),
            "confidence_analysis": str(PLOTS_DIR / "confidence_analysis.png"),
        },
        "best_model": str(BEST_MODEL_DIR),
    }
}

with open(EXPERIMENT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("✅ ENHANCED RTX 3050 EXPERIMENT COMPLETE")
print("=" * 80)
print(f"\n📁 Experiment directory: {EXPERIMENT_DIR}")
print(f"\n💻 GPU: RTX 3050 6GB")
print("\n🚀 Enhancements & Separate Plot Outputs Configured:")
for plot_key in summary["files"]["plots"].keys():
    print(f"  📊 {plot_key}.png")

# Final memory cleanup
clear_memory()