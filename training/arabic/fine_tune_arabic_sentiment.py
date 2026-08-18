import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed,
)
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "global_data_libelised.csv"
OUT_DIR = PROJECT_ROOT / "experiments"
TASK = "arabic_sentiment"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"

VALID_LANG = {"ar", "da"}

# =======================
# EXACT PARAMETERS FROM BEST VERSION (87.7% ACCURACY)
# =======================
# DATA SPLIT RATIOS (80/10/10)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MAX_LEN = 512
STRIDE = 256  # KEEP ORIGINAL

EPOCHS = 20  # EXACT
LR = 1e-5  # EXACT
BATCH_SIZE = 4  # EXACT
GRAD_ACCUM = 4  # EXACT
EVAL_STEPS = 4000  # EXACT
SEED = 42  # EXACT

WARMUP_RATIO = 0.2  # EXACT
WEIGHT_DECAY = 0.03  # EXACT
LABEL_SMOOTHING = 0.15  # EXACT
MAX_GRAD_NORM = 0.5  # EXACT

FOCAL_ALPHA = 0.45  # EXACT
FOCAL_GAMMA = 2.5  # EXACT

EARLY_STOPPING_PATIENCE = 20  # EXACT
EARLY_STOPPING_THRESHOLD = 0.005  # EXACT

AUGMENT_FACTOR = 1  # EXACT

DATA_SUBSET_RATIO = 1.0

# =======================
# RUN CONTROL
# =======================
RESUME_LAST_CHECKPOINT = False  # Set to True to resume training from the latest checkpoint


SENT2ID = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
ID2SENT = {v: k for k, v in SENT2ID.items()}
NUM_LABELS = 3
ID2LABEL = {i: ID2SENT[i] for i in range(NUM_LABELS)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

def parse_sentiment(x):
    if pd.isna(x):
        return -1
    s = str(x).strip()
    if s.isdigit():
        i = int(s)
        return i if i in (0, 1, 2) else -1
    return SENT2ID.get(s.upper(), -1)

def compute_class_weights(df, epoch=0):
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)
    n_classes = len(class_counts)
    weights = total / (n_classes * class_counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 3.0)
    
    if epoch > 5:
        weights[1] = weights[1] * 0.8
        weights[2] = weights[2] * 1.2
        weights = np.clip(weights, 0.5, 3.5)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\nClass Weights (epoch {epoch}):")
    for i, w in enumerate(weights_tensor):
        label = ID2LABEL[i]
        count = class_counts[i]
        pct = (count / total) * 100
        print(f"  {label}: weight={w:.3f} (count={count:,}, {pct:.1f}%)")
    return weights_tensor

def analyze_class_distribution(df, name="Dataset"):
    print(f"\n{name} Class Distribution:")
    counts = df['label'].value_counts().sort_index()
    percentages = df['label'].value_counts(normalize=True).sort_index()
    total = len(df)
    for label in sorted(df['label'].unique()):
        label_name = ID2LABEL[label]
        count = counts.get(label, 0)
        pct = percentages.get(label, 0) * 100
        bar = "█" * int(pct / 2)
        warning = ""
        if label_name == "POSITIVE" and pct < 15:
            warning = " UNDERREPRESENTED"
        elif label_name == "NEUTRAL" and pct > 35:
            warning = " MAJORITY"
        print(f"  {label_name}: {count:>8,} ({pct:>5.1f}%) {bar} {warning}")
    return counts, percentages

def augment_positive_class(df, augment_factor=0.5):
    positive_df = df[df['label'] == 2].copy()
    if len(positive_df) == 0:
        return df
    
    n_augment = int(len(positive_df) * augment_factor)
    if n_augment == 0:
        return df
    
    augmented_parts = []
    intensifiers = [" جداً", " حقاً", " فعلاً", " للغاية", " تماماً"]
    
    for _ in range(n_augment):
        row = positive_df.sample(1, random_state=SEED)
        text = row['text'].values[0]
        text = text + np.random.choice(intensifiers)
        row['text'] = text
        augmented_parts.append(row)
    
    result = pd.concat([df] + augmented_parts, ignore_index=True)
    print(f"  POSITIVE class augmented: +{len(augmented_parts):,} samples")
    return result

def split_percentage(df_group, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    temp_ratio = val_ratio + test_ratio
    try:
        train_df, temp_df = train_test_split(
            df_group, 
            test_size=temp_ratio, 
            random_state=seed, 
            stratify=df_group["label"]
        )
    except Exception:
        temp_df = df_group.sample(frac=temp_ratio, random_state=seed)
        train_df = df_group.drop(temp_df.index)
        
    eval_test_split_ratio = test_ratio / temp_ratio
    try:
        eval_df, test_df = train_test_split(
            temp_df, 
            test_size=eval_test_split_ratio, 
            random_state=seed, 
            stratify=temp_df["label"]
        )
    except Exception:
        test_df = temp_df.sample(frac=eval_test_split_ratio, random_state=seed)
        eval_df = temp_df.drop(test_df.index)
        
    return train_df, eval_df, test_df

def check_text_lengths(df, tokenizer, max_len=512, sample_size=5000):
    print("\n📏 Checking text lengths...")
    sample_size = min(sample_size, len(df))
    lengths = []
    for text in df['text'].head(sample_size):
        tokens = tokenizer.encode(text, truncation=False)
        lengths.append(len(tokens))
    lengths = np.array(lengths)
    exceed = (lengths > max_len).sum()
    pct_exceed = (exceed / len(lengths)) * 100
    max_len_actual = lengths.max()
    mean_len = lengths.mean()
    p95 = np.percentile(lengths, 95)
    print(f"  Analyzed {len(lengths):,} random samples")
    print(f"  Mean length: {mean_len:.0f} tokens")
    print(f"  95th percentile: {p95:.0f} tokens")
    print(f"  Max length: {max_len_actual} tokens")
    print(f"  Exceeding {max_len}: {exceed} samples ({pct_exceed:.1f}%)")
    return lengths

def process_long_document_sliding_window(text, tokenizer, max_len=512, stride=256):
    tokens = tokenizer(
        text, 
        truncation=False,
        return_tensors='pt',
        return_overflowing_tokens=True,
        stride=stride,
        max_length=max_len,
        padding=False,
    )
    
    input_ids = tokens['input_ids']
    attention_mask = tokens.get('attention_mask', None)
    
    if len(input_ids) <= 1:
        tokenized = tokenizer(text, truncation=True, max_length=max_len, padding='max_length')
        return {'chunks': [tokenized], 'num_chunks': 1, 'is_long': False}
    
    chunks = []
    for i in range(len(input_ids)):
        chunk_ids = input_ids[i]
        chunk_mask = attention_mask[i] if attention_mask is not None else torch.ones_like(chunk_ids)
        
        if len(chunk_ids) < max_len:
            pad_len = max_len - len(chunk_ids)
            chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
            chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])
        
        chunks.append({
            'input_ids': chunk_ids.numpy().tolist(),
            'attention_mask': chunk_mask.numpy().tolist(),
        })
    
    return {'chunks': chunks, 'num_chunks': len(chunks), 'is_long': True}

def create_sliding_window_dataset(df, tokenizer, max_len=512, stride=256):
    all_data = []
    total_chunks = 0
    
    for idx, row in df.iterrows():
        text = row['text']
        label = row['label']
        
        result = process_long_document_sliding_window(text, tokenizer, max_len, stride)
        
        for chunk in result['chunks']:
            all_data.append({
                'input_ids': torch.tensor(chunk['input_ids']),
                'attention_mask': torch.tensor(chunk['attention_mask']),
                'label': label,
                'original_idx': idx,
                'chunk_idx': 0,
                'num_chunks': result['num_chunks'],
                'is_long': result['is_long'],
            })
            total_chunks += 1
    
    print(f"  Created {len(all_data):,} chunks from {len(df):,} documents")
    print(f"  Average chunks per document: {total_chunks/len(df):.2f}")
    
    return all_data

def collate_with_chunks(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    num_chunks = torch.tensor([item['num_chunks'] for item in batch])
    is_long = torch.tensor([1 if item['is_long'] else 0 for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'num_chunks': num_chunks,
        'is_long': is_long,
    }

class SlidingWindowDatasetWrapper:
    def __init__(self, df, tokenizer, max_len=512, stride=256):
        self.data = create_sliding_window_dataset(df, tokenizer, max_len, stride)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 20, threshold: float = 0.005):
        self.patience = patience
        self.threshold = threshold
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
            print(f"[EarlyStopping] Initial best eval_f1_macro: {self.best_metric:.4f}")
            return

        if current_metric > self.best_metric + self.threshold:
            print(
                f"[EarlyStopping] Improved: {self.best_metric:.4f} → {current_metric:.4f} "
                f"at step {state.global_step} (patience reset)"
            )
            self.best_metric = current_metric
            self.best_step = state.global_step
            self.wait = 0
        else:
            self.wait += 1
            print(
                f"[EarlyStopping] No improvement ({current_metric:.4f} vs best {self.best_metric:.4f}). "
                f"Patience: {self.wait}/{self.patience}"
            )

        if self.wait >= self.patience:
            self.stopped_epoch = state.epoch
            control.should_training_stop = True
            print(f"\n[EarlyStopping] Triggered at step {state.global_step} "
                  f"(epoch {state.epoch:.2f})")
            print(f"[EarlyStopping] Best eval_f1_macro: {self.best_metric:.4f} "
                  f"at step {self.best_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.stopped_epoch > 0:
            print(f"\n[EarlyStopping] Training stopped early at epoch "
                  f"{self.stopped_epoch:.2f}")

class FocalLossTrainer(Trainer):
    def __init__(self, class_weights=None, alpha=0.45, gamma=2.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
        self.training_losses = []
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        _EXTRA_KEYS = {'num_chunks', 'is_long', 'original_idx', 'chunk_idx'}
        model_inputs = {k: v for k, v in inputs.items() if k not in _EXTRA_KEYS}
        labels = model_inputs.get("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        ce = ce_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        pt = torch.exp(-ce)
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
        metrics = super().evaluate(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        if metric_key_prefix == "eval" and "eval_f1_macro" in metrics:
            self.state.best_metric = metrics["eval_f1_macro"]
            if self.state.best_model_checkpoint is None:
                self.state.best_model_checkpoint = self.args.output_dir
        return metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
    report = classification_report(labels, preds, target_names=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], output_dict=True, zero_division=0)
    for label in ['NEGATIVE', 'NEUTRAL', 'POSITIVE']:
        metrics[f"f1_{label}"] = report[label]['f1-score']
        metrics[f"recall_{label}"] = report[label]['recall']
        metrics[f"precision_{label}"] = report[label]['precision']
    return metrics

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], yticklabels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'], annot_kws={'size': 12})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {split_name}\nNEG={class_acc[0]:.1%}, NEU={class_acc[1]:.1%}, POS={class_acc[2]:.1%}', fontsize=12)
    fig.tight_layout()
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / f'confusion_matrix_{split_name}.png', dpi=150)
    plt.close(fig)
    return class_acc

def save_predictions(trainer, test_dataset, run_dir, split_name):
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    results_df = pd.DataFrame({
        'prediction': preds,
        'label': labels,
        'confidence': np.max(predictions.predictions, axis=-1),
        'predicted_class': [ID2LABEL[p] for p in preds],
        'true_class': [ID2LABEL[l] for l in labels],
    })
    results_df['correct'] = (results_df['prediction'] == results_df['label']).astype(int)
    results_df.to_csv(run_dir / f'predictions_{split_name}.csv', index=False)

def save_run_artifacts(trainer, run_dir, eval_msa_ds, eval_dial_ds):
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train_logs, eval_logs = _extract_logs(trainer.state.log_history)
    res_msa = trainer.evaluate(eval_msa_ds, metric_key_prefix="msa")
    res_dial = trainer.evaluate(eval_dial_ds, metric_key_prefix="dialect")
    eval_results = {"msa": res_msa, "dialect": res_dial}
    (run_dir / "eval_results.json").write_text(json.dumps(eval_results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print("\nMSA:")
    for key in ['msa_accuracy', 'msa_f1_macro', 'msa_f1_NEGATIVE', 'msa_f1_NEUTRAL', 'msa_f1_POSITIVE']:
        if key in res_msa:
            print(f"  {key}: {res_msa[key]:.4f}")
    print("\nDialect:")
    for key in ['dialect_accuracy', 'dialect_f1_macro', 'dialect_f1_NEGATIVE', 'dialect_f1_NEUTRAL', 'dialect_f1_POSITIVE']:
        if key in res_dial:
            print(f"  {key}: {res_dial[key]:.4f}")
    
    plot_confusion_matrix(trainer, eval_msa_ds, run_dir, 'msa')
    plot_confusion_matrix(trainer, eval_dial_ds, run_dir, 'dialect')
    
    def _steps(logs, key): return [e["step"] for e in logs if key in e]
    def _vals(logs, key): return [e[key] for e in logs if key in e]
    STYLE = dict(linewidth=1.8)
    
    if train_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"), color="#e07b39", label="Train loss", **STYLE)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "train_loss.png", dpi=150)
        plt.close(fig)
    
    if eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"), color="#4a90d9", label="Eval loss", **STYLE)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
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
            ax1.set_title("Eval F1 Macro & Accuracy")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "eval_f1_accuracy.png", dpi=150)
            plt.close(fig)
    
    if train_logs and eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(_steps(train_logs, "loss"), _vals(train_logs, "loss"), color="#e07b39", label="Train loss", **STYLE)
        ax.plot(_steps(eval_logs, "eval_loss"), _vals(eval_logs, "eval_loss"), color="#4a90d9", label="Eval loss", **STYLE)
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
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    base_experiment_dir = OUT_DIR / TASK
    latest_checkpoint = None
    latest_run_dir = None
    
    if RESUME_LAST_CHECKPOINT and base_experiment_dir.exists():
        run_dirs = [d for d in base_experiment_dir.iterdir() if d.is_dir()]
        run_dirs.sort()
        for r_dir in reversed(run_dirs):
            checkpoints = [d for d in r_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            # Filter to checkpoints that contain trainer_state.json and a model file (safetensors or bin)
            valid_checkpoints = []
            for cp in checkpoints:
                if (cp / "trainer_state.json").exists() and (
                    (cp / "pytorch_model.bin").exists() or (cp / "model.safetensors").exists()
                ):
                    valid_checkpoints.append(cp)
            if valid_checkpoints:
                def get_step(d):
                    try:
                        return int(d.name.split("-")[-1])
                    except ValueError:
                        return -1
                valid_checkpoints.sort(key=get_step)
                latest_checkpoint = valid_checkpoints[-1]
                latest_run_dir = r_dir
                break
                
    if latest_checkpoint is not None:
        run_ts = latest_run_dir.name
        run_dir = latest_run_dir
        print(f"\n🔄 Found existing checkpoint: {latest_checkpoint.name} in {latest_run_dir.name}")
        print(f"Resuming training from this checkpoint...")
    else:
        run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = OUT_DIR / TASK / run_ts
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🆕 No existing checkpoint found. Starting new training run...")
    
    print(f"\n{'='*60}")
    print(f"🚀 FINAL TRAINING RUN - USING PROVEN PARAMETERS")
    print(f"Run: {run_ts}")
    print(f"Using EXACT hyperparameters from best run (87.7% accuracy)")
    print(f"{'='*60}\n")
    print(f"Loading data from: {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print(f"Initial rows: {len(df):,}")
    
    df["text"] = df["text"].astype(str)
    df["language"] = df["language"].astype(str).str.lower().str.strip()
    df = df[df["language"].isin(VALID_LANG)].copy()
    df["label"] = df["sentiment"].apply(parse_sentiment).astype(int)
    df = df[df["label"] != -1].copy()
    print(f"\nAfter filtering: {len(df):,} valid samples")
    
    df["variant"] = np.where(df["language"] == "da", "dialect", "msa")
    msa_df = df[df["variant"] == "msa"].copy()
    dial_df = df[df["variant"] == "dialect"].copy()
    
    analyze_class_distribution(df, "FULL DATASET")
    analyze_class_distribution(msa_df, "MSA Data")
    analyze_class_distribution(dial_df, "Dialect Data")
    
    # Split MSA and Dialect datasets into 80% train, 10% eval, 10% test
    train_msa, eval_msa, test_msa = split_percentage(msa_df, VAL_RATIO, TEST_RATIO, SEED)
    train_dial, eval_dial, test_dial = split_percentage(dial_df, VAL_RATIO, TEST_RATIO, SEED)
    
    print(f"\nData Split Summary:")
    print(f"  MSA - Train: {len(train_msa):,}, Eval: {len(eval_msa):,}, Test: {len(test_msa):,}")
    print(f"  Dialect - Train: {len(train_dial):,}, Eval: {len(eval_dial):,}, Test: {len(test_dial):,}")
    print(f"  Total Train: {len(train_msa) + len(train_dial):,}")
    print(f"  Total Eval: {len(eval_msa) + len(eval_dial):,}")
    print(f"  Total Test: {len(test_msa) + len(test_dial):,}")
    
    train_df = pd.concat([train_msa, train_dial], ignore_index=True)
    eval_mix = pd.concat([eval_msa, eval_dial], ignore_index=True)
    test_mix = pd.concat([test_msa, test_dial], ignore_index=True)
    
    print("\n📈 Augmenting POSITIVE class...")
    train_df = augment_positive_class(train_df, augment_factor=AUGMENT_FACTOR)
    
    class_weights = compute_class_weights(train_df, epoch=0)
    
    print(f"\nFinal Training Data:")
    print(f"  Total: {len(train_df):,}")
    print(f"  Dialect ratio: {round((train_df['variant'] == 'dialect').mean(), 3)}")
    analyze_class_distribution(train_df, "TRAINING DATA")
    
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    check_text_lengths(train_df, tok, MAX_LEN, sample_size=5000)
    
    print(f"\nPreparing datasets with SLIDING WINDOW...")
    print(f"  Max length: {MAX_LEN}")
    print(f"  Stride: {STRIDE}")
    
    train_sliding = SlidingWindowDatasetWrapper(train_df, tok, max_len=MAX_LEN, stride=STRIDE)
    eval_msa_sliding = SlidingWindowDatasetWrapper(eval_msa, tok, max_len=MAX_LEN, stride=STRIDE)
    eval_dial_sliding = SlidingWindowDatasetWrapper(eval_dial, tok, max_len=MAX_LEN, stride=STRIDE)
    eval_mix_sliding = SlidingWindowDatasetWrapper(eval_mix, tok, max_len=MAX_LEN, stride=STRIDE)
    
    print(f"\nDataset sizes (chunks):")
    print(f"  Training: {len(train_sliding):,}")
    print(f"  Eval MSA: {len(eval_msa_sliding):,}")
    print(f"  Eval Dialect: {len(eval_dial_sliding):,}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()
    
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION (EXACTLY FROM BEST RUN)")
    print(f"{'='*60}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Focal Alpha: {FOCAL_ALPHA}")
    print(f"  Focal Gamma: {FOCAL_GAMMA}")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Max Grad Norm: {MAX_GRAD_NORM}")
    print(f"  Sliding window stride: {STRIDE}")
    print(f"  Augmentation factor: {AUGMENT_FACTOR}")
    print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Eval Steps: {EVAL_STEPS}")
    print(f"{'='*60}\n")
    
    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(4, BATCH_SIZE * 2),
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        logging_steps=50,
        max_grad_norm=MAX_GRAD_NORM,
        label_smoothing_factor=LABEL_SMOOTHING,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=1000,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_only_model=True,
        optim="adamw_torch",
    )
    
    hyperparams = {
        "run_timestamp": run_ts,
        "model_name": MODEL_NAME,
        "task": TASK,
        "data_file": str(DATA_FILE),
        "data_subset_ratio": DATA_SUBSET_RATIO,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": effective_batch_size,
        "max_len": MAX_LEN,
        "stride": STRIDE,
        "eval_steps": EVAL_STEPS,
        "seed": SEED,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing_factor": LABEL_SMOOTHING,
        "max_grad_norm": MAX_GRAD_NORM,
        "lr_scheduler_type": "cosine_with_restarts",
        "class_weights": class_weights.tolist(),
        "use_focal_loss": True,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "use_sliding_window": True,
        "sliding_window_stride": STRIDE,
        "train_size": len(train_df),
        "eval_msa_size": len(eval_msa),
        "eval_dialect_size": len(eval_dial),
        "test_msa_size": len(test_msa),
        "test_dialect_size": len(test_dial),
        "class_distribution": train_df['label'].value_counts().to_dict(),
        "dialect_ratio": float((train_df['variant'] == 'dialect').mean()),
        "positive_augmentation": True,
        "augmentation_factor": AUGMENT_FACTOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
        "valid_languages": list(VALID_LANG),
    }
    (run_dir / "hyperparameters.json").write_text(json.dumps(hyperparams, ensure_ascii=False, indent=2), encoding="utf-8")
    
    trainer = FocalLossTrainer(
        class_weights=class_weights,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        model=model,
        args=args,
        train_dataset=train_sliding,
        eval_dataset=eval_mix_sliding,
        processing_class=tok,
        data_collator=collate_with_chunks,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(patience=EARLY_STOPPING_PATIENCE, threshold=EARLY_STOPPING_THRESHOLD)],
    )
    
    print(f"\n{'='*60}")
    print("Starting Training with PROVEN parameters...")
    print(f"{'='*60}\n")
    resume_path = str(latest_checkpoint) if latest_checkpoint is not None else None
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(run_dir))
    tok.save_pretrained(str(run_dir))
    
    print("\nEvaluating on held-out sets...")
    save_run_artifacts(trainer, run_dir, eval_msa_sliding, eval_dial_sliding)
    save_predictions(trainer, eval_msa_sliding, run_dir, "eval_msa")
    save_predictions(trainer, eval_dial_sliding, run_dir, "eval_dialect")
    
    print("\nEvaluating on test sets...")
    test_msa_sliding = SlidingWindowDatasetWrapper(test_msa, tok, max_len=MAX_LEN, stride=STRIDE)
    test_dial_sliding = SlidingWindowDatasetWrapper(test_dial, tok, max_len=MAX_LEN, stride=STRIDE)
    test_mix_sliding = SlidingWindowDatasetWrapper(test_mix, tok, max_len=MAX_LEN, stride=STRIDE)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    
    test_msa_results = trainer.evaluate(test_msa_sliding, metric_key_prefix="test_msa")
    test_dial_results = trainer.evaluate(test_dial_sliding, metric_key_prefix="test_dialect")
    test_mix_results = trainer.evaluate(test_mix_sliding, metric_key_prefix="test_mix")
    
    print("\nTest MSA:")
    for key in ['test_msa_accuracy', 'test_msa_f1_macro', 'test_msa_f1_NEGATIVE', 'test_msa_f1_NEUTRAL', 'test_msa_f1_POSITIVE']:
        if key in test_msa_results:
            print(f"  {key}: {test_msa_results[key]:.4f}")
    
    print("\nTest Dialect:")
    for key in ['test_dialect_accuracy', 'test_dialect_f1_macro', 'test_dialect_f1_NEGATIVE', 'test_dialect_f1_NEUTRAL', 'test_dialect_f1_POSITIVE']:
        if key in test_dial_results:
            print(f"  {key}: {test_dial_results[key]:.4f}")

    print("\nTest Combined:")
    for key in ['test_mix_accuracy', 'test_mix_f1_macro', 'test_mix_f1_NEGATIVE', 'test_mix_f1_NEUTRAL', 'test_mix_f1_POSITIVE']:
        if key in test_mix_results:
            print(f"  {key}: {test_mix_results[key]:.4f}")
    
    test_results = {
        "msa": test_msa_results, 
        "dialect": test_dial_results,
        "mix": test_mix_results
    }
    (run_dir / "test_results.json").write_text(json.dumps(test_results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    plot_confusion_matrix(trainer, test_msa_sliding, run_dir, 'test_msa')
    plot_confusion_matrix(trainer, test_dial_sliding, run_dir, 'test_dialect')
    plot_confusion_matrix(trainer, test_mix_sliding, run_dir, 'test_mix')
    
    save_predictions(trainer, test_msa_sliding, run_dir, "test_msa")
    save_predictions(trainer, test_dial_sliding, run_dir, "test_dialect")
    save_predictions(trainer, test_mix_sliding, run_dir, "test_mix")
    
    print(f"\n{'='*60}")
    print("✅ FINAL TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Best model saved at: {run_dir}")
    print(f"\nData split summary:")
    print(f"  Training: {len(train_df):,} samples")
    print(f"  Validation: {len(eval_mix):,} samples")
    print(f"  Test: {len(test_msa) + len(test_dial):,} samples")
    print(f"\nParameters (EXACTLY from best run):")
    print(f"  ✓ Epochs: {EPOCHS}")
    print(f"  ✓ Learning rate: {LR}")
    print(f"  ✓ Focal Alpha: {FOCAL_ALPHA}")
    print(f"  ✓ Focal Gamma: {FOCAL_GAMMA}")
    print(f"  ✓ Sliding window stride: {STRIDE}")
    print(f"  ✓ Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  ✓ Weight decay: {WEIGHT_DECAY}")
    print(f"  ✓ Max grad norm: {MAX_GRAD_NORM}")
    print(f"  ✓ Eval steps: {EVAL_STEPS}")
    print(f"  ✓ Augmentation factor: {AUGMENT_FACTOR}")
    print(f"  ✓ Label smoothing: {LABEL_SMOOTHING}")
    print(f"{'='*60}\n")
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

if __name__ == "__main__":
    main()