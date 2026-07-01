import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
DATA_FILE = "fine_tune_data/global_data_merged.csv"
VALID_LANG = {"msa", "egy", "lev", "glf", "mgr"}

# Load some data
df = pd.read_csv(DATA_FILE)
df["text"] = df["text"].astype(str)
df["language"] = df["language"].astype(str).str.lower().str.strip()
df = df[df["language"].isin(VALID_LANG)].copy()

# Print label maps to ensure correct parsing
from fine_tune_arabic_topic import parse_topic, ID2LABEL, LABEL2ID, NUM_LABELS
df["label"] = df["topic"].apply(parse_topic).astype(int)
df = df[df["label"] != -1].copy()

print("Number of unique labels in subset:", df["label"].nunique())
print("Label range:", df["label"].min(), "to", df["label"].max())

# Tokenize a sample
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
sample_texts = df["text"].head(8).tolist()
sample_labels = df["label"].head(8).tolist()

inputs = tok(sample_texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
inputs["labels"] = torch.tensor(sample_labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# Test forward pass in FP32
print("--- FP32 Forward Pass ---")
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    print("Loss:", outputs.loss.item())
    print("Logits (first 2 rows):", outputs.logits[:2])
    print("Logits finite:", torch.isfinite(outputs.logits).all().item())

# Test forward pass in BF16
print("--- BF16 Forward Pass ---")
model_bf16 = model.to(dtype=torch.bfloat16, device="cuda")
inputs_bf16 = {k: v.to(device="cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
with torch.no_grad():
    outputs_bf16 = model_bf16(**inputs_bf16)
    print("Loss (BF16):", outputs_bf16.loss.item())
    print("Logits (BF16, first 2 rows):", outputs_bf16.logits[:2])
    print("Logits finite (BF16):", torch.isfinite(outputs_bf16.logits).all().item())
