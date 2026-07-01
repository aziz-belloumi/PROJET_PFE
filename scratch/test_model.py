import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
NUM_LABELS = 18

print("Loading tokenizer and model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\nModel class:", model.__class__.__name__)
print("Classifier:", model.classifier)

# Test with a dummy input
text = "اليوم:الريكسوس، تجمع متظاهري (لا للتمديد) أمام مقر المؤتمر"
inputs = tok(text, return_tensors="pt").to(device)
labels = torch.tensor([5]).to(device) # dummy label

print("\nInputs keys:", list(inputs.keys()))
outputs = model(**inputs, labels=labels)

print("Loss:", outputs.loss.item())
print("Logits shape:", outputs.logits.shape)
print("Logits values:", outputs.logits[0].detach().cpu().numpy())
