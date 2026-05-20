import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List

# Model name for English to French translation
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

# Auto-detect CUDA and move model to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once at module level to avoid repeated loading
TOKENIZER = MarianTokenizer.from_pretrained(MODEL_NAME)
MODEL = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

def _translate_chunk(text: str) -> str:
    """Helper function to translate a small chunk of text (max 512 tokens)."""
    if not text.strip():
        return text
        
    inputs = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        translated_tokens = MODEL.generate(**inputs)
    
    return TOKENIZER.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_en_to_fr(text: str) -> str:
    """
    Translates a single English text to French.
    Handles long articles by splitting into paragraphs and sentences.
    """
    if not text or not text.strip():
        return text

    # Split into paragraphs
    paragraphs = text.split("\n")
    translated_paragraphs = []

    for p in paragraphs:
        if not p.strip():
            translated_paragraphs.append("")
            continue
            
        # Check token count for the paragraph
        tokens = TOKENIZER(p, return_tensors="pt")["input_ids"]
        if tokens.shape[1] <= 512:
            # Paragraph is small enough
            translated_paragraphs.append(_translate_chunk(p))
        else:
            # Paragraph exceeds 512 tokens, split into sentences
            # Helsinki models work better with full sentences, so we split on ". "
            sentences = p.split(". ")
            translated_sentences = []
            for s in sentences:
                if not s.strip():
                    continue
                # Add back the period if it was a real sentence end
                text_to_translate = s if s.endswith(".") else s + "."
                translated_sentences.append(_translate_chunk(text_to_translate))
            translated_paragraphs.append(" ".join(translated_sentences))

    return "\n".join(translated_paragraphs)

def translate_batch(texts: List[str]) -> List[str]:
    """
    Translates a list of articles. 
    Processes articles individually to respect the paragraph/sentence splitting logic.
    """
    return [translate_en_to_fr(t) for t in texts]

if __name__ == "__main__":
    # Quick test
    test_text = "This is a test sentence. This is another one."
    print(f"EN: {test_text}")
    print(f"FR: {translate_en_to_fr(test_text)}")
