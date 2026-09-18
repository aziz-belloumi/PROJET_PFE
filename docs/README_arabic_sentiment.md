# Arabic Sentiment Analysis — Fine-Tuning Documentation

This document provides a comprehensive technical breakdown of [`fine_tune_arabic_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_sentiment.py). It covers the architectural rationale, mathematical formulations of loss functions, hyperparameter selection criteria, data preprocessing with sliding-window tokenization, class imbalance mitigation, and complete code walkthrough.

---

## 1. Overview & Objective

* **Target Task**: 3-Class Sentiment Classification (`NEGATIVE`: 0, `NEUTRAL`: 1, `POSITIVE`: 2).
* **Target Script**: [`training/arabic/fine_tune_arabic_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_sentiment.py)
* **Base Model**: [`CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment)
* **Dataset**: Arabic subset (`language in {'ar', 'da'}`) from `data/global_data_libelised.csv`.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 (4GB–6GB VRAM) with mixed precision (`fp16`) and gradient checkpointing.
* **Benchmark Target**: Proven configuration achieving **87.7% Accuracy / High Macro-F1**.

---

## 2. Theoretical & Architectural Rationale

### 2.1 Why CamelBERT-Mix Sentiment?
* **Linguistic Domain**: Arabic possesses rich morphology, dialectal divergence, and orthographic variability. General multilingual models (such as mBERT or XLM-RoBERTa) distribute parameter capacity across 100+ languages, leading to suboptimal tokenization and morphological representations for Arabic.
* **Pre-training Adaptation**: CamelBERT-Mix (Inoue et al., 2021) was pre-trained on Modern Standard Arabic (MSA), classical Arabic, and dialectal Arabic (da).
* **Warm-Start Sentiment Checkpoint**: The `camelbert-mix-sentiment` checkpoint comes with pre-adapted polarity representations, reducing fine-tuning convergence time by 2–3 epochs compared to training a raw BERT from scratch.

### 2.2 Handling Long Documents: Sliding Window Tokenization
* **Problem**: Arabic news articles frequently exceed the standard 512-token BERT limit. Truncating texts discards the middle and conclusion paragraphs where dominant sentiments often emerge.
* **Solution**: A sliding window mechanism encodes documents with a maximum sequence length $L=512$ and a stride $S=256$ (50% overlap).
* **Mechanism**:
  $$\text{Window}_k = \text{tokens}[k \cdot S : k \cdot S + L], \quad k \in \{0, 1, \dots, \lfloor (N - L)/S \rfloor\}$$
  Each chunk inherits the ground-truth document label. This retains complete text context and doubles training sample density for long articles.

---

## 3. Mathematical Formulations & Loss Functions

### 3.1 Weighted Focal Loss with Dynamic Alpha
To address class imbalance (where `NEUTRAL` and `NEGATIVE` dominate over `POSITIVE`), a customized Focal Loss is implemented:

$$\text{FL}(p_t) = -\alpha \cdot w_c \cdot (1 - p_t)^\gamma \log(p_t + \epsilon)$$

Where:
* $p_t = \exp(-\text{CE}(z, y))$ is the model's estimated probability for the true class $y$.
* $\gamma = 2.5$ is the focusing parameter. When an example is well-classified ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, downweighting easy samples and concentrating gradient updates on hard samples.
* $\alpha = 0.45$ scales the overall focal loss magnitude.
* $\epsilon = 10^{-8}$ prevents numerical instability during $\log$ evaluation.
* $w_c$ is the normalized class weight for class $c$.

### 3.2 Dynamic Class Weights
Computed dynamically from the empirical training distribution:

$$w_c = \text{clip}\left(\frac{N_{\text{total}}}{C \cdot N_c} \cdot \frac{1}{\bar{w}}, 0.5, 3.0\right)$$

* For epochs $> 5$, a dynamic curriculum shift is applied: $w_{\text{NEUTRAL}} \leftarrow w_{\text{NEUTRAL}} \times 0.8$ and $w_{\text{POSITIVE}} \leftarrow w_{\text{POSITIVE}} \times 1.2$, clipped to $[0.5, 3.5]$.

### 3.3 Label Smoothing
Label smoothing ($\epsilon_{\text{smooth}} = 0.15$) regularizes the cross-entropy targets:

$$y_k^{\text{smooth}} = (1 - \epsilon_{\text{smooth}}) \cdot \mathbb{I}(y = k) + \frac{\epsilon_{\text{smooth}}}{K}$$

This prevents overconfidence on majority classes and reduces model calibration error.

---

## 4. Hyperparameter Choices & Empirical Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `512` | Full Transformer receptive field capacity. |
| `STRIDE` | `256` | 50% token overlap preserving boundary context between consecutive chunks. |
| `EPOCHS` | `20` | Sufficient convergence window when combined with early stopping. |
| `LR` | `1e-5` | Small learning rate for fine-tuning a pre-warmed sentiment backbone without catastrophic forgetting. |
| `BATCH_SIZE` | `4` | Maximum physical per-device batch size that fits in 4GB–6GB VRAM at 512 tokens with fp16. |
| `GRAD_ACCUM` | `4` | Simulates an effective batch size of $4 \times 4 = 16$. |
| `WARMUP_RATIO` | `0.2` | 20% warmup steps with a hard floor of 1000 steps to stabilize classification head gradients. |
| `WEIGHT_DECAY` | `0.03` | $L_2$ regularization preventing overfitting on dialectal slang. |
| `LABEL_SMOOTHING` | `0.15` | Softens target distributions to prevent majority-class overconfidence. |
| `MAX_GRAD_NORM` | `0.5` | Gradient clipping to prevent gradient explosion on long text chunks. |
| `FOCAL_ALPHA` | `0.45` | Overall scaling factor for focal loss. |
| `FOCAL_GAMMA` | `2.5` | Strong focusing exponent prioritizing rare/hard positive sentiment instances. |
| `EARLY_STOPPING_PATIENCE` | `20` | Tolerates evaluation plateaus on noisy dialectal validation samples. |
| `EARLY_STOPPING_THRESHOLD` | `0.005` | Minimum delta in `eval_f1_macro` required to reset early stopping wait counter. |
| `AUGMENT_FACTOR` | `1.0` | 100% synthetic augmentation of minority POSITIVE class. |
| `EVAL_STEPS` | `4000` | Step interval for validation and checkpoint evaluation. |
| `LR_SCHEDULER` | `cosine_with_restarts` | Periodic learning rate restarts to escape shallow local minima. |

---

## 5. Preprocessing & Augmentation Pipeline

### 5.1 Language & Sentiment Filtering
* Filters dataset where `language` $\in \{\text{'ar'}, \text{'da'}\}$.
* Distinguishes dialectal Arabic (`variant = 'dialect'`) from Modern Standard Arabic (`variant = 'msa'`).
* Normalizes labels: `"NEGATIVE": 0`, `"NEUTRAL": 1`, `"POSITIVE": 2`.

### 5.2 Stratified Partitioning (80/10/10)
* Partitioned independently for MSA and Dialect subsets using `split_percentage`:
  * **Train Set**: 80%
  * **Validation Set**: 10%
  * **Test Set**: 10%
* Preserves natural dialectal distribution across splits.

### 5.3 Positive Class Augmentation
* The positive class is augmented using Arabic intensifiers:
  $$\text{Intensifiers} = \{\text{" جداً"}, \text{" حقاً"}, \text{" فعلاً"}, \text{" للغاية"}, \text{" تماماً"}\}$$
* Samples from class 2 are randomly suffixed with an intensifier, doubling positive instances.

---

## 6. Code Walkthrough & Component Reference

### 6.1 Data Processing Functions
* `parse_sentiment(x)`: Converts strings/digits to $\{0, 1, 2\}$, returning $-1$ for invalid values.
* `compute_class_weights(df, epoch)`: Computes normalized inverse class frequencies.
* `analyze_class_distribution(df, name)`: Prints ASCII bar distribution and underrepresentation warnings.
* `augment_positive_class(df, augment_factor)`: Appends intensifiers to positive samples.
* `split_percentage(df_group, val_ratio, test_ratio, seed)`: Two-stage stratified splitting.
* `check_text_lengths(df, tokenizer, max_len, sample_size)`: Reports token statistics (mean, 95th percentile, max).
* `process_long_document_sliding_window(text, tokenizer, max_len, stride)`: Uses `return_overflowing_tokens=True` to create overlapping token chunks.
* `create_sliding_window_dataset(df, tokenizer, max_len, stride)`: Flattens all chunk dictionaries into a structured dataset.
* `collate_with_chunks(batch)`: Custom collator stacking tensors while preserving chunk metadata (`num_chunks`, `is_long`).

### 6.2 Custom Classes
* `SlidingWindowDatasetWrapper`: PyTorch Dataset wrapper providing indexing over sliding-window chunks.
* `EarlyStoppingCallback`: Monitors `eval_f1_macro`, stopping training if no improvement exceeding threshold occurs within patience steps.
* `FocalLossTrainer(Trainer)`: Subclasses HuggingFace `Trainer`. Overrides `compute_loss` to compute weighted focal loss, stripping sliding-window metadata keys before forwarding to the Transformer model.

### 6.3 Evaluation & Visualization
* `compute_metrics(eval_pred)`: Computes Accuracy, Macro F1, Weighted F1, and per-class Precision/Recall/F1.
* `plot_confusion_matrix(trainer, eval_dataset, run_dir, split_name)`: Generates seaborn confusion matrices with per-class percentage annotations.
* `save_predictions(trainer, test_dataset, run_dir, split_name)`: Exports `predictions_<split>.csv` containing prediction, true label, confidence, and correctness indicator.
* `save_run_artifacts(trainer, run_dir, eval_msa_ds, eval_dial_ds)`: Generates training/eval loss curves, F1 macro progression plots, and `eval_results.json`.

---

## 7. Execution & Resumption

### Run Script
```powershell
python training/arabic/fine_tune_arabic_sentiment.py
```

### Auto-Resume Feature
Setting `RESUME_LAST_CHECKPOINT = True` on line 72 automatically scans `experiments/arabic_sentiment/` for the latest valid checkpoint containing `trainer_state.json` and `pytorch_model.bin`/`model.safetensors`, continuing training without data loss.

### Output Artifacts
All outputs are saved to `experiments/arabic_sentiment/<timestamp>/`:
* `best_model/` / Root directory: Saved model weights & tokenizer.
* `hyperparameters.json`: Complete snapshot of all hyperparameters and split sizes.
* `eval_results.json` & `test_results.json`: Metric reports across MSA, Dialect, and Mixed sets.
* `plots/`: Loss curves, F1-Macro curves, and confusion matrices.
