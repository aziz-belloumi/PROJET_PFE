# English Sentiment Analysis — Fine-Tuning Documentation

This document provides a comprehensive technical reference for [`fine_tune_english_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_sentiment.py). It details the large-scale RoBERTa backbone, online text augmentation engine, Focal Loss with Label Smoothing, OOM recovery mechanism, and the full evaluation suite.

---

## 1. Overview & Objective

* **Target Task**: 3-Class Sentiment Classification (`NEGATIVE`, `NEUTRAL`, `POSITIVE`).
* **Target Script**: [`training/english/fine_tune_english_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_sentiment.py)
* **Base Model**: [`cardiffnlp/twitter-roberta-large-topic-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-large-topic-sentiment-latest) (355M Parameters).
* **Dataset**: English subset (`language == 'en'`) from `data/global_data_libelised.csv`.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 (6GB VRAM) with mixed precision (`fp16`), gradient checkpointing, expandable CUDA segments, and dynamic OOM recovery.

---

## 2. Theoretical & Architectural Rationale

### 2.1 Twitter-RoBERTa-Large Backbone
* **Domain Alignment**: News headlines and social commentary are concise, informal, and stylistically diverse. The Cardiff NLP Twitter-RoBERTa-Large model was pre-trained on 58M tweets and fine-tuned on sentiment polarity benchmarks.
* **Model Capacity**: RoBERTa-Large (24 layers, 1024 hidden size, 16 attention heads, 355M parameters) provides significantly higher representational capacity than base models, capturing subtle contextual valence and complex sarcasm.

### 2.2 Online Dynamic Text Augmentation
Rather than static data generation, an on-the-fly `TextAugmenter` applies stochastic transformations during training data loading (`AUGMENT_PROB = 0.3`):
1. **Sentiment-Preserving Synonym Replacement**: Substitutes words from a curated sentiment dictionary (e.g., *good $\to$ great*, *terrible $\to$ abysmal*, *adore $\to$ cherish*) with a $15\%$ per-word probability while preserving capitalization.
2. **Random Word Deletion**: Randomly drops words with keep probability $p=0.85$ to simulate telegraphic or incomplete news sentences.
3. **Adjacent Word Swap**: Swaps adjacent word tokens to encourage invariance to minor syntactic variations.

---

## 3. Mathematical Formulations & Loss Function

### 3.1 Focal Loss with Label Smoothing
To prevent model overconfidence on frequent classes while directing gradients toward hard-to-classify samples, the script implements an integrated Focal Loss with smoothed label targets:

$$\mathcal{L}_{\text{FocalSmoothed}} = \frac{1}{B} \sum_{i=1}^B (1 - p_{t,i})^\gamma \cdot \text{CE}(\mathbf{z}_i, \mathbf{y}_i^{\text{smooth}})$$

Where:
* **Smoothed Target**:
  $$y_{i, k}^{\text{smooth}} = \begin{cases} 1 - \epsilon_{\text{smooth}}, & \text{if } k = y_i \\ \frac{\epsilon_{\text{smooth}}}{K - 1}, & \text{if } k \neq y_i \end{cases}$$
  With label smoothing factor $\epsilon_{\text{smooth}} = 0.05$ and $K = 3$.
* **Probability Ratio**: $p_{t,i} = \exp(-\text{CE}(\mathbf{z}_i, \mathbf{y}_i))$.
* **Focusing Exponent**: $\gamma = 2.0$.

### 3.2 Smoothed Inverse Class Weights
To prevent extreme weight disparities from destabilizing training:

$$w_c = \frac{1}{N_c}, \quad \tilde{w}_c = \frac{w_c}{\sum_{k} w_k} \cdot K$$
$$w_c^{\text{final}} = \tilde{w}_c \times 0.9 + 0.1$$

---

## 4. Hyperparameter Choices & Empirical Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `160` | News headlines and short articles rarely exceed 160 tokens; saves VRAM for the large 355M backbone. |
| `TRAIN_BATCH_SIZE` | `8` | Maximum physical batch size fitting in 6GB VRAM with `fp16` and gradient checkpointing. |
| `GRAD_ACCUM` | `4` | Delivers an effective batch size of $8 \times 4 = 32$. |
| `LEARNING_RATE` | `5e-6` | Conservative learning rate required for fine-tuning 355M parameter RoBERTa without destructive updates. |
| `NUM_EPOCHS` | `12` | Extended training schedule to ensure convergence with low learning rate and cosine decay. |
| `WARMUP_RATIO` | `0.1` | 10% gradual learning rate warmup. |
| `WEIGHT_DECAY` | `0.01` | Weight decay regularizing large parameter matrices. |
| `LABEL_SMOOTHING`| `0.05` | Prevents overconfidence without diluting polarity boundaries. |
| `FOCAL_GAMMA` | `2.0` | Quadratic dampening of easy loss components. |
| `AUGMENT_PROB` | `0.3` | 30% chance per sample per epoch to receive synthetic augmentation. |
| `EARLY_STOPPING_PATIENCE` | `10` | Generous patience window avoiding premature termination during loss plateaus. |
| `LR_SCHEDULER` | `cosine` | Smooth decay of learning rate to zero. |

---

## 5. Fault Tolerance & Memory Optimization

### 5.1 Dynamic OOM Recovery Guard
The script wraps `trainer.train()` inside an automatic retry loop:
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_memory()
            training_args.per_device_train_batch_size = max(2, training_args.per_device_train_batch_size // 2)
```
If CUDA encounters an Out-Of-Memory error, GPU memory caches are purged and the physical batch size is halved dynamically.

### 5.2 Early Stopping with Warmup Protection
Standard early stopping algorithms can terminate training prematurely if evaluation metrics fluctuate during early warmup steps. The custom `EarlyStoppingWithWarmup` suppresses early stopping checks until `global_step >= warmup_steps (1000)`.

### 5.3 Intelligent Auto-Resumption
The script scans `sentiment_experiments/english_sentiment_enhanced/` on startup. If an incomplete run directory is detected, it validates the latest checkpoint and resumes execution seamlessly.

---

## 6. Preprocessing & Data Pipeline

1. **Text Cleaning (`clean_text`)**:
   * Lowercasing and URL removal (`http\S+`, `www\S+`).
   * Negation handling: transforms negation triggers (`not`, `no`, `never`, `none`) into joined prefixes (e.g. `"not happy"` $\to$ `"not_happy"`).
   * Filtering invalid/unknown labels.
2. **Stratified Split (70/15/15)**:
   * **Train**: 70%
   * **Validation**: 15%
   * **Test**: 15%
3. **Dataset & Collation**: `EnhancedDataset` applies on-the-fly text augmentation, and `enhanced_collate_fn` executes batched tokenization with padding.

---

## 7. Comprehensive Evaluation & Visualization Suite

The script automatically generates 10 high-resolution diagnostic plots in `plots/`:

1. **`train_loss.png`**: Step-by-step training loss curve.
2. **`eval_loss.png`**: Validation loss trajectory across evaluation intervals.
3. **`learning_curve.png`**: Overlay comparing train loss vs. validation loss.
4. **`eval_f1_accuracy.png`**: Dual-axis trajectory of Validation Macro F1 and Accuracy.
5. **`test_confusion_matrix.png`**: Side-by-side heatmaps of raw counts and normalized percentages on the held-out test set.
6. **`val_confusion_matrix.png`**: Validation set confusion matrices.
7. **`roc_curves.png`**: One-vs-Rest Receiver Operating Characteristic (ROC) curves with computed Area Under Curve (AUC) for each sentiment class.
8. **`precision_recall_curves.png`**: Precision-Recall curves with Average Precision (AP) scores.
9. **`per_class_f1_bar.png`**: Comparative bar chart of per-class F1 scores (Validation vs. Test).
10. **`confidence_analysis.png`**: Distribution histograms comparing model confidence on correct vs. incorrect predictions.

---

## 8. Code Walkthrough & Component Reference

### 8.1 Functions
* `clean_text(text)`: Cleans URLs, normalizes whitespace, and handles negation prefixes.
* `load_and_prepare_data_enhanced()`: Loads CSV, filters English data, executes stratified 70/15/15 split.
* `get_class_weights_enhanced(train_df, num_labels)`: Calculates smoothed inverse class weights.
* `clear_memory()`: Runs `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()`.
* `find_resume_point()`: Discovers unfinished runs and valid checkpoints.
* `plot_separated_metrics(log_history, plots_dir)`: Renders loss, learning curves, and validation trajectories.

### 8.2 Classes
* `TextAugmenter`: Online synonym replacement, word deletion, and adjacent swapping.
* `EnhancedDataset(Dataset)`: PyTorch Dataset yielding texts and labels with conditional online augmentation.
* `EnhancedFocalLossTrainer(Trainer)`: Subclasses `Trainer` to implement Focal Loss with label smoothing.
* `EarlyStoppingWithWarmup(EarlyStoppingCallback)`: Ignores early stopping triggers during the warmup phase.

---

## 9. Execution Guide

### Run Script
```powershell
python training/english/fine_tune_english_sentiment.py
```

### Outputs
Stored in `sentiment_experiments/english_sentiment_enhanced/run_<timestamp>/`:
* `best_model/`: Model weights, configuration, and tokenizer.
* `checkpoints/`: Periodic checkpoint snapshots.
* `results/`:
  * `hyperparameters.json`
  * `data_info.json`
  * `metrics.json`
  * `classification_report.json`
  * `confusion_matrix_test.csv` & `confusion_matrix_val.csv`
  * `predictions_with_confidence.csv`
* `plots/`: All 10 generated evaluation figures.
* `summary.json`: Top-level metadata and execution summary.
