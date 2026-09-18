# English Topic Classification — Fine-Tuning Documentation

This document provides a complete technical guide to [`fine_tune_english_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_topic.py). It covers the optimized RoBERTa-base recipe, two-stage curriculum loss scheduling, custom dropout injection, class-weighted Focal Loss, and the diagnostic evaluation pipeline.

---

## 1. Overview & Objective

* **Target Task**: 18-Category Topic Classification.
* **Target Script**: [`training/english/fine_tune_english_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_topic.py)
* **Base Model**: [`roberta-base`](https://huggingface.co/roberta-base) (125M Parameters, Optimized Fine-Tuning Recipe).
* **Dataset**: English articles (`language == 'en'`) from `data/global_data_libelised.csv`.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 with mixed precision (`fp16`) and AMP `autocast()`.

---

## 2. Theoretical & Architectural Rationale

### 2.1 RoBERTa-Base (Optimized Recipe)
* **Backbone Choice**: While `roberta-base` is more compact than large models, a refined fine-tuning recipe (elevated dropout, high label smoothing, weighted focal loss, and heavy weight decay) enables it to outperform heavier backbones while maintaining fast inference speeds and a low memory footprint.
* **Architecture-Wide Regularization**:
  To prevent overfitting across 18 distinct topic categories with uneven sample support:
  * `classifier_dropout = 0.35` (Classification head)
  * `hidden_dropout_prob = 0.35` (Hidden representations across all Transformer layers)
  * `attention_probs_dropout_prob = 0.35` (Self-attention matrix dropout)

---

## 3. Mathematical Formulations & Loss Architecture

### 3.1 Two-Stage Curriculum Loss Strategy
The custom `EnhancedTrainer` employs a two-stage loss schedule:
1. **Warmup Phase ($\text{Step} \le 100$)**:
   Standard Cross-Entropy with high Label Smoothing ($\epsilon = 0.25$) and class weights:
   $$\mathcal{L}_{\text{Stage 1}} = -\sum_{k=1}^K w_k \cdot y_k^{\text{smooth}} \log(p_k)$$
   This provides smooth, stable gradient vectors while the randomly initialized classifier head aligns with the Transformer embeddings.
2. **Focal Hard-Mining Phase ($\text{Step} > 100$)**:
   Switches automatically to Class-Weighted Focal Loss:
   $$\mathcal{L}_{\text{Stage 2}} = \frac{1}{B} \sum_{i=1}^B w_{y_i} \cdot \alpha \cdot (1 - p_{t,i})^\gamma \cdot \text{CE}(\mathbf{z}_i, y_i)$$
   Where $\alpha = 0.25$ and $\gamma = 2.0$.

### 3.2 Balanced Class Weights
Calculated using the standard scikit-learn balanced formulation:

$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$

---

## 4. Hyperparameter Choices & Empirical Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `512` | Full contextual span required to identify topics that only become explicit in document bodies. |
| `TRAIN_BATCH_SIZE` | `8` | Fits in GPU VRAM with sequence length 512. |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Delivers an effective batch size of $8 \times 4 = 32$. |
| `LEARNING_RATE` | `1e-5` | Standard optimal fine-tuning rate for RoBERTa. |
| `NUM_EPOCHS` | `6` | Fast convergence within 6 epochs due to curriculum loss scheduling. |
| `WARMUP_RATIO` | `0.1` | 10% warmup steps (with minimum 200 steps floor). |
| `WEIGHT_DECAY` | `0.08` | High $L_2$ regularization countering vocabulary memorization on dominant topics. |
| `LABEL_SMOOTHING` | `0.25` | Strong smoothing mitigating high-entropy topic boundary ambiguities. |
| `DROPOUT_RATE` | `0.35` | Injected across hidden layers, attention maps, and linear classification head. |
| `MAX_GRAD_NORM` | `0.3` | Tight gradient clipping preventing optimization divergence. |
| `EARLY_STOPPING_PATIENCE` | `2` | Epoch-level early stopping patience. |
| `LR_SCHEDULER` | `cosine` | Smooth decay of learning rate to zero. |

---

## 5. Preprocessing & Data Pipeline

1. **Filtering & Cleaning**:
   * Text length bounded between 10 characters and 3000 characters.
   * `LabelEncoder` maps 18 topic strings to integer IDs $[0, 17]$.
2. **Stratified Partitioning (80/10/10)**:
   * **Train**: 80%
   * **Validation**: 10%
   * **Test**: 10%
3. **Pre-tokenized Dataset (`RobertaDataset`)**: Pre-tokenizes and caches complete tensor dictionaries during initialization to maximize GPU throughput during training.

---

## 6. Diagnostic Visualizations & Reporting

The script generates a comprehensive suite of evaluation figures in `plots/`:

1. **Confusion Matrices (Raw & Normalized)**:
   * `confusion_matrix_raw_train.png` / `confusion_matrix_norm_train.png`
   * `confusion_matrix_raw_validation.png` / `confusion_matrix_norm_validation.png`
   * `confusion_matrix_raw_test.png` / `confusion_matrix_norm_test.png`
2. **Per-Class Breakdown Charts (Horizontal Bars for All 18 Classes)**:
   * `precision_*.png`: Precision per class with value callouts.
   * `recall_*.png`: Recall per class.
   * `f1_*.png`: F1 scores per class with a red target reference line at $0.80$.
   * `support_*.png`: Number of evaluation samples per class.
3. **Training Dynamics**:
   * `train_loss.png`, `val_loss.png`, `train_val_loss.png`
   * `val_accuracy.png`, `val_f1.png`, `val_accuracy_f1.png`
   * `learning_rate.png` (tracks cosine decay schedule)
4. **Comparative Analysis**:
   * `metrics_comparison.png`: Side-by-side bar chart of Accuracy, Macro F1, and Weighted F1 across Train, Validation, and Test sets.
   * `class_distribution_*.png`: Bar distribution of training, validation, and test splits.

---

## 7. Code Walkthrough & Component Reference

### 7.1 Functions
* `load_and_prepare_data()`: Reads CSV, filters English data, enforces length constraints, fits `LabelEncoder`, and executes stratified splits.
* `compute_metrics(eval_pred)`: Computes Accuracy, Macro Precision, Macro Recall, Macro F1, and Weighted F1.
* `plot_confusion_matrix(y_true, y_pred, class_names, title, save_dir)`: Plots raw count and normalized confusion heatmaps.
* `plot_performance_metrics(y_true, y_pred, class_names, title, save_dir)`: Renders 4 horizontal bar charts (Precision, Recall, F1, Support).
* `plot_training_history(history, save_dir)`: Extracts log history and plots 7 training/validation dynamic curves.
* `evaluate_and_plot(model, dataset, class_names, split_name, save_dir)`: Full evaluation wrapper generating metrics and plots for a given split.

### 7.2 Classes
* `RobertaDataset(Dataset)`: Pre-tokenizes texts with sequence length 512 during initialization.
* `FocalLoss(nn.Module)`: PyTorch module computing Focal Loss with $\alpha=0.25$ and $\gamma=2.0$.
* `EnhancedTrainer(Trainer)`: Subclasses `Trainer` with mixed precision `autocast()` and curriculum loss switching.

---

## 8. Execution Guide

### Run Script
```powershell
python training/english/fine_tune_english_topic.py
```

### Outputs
Stored in `experiments/english_topic/run_<timestamp>/`:
* `best_model/`: Model weights, `model_info.json`, tokenizer, and serialized `label_encoder.pkl`.
* `results/`:
  * `hyperparameters.json`
  * `data_info.json`
  * `all_metrics.json`
  * `classification_report.json`
  * `confusion_matrix.csv`
  * `predictions.csv`
* `plots/`: Full diagnostic plot suite.
* `summary.json`: Top-level experiment summary.
