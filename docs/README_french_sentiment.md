# French Sentiment Analysis — Fine-Tuning Documentation

This document provides a comprehensive technical guide to [`fine_tune_french_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_sentiment.py). It covers the CamemBERT-large architecture, cross-lingual MarianMT translation integration, extreme Focal Loss calibration ($\gamma=4.0$), positive class intensifier augmentation, and complete script execution details.

---

## 1. Overview & Objective

* **Target Task**: 3-Class Sentiment Classification (`NEGATIVE`: 0, `NEUTRAL`: 1, `POSITIVE`: 2).
* **Target Script**: [`training/french/fine_tune_french_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_sentiment.py)
* **Base Model**: [`camembert/camembert-large`](https://huggingface.co/camembert/camembert-large) (435M Parameters).
* **Dataset**: Real French news articles combined with cross-lingually translated English news data from `translated_data/`.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 (4GB–6GB VRAM) with mixed precision (`fp16`) and `expandable_segments`.
* **Performance Target**: **>80% Overall Accuracy** with high minority Positive class recall.

---

## 2. Theoretical & Architectural Rationale

### 2.1 CamemBERT-Large (435M Parameters)
* **Why CamemBERT-Large over Base?**: Preliminary runs with `camembert-base` (110M params) plateaued at ~74% accuracy due to capacity limits when processing syntactically elaborate French journalistic texts. Upgrading to `camembert-large` (24 layers, 1024 hidden size, 16 attention heads, 435M parameters) broke the 80% accuracy threshold.
* **Why Gradient Checkpointing is Disabled**: While gradient checkpointing saves VRAM, its compute re-evaluation overhead on a 435M parameter model degrades training throughput on laptop GPUs. VRAM is instead constrained by selecting `MAX_LEN = 384` and `BATCH_SIZE = 4`.

### 2.2 Cross-Lingual Translation Augmentation
* To overcome the smaller size of the raw French dataset, English news articles are pre-translated using `Helsinki-NLP/opus-mt-en-fr` and cached in Parquet format.
* Real French and translated French are merged, deduplicated, and partitioned together, increasing training volume while CamemBERT-large demonstrates high robustness to minor translation artifacts.

---

## 3. Mathematical Formulations & Loss Function

### 3.1 High-Gamma Focal Loss ($\gamma = 4.0, \alpha = 0.65$)
In news corpora, the `POSITIVE` sentiment class is severely underrepresented. A standard Cross-Entropy loss yields low recall on positive instances. To counter this, an aggressive focusing exponent $\gamma = 4.0$ is utilized:

$$\text{FL}(p_t) = -\alpha \cdot w_c \cdot (1 - p_t)^{\gamma} \cdot \log(p_t + 10^{-8})$$

* When $p_t = 0.9$ (easy sample): $(1 - 0.9)^4 = 0.0001$ ($99.99\%$ loss attenuation).
* When $p_t = 0.2$ (hard/confused sample): $(1 - 0.2)^4 = 0.4096$ (maintains gradient signal).

### 3.2 Enhanced Class Weighting Formulation
Custom weight scaling aggressively amplifies the gradient contribution of positive samples:

$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
$$w_{\text{POSITIVE}} \leftarrow w_{\text{POSITIVE}} \times 2.5, \quad w_{\text{NEUTRAL}} \leftarrow w_{\text{NEUTRAL}} \times 0.7, \quad w_{\text{NEGATIVE}} \leftarrow w_{\text{NEGATIVE}} \times 1.1$$
$$w_c^{\text{final}} = \text{clip}(w_c, 0.5, 5.0)$$

---

## 4. Hyperparameter Choices & Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `384` | Optimal trade-off covering 95%+ of French news articles while fitting 435M parameters into VRAM. |
| `TRAIN_BATCH_SIZE` | `4` | Maximum physical batch size fitting in 4GB–6GB VRAM at 384 tokens with `fp16`. |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Delivers an effective batch size of $4 \times 8 = 32$. |
| `LEARNING_RATE` | `1e-5` | Optimal fine-tuning rate for CamemBERT-large. |
| `NUM_EPOCHS` | `15` | Enables full convergence across real and translated data distributions. |
| `WARMUP_RATIO` | `0.2` | 20% warmup steps preventing early gradient instability. |
| `WEIGHT_DECAY` | `0.05` | Strong regularization preventing overfitting on 435M weights. |
| `FOCAL_ALPHA` | `0.65` | Loss scale coefficient. |
| `FOCAL_GAMMA` | `4.0` | High focusing parameter for extreme class imbalance recovery. |
| `AUGMENT_FACTOR` | `0.5` | Synthetically expands the POSITIVE class by 50%. |
| `EARLY_STOPPING_PATIENCE` | `10` | Evaluation patience steps. |
| `EARLY_STOPPING_THRESHOLD`| `0.003`| Minimum improvement threshold in `eval_f1_macro`. |

---

## 5. Preprocessing & Augmentation Pipeline

### 5.1 French Intensifier Augmentation
The training partition undergoes synthetic expansion of the `POSITIVE` class (`AUGMENT_FACTOR = 0.5`) by randomly injecting French intensifiers:

$$\text{Intensifiers} = \{\text{"très"}, \text{"vraiment"}, \text{"extrêmement"}, \text{"absolument"}, \text{"totalement"}, \text{"complètement"}, \text{"particulièrement"}, \text{"tellement"}, \text{"incroyablement"}, \text{"formidablement"}, \text{"fantastiquement"}, \text{"merveilleusement"}\}$$

Intensifiers are randomly prefixed or suffixed to positive news headlines, improving model sensitivity to positive polarity cues.

### 5.2 Stratified Partitioning (80/10/10)
* **Train**: 80%
* **Validation**: 10%
* **Test**: 10%

---

## 6. Code Walkthrough & Component Reference

### 6.1 Key Functions
* `load_cached_data()`: Reads `translated_french_data_all.parquet` and `real_french_data.parquet`, merging and verifying column integrity.
* `augment_positive_class(df, augment_factor)`: Implements intensifier-based positive sample expansion.
* `collate_fn(batch)`: Tokenizes dynamic batches with padding to `max_length = 384`.
* `compute_metrics(eval_pred)`: Computes Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall, and per-class metrics.
* `plot_confusion_matrix(trainer, eval_dataset, run_dir, split_name)`: Generates seaborn heatmap annotated with per-class percentage accuracies.
* `save_run_artifacts(trainer, run_dir, eval_dataset, split_name)`: Plots loss trajectories, F1-macro progression, and saves `eval_results.json`.

### 6.2 Custom Classes
* `SentimentDataset(Dataset)`: PyTorch Dataset yielding raw text strings and integer labels.
* `CustomTrainer(Trainer)`: Subclasses HuggingFace `Trainer` to compute custom High-Gamma Weighted Focal Loss and track `best_f1`.

---

## 7. Execution Guide

### Run Script
```powershell
python training/french/fine_tune_french_sentiment.py
```

### Outputs
Stored in `experiments/french_sentiment_camembert_large_final/`:
* `best_model/`: Model weights, configuration, and tokenizer.
* `checkpoints/`: Checkpoint snapshots.
* `label_mapping.json`: ID to sentiment name mapping.
* `test_predictions.csv`: Sample-by-sample predictions on the test set.
* `summary.json`: Complete configuration and outcome summary.
* `plots/`:
  * `confusion_matrix_test.png`
  * `train_loss.png`
  * `eval_loss.png`
  * `eval_f1_accuracy.png`
  * `learning_curve.png`
