# Arabic Topic Classification — Fine-Tuning Documentation

This document provides an in-depth technical guide to [`fine_tune_arabic_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_topic.py). It details the 18-category classification pipeline, Layerwise Learning Rate Decay (LLRD) architecture, Focal Loss formulation, class weight regularization, and the complete script structure.

---

## 1. Overview & Objective

* **Target Task**: 18-Category Topic Classification (Politics, Economy, Security, Energy, Conflict, Elections, Justice, Health, Weather, Sports, Culture, Education, Technology, Environment, Diplomacy, Religion, Migration, General).
* **Target Script**: [`training/arabic/fine_tune_arabic_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_topic.py)
* **Base Model**: [`CAMeL-Lab/bert-base-arabic-camelbert-mix`](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix)
* **Dataset**: Arabic articles (`language in {'ar', 'da'}`) from `data/global_data_libelised.csv`.
* **Hardware Profile**: Optimized for NVIDIA RTX 3050 with PyTorch 2.0 compile support, automatic bf16/fp16 mixed precision, and fused AdamW optimizer.

---

## 2. Target Categories & Label Mapping

The model classifies Arabic texts into 18 standardized categories:

| ID | English Label | Arabic Name | ID | English Label | Arabic Name |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | **Politics** | السياسة | `9` | **Sports** | الرياضة |
| `1` | **Economy** | الاقتصاد | `10` | **Culture** | الثقافة |
| `2` | **Security** | الأمن | `11` | **Education** | التعليم |
| `3` | **Energy** | الطاقة | `12` | **Technology** | التكنولوجيا |
| `4` | **Conflict** | النزاع | `13` | **Environment** | البيئة |
| `5` | **Elections** | الانتخابات | `14` | **Diplomacy** | الدبلوماسية |
| `6` | **Justice** | العدالة / العدل | `15` | **Religion** | الدين |
| `7` | **Health** | الصحة | `16` | **Migration** | الهجرة |
| `8` | **Weather** | الطقس | `17` | **General** | عام |

Text normalization (`NFKD` decomposition, diacritic stripping) ensures resilient topic matching across Arabic spelling variants.

---

## 3. Theoretical & Architectural Rationale

### 3.1 Layerwise Learning Rate Decay (LLRD)
* **Rationale**: Standard uniform learning rates often cause *catastrophic forgetting* in the lower Transformer layers. In BERT architectures, lower layers capture general morphological and syntactic features (e.g., Arabic root-pattern templates and affixes), while upper layers capture task-specific semantic associations.
* **Mathematical Formula**:
  For parameter group $\ell \in \{0, 1, \dots, 11\}$ (where $\ell=11$ is the topmost Transformer layer):
  $$\eta_\ell = \eta_{\text{base}} \cdot \xi^{(11 - \ell)}$$
  Where $\eta_{\text{base}} = 10^{-5}$ and decay factor $\xi = 0.95$.
  * **Classification Head**: $\eta_{\text{head}} = 10^{-5}$
  * **Top Transformer Layer (11)**: $\eta_{11} = 10^{-5} \times 0.95^0 = 10^{-5}$
  * **Bottom Layer (0)**: $\eta_0 = 10^{-5} \times 0.95^{11} \approx 5.68 \times 10^{-6}$
  * **Embeddings**: $\eta_{\text{emb}} = 10^{-5} \times 0.95^{12} \approx 5.40 \times 10^{-6}$
* **Weight Decay Regularization**: Applied ($0.01$) to all weight tensors except biases and `LayerNorm.weight` (which have decay set to $0.0$).

---

## 4. Mathematical Formulations & Loss Function

### 4.1 Class-Weighted Focal Loss
To balance learning across dense classes (e.g., Politics, General) and sparse classes (e.g., Environment, Weather), a class-weighted Focal Loss is used:

$$\mathcal{L}_{\text{Focal}}(\mathbf{z}, y) = -\alpha_y \cdot (1 - p_y)^\gamma \cdot \log(p_y)$$

Where:
* $p_y = \text{Softmax}(\mathbf{z})_y$ is the predicted probability of ground-truth class $y$.
* $\gamma = 1.5$ reduces the gradient contribution from easy examples.
* $\alpha_y$ is the pre-computed inverse frequency weight for class $y$.

### 4.2 Robust Inverse-Frequency Class Weights
Class weights are scaled and clipped to prevent rare classes from generating unstable gradient spikes:

$$w_c = \frac{1}{N_c}, \quad \tilde{w}_c = \frac{w_c}{\sum_{k=0}^{K-1} w_k} \cdot K$$
$$w_c^{\text{final}} = \min\left(\tilde{w}_c, 10 \cdot \text{median}(\tilde{\mathbf{w}})\right)$$

---

## 5. Hyperparameter Choices & Justifications

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| `MAX_LEN` | `512` | Topic signals often appear across the full body of news articles. |
| `EPOCHS` | `15` | Enables full convergence across 18 distinct categories. |
| `LR` | `1e-5` | Base learning rate for the top layers, decayed smoothly downward via LLRD. |
| `BATCH_SIZE` | `8` | Fits comfortably in 4GB–6GB VRAM on Ampere GPUs. |
| `GRAD_ACCUM` | `1` | Direct gradient updates per batch. |
| `LLRD_DECAY` | `0.95` | Retains linguistic priors in lower BERT layers. |
| `WARMUP_RATIO` | `0.15` | Gradual LR ramp-up to prevent instability in randomly initialized linear head. |
| `WEIGHT_DECAY` | `0.01` | Prevents overfitting on dominant topic vocabularies. |
| `FOCAL_GAMMA` | `1.5` | Moderates gradient magnitude of frequent topics. |
| `MAX_GRAD_NORM` | `0.3` | Tight gradient clipping preventing gradient explosions during multi-class training. |
| `EARLY_STOPPING_PATIENCE` | `10` | Allows model sufficient evaluation cycles to escape saddle points on rare classes. |
| `EARLY_STOPPING_THRESHOLD` | `0.001` | Minimum required improvement in `eval_f1_macro`. |

---

## 6. Preprocessing & Data Pipeline

1. **Normalization & Cleaning**: `norm()` uses Unicode `NFKD` decomposition to strip diacritics and normalize character casings.
2. **Topic Extraction**: `parse_topic()` handles integer strings, localized Arabic topic names, and English aliases.
3. **Stratified Splitting**:
   * MSA and Dialect subsets are split separately with an 80% Train / 10% Validation / 10% Test split.
   * Dialect data is integrated into the training split while maintaining evaluation integrity.
4. **HuggingFace Dataset Mapping**: Batched tokenization removes raw text columns from memory, streaming tensors directly into GPU memory via `DataCollatorWithPadding`.

---

## 7. Code Walkthrough & Component Reference

### 7.1 Key Functions
* `norm(x)`: Unicode normalization and combining character removal.
* `parse_topic(x)`: Maps numeric or text labels to integer category IDs $[0, 17]$.
* `split_percentage(df_group, test_ratio)`: Stratified train/test split helper.
* `build_class_weights(train_df, num_labels, device)`: Generates median-clipped class weights tensor.
* `get_layerwise_lr(model, base_lr, decay)`: Groups model parameters into 13 distinct learning rate tiers with selective weight decay.
* `compute_metrics(eval_pred)`: Returns Accuracy and Macro F1.

### 7.2 Custom Classes
* `FocalLoss(nn.Module)`: PyTorch module computing weighted Focal Loss.
* `EarlyStoppingCallback(TrainerCallback)`: Early stopping monitor tracking `eval_f1_macro`.
* `ConfusionMatrixCallback(TrainerCallback)`: Periodically computes and saves confusion matrices (CSV + PNG) at the end of training epochs.
* `WeightedTrainer(Trainer)`:
  * Overrides `create_optimizer()` to instantiate `AdamW` (with fused CUDA acceleration) using LLRD parameter groups.
  * Overrides `compute_loss()` to calculate weighted Focal Loss.

### 7.3 Artifact Generation & Visualizations
* `save_run_artifacts()`: Exports complete evaluation results, classification reports (`classification_report_primary.json`, `classification_report_test.json`), high-resolution confusion matrix heatmaps (`confusion_matrix_test.png`), and per-class Precision/Recall/F1 bar charts (`per_class_f1_test.png`).

---

## 8. Execution & Output Directory

### Run Script
```powershell
python training/arabic/fine_tune_arabic_topic.py
```

### Outputs
Saved in `experiments/topic/<timestamp>/`:
* `best_model/` / Root: Exported model weights and tokenizer.
* `hyperparameters.json`: Configuration snapshot.
* `eval_results.json`: Validation and test metrics.
* `classification_report_test.json`: Per-category precision, recall, and F1 scores.
* `plots/`:
  * `train_loss.png` & `eval_loss.png`
  * `eval_f1_accuracy.png`
  * `learning_curve.png`
  * `confusion_matrix_test.png`
  * `per_class_f1_test.png`
