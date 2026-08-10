# Med-AGCNet

**Official PyTorch Implementation of Med-AGCNet: Adaptive Global Context Network for Medical Image Classification**

Med-AGCNet is a deep learning architecture designed for medical image classification by combining **local feature extraction**, **large receptive-field modeling**, and **global contextual information** within a unified convolutional framework.

The architecture introduces an **Adaptive Global Context Block (AGCB)** that integrates complementary feature branches and dynamically combines their representations. The implementation is designed as a reproducible research pipeline and includes model training, evaluation, explainability, baseline comparison, ablation studies, and automated result generation.

---

## Overview

Conventional convolutional neural networks are effective at extracting local visual patterns but may have limitations in modeling long-range dependencies and broader contextual relationships within medical images.

Med-AGCNet addresses this limitation using three complementary mechanisms:

* **Local Convolution Branch** for fine-grained spatial features
* **Large Receptive Field Branch** for wider contextual information
* **Global Context Attention Branch** for long-range dependencies

The outputs of these branches are combined through an **Adaptive Fusion mechanism**, followed by residual feature refinement.

The complete implementation is provided in:

```text
med_agcnet_research.py
```

---

## Architecture

The overall Med-AGCNet pipeline can be summarized as:

```text
Input Medical Image
        │
        ▼
┌─────────────────────┐
│      CNN Stem       │
│ Conv + BN + GELU    │
│     + MaxPool       │
└─────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Adaptive Global Context Block │
│                               │
│  ┌─────────────────────────┐  │
│  │ Local Convolution       │  │
│  └─────────────────────────┘  │
│                               │
│  ┌─────────────────────────┐  │
│  │ Large Receptive Field   │  │
│  └─────────────────────────┘  │
│                               │
│  ┌─────────────────────────┐  │
│  │ Global Context Attention│  │
│  └─────────────────────────┘  │
│              │                │
│              ▼                │
│      Adaptive Fusion          │
│              │                │
│        Residual Connection    │
└───────────────────────────────┘
        │
        ▼
 Multiple Hierarchical Stages
        │
        ▼
 Global Average Pooling
        │
        ▼
      Classifier
        │
        ▼
 Medical Image Prediction
```

The network processes features hierarchically through multiple AGCB stages while progressively increasing the feature representation capacity.

---

## Main Features

The implementation provides a complete experimental pipeline including:

### Model Architecture

* Med-AGCNet full architecture
* Adaptive Global Context Blocks
* Local convolution modeling
* Large receptive-field modeling
* Global context attention
* Adaptive feature fusion
* Residual feature refinement
* Hierarchical feature extraction

### Training

* PyTorch-based training pipeline
* AdamW optimizer
* Learning-rate scheduling
* Automatic CPU / CUDA / MPS device selection
* Automatic Mixed Precision where supported
* Deterministic experiment mode
* Reproducible random seeds
* Best-model checkpointing
* Optional gradient clipping
* Optional early stopping
* Class imbalance handling

### Class-Imbalance Strategies

The implementation supports:

```text
none
weighted_loss
weighted_sampler
both
```

The default configuration uses:

```text
weighted_loss
```

---

## Evaluation Metrics

Med-AGCNet reports multiple classification metrics including:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1-Score
* Sensitivity
* Specificity
* Matthews Correlation Coefficient (MCC)
* ROC-AUC
* PR-AUC
* Confusion Matrix

For binary classification tasks, the implementation can automatically optimize the classification threshold using the validation set.

Supported threshold optimization objectives include:

```text
balanced_accuracy
f1
```

---

## Explainable AI — Grad-CAM

The implementation includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** for model interpretability.

Grad-CAM can be used to visualize image regions that contributed most strongly to the model prediction.

Generated explainability outputs include:

```text
Original Image
Prediction
Class Probabilities
Inference Time
Grad-CAM Heatmap
Grad-CAM Overlay
```

This functionality enables qualitative analysis of the spatial regions influencing the network's decision.

---

## Supported Datasets

### PneumoniaMNIST NPZ

The implementation directly supports PneumoniaMNIST-style `.npz` datasets.

The NPZ file should contain:

```text
train_images
train_labels
val_images
val_labels
test_images
test_labels
```

Example:

```text
pneumoniamnist.npz
```

The implementation automatically converts grayscale medical images into three-channel images for compatibility with the network architecture.

---

### ImageFolder Dataset

Custom medical imaging datasets can also be organized using the following structure:

```text
dataset/
│
├── train/
│   ├── class_0/
│   │   ├── image_001.png
│   │   └── ...
│   │
│   └── class_1/
│       ├── image_001.png
│       └── ...
│
├── val/
│   ├── class_0/
│   └── class_1/
│
└── test/
    ├── class_0/
    └── class_1/
```

The classes are automatically detected from the folder structure.

---

## Synthetic Data Mode

A synthetic dataset mode is included exclusively for software testing and pipeline validation.

Run:

```bash
python med_agcnet_research.py --mode train --fake-data --epochs 1
```

> **Important:** Synthetic/FakeData results must not be interpreted as scientific or medical evidence.

Real medical datasets must be used for experimental results reported in research publications.

---

## Baseline Models

Med-AGCNet can be compared against several reference architectures:

* SimpleCNN
* ResNet-18
* EfficientNet-B0
* Med-AGCNet

The baseline comparison evaluates:

```text
Accuracy
Balanced Accuracy
Precision
Recall
F1-Score
Parameter Count
Runtime
```

This provides a consistent experimental framework for comparing Med-AGCNet with conventional convolutional architectures.

---

## Ablation Study

The implementation provides dedicated Med-AGCNet variants for component-level analysis.

Available configurations include:

### Full Model

```text
med_agcnet_full
```

All proposed components are enabled.

### Without Global Context

```text
med_agcnet_no_global
```

Removes the Global Context Attention branch.

### Without Large Receptive Field

```text
med_agcnet_no_large_rf
```

Removes the large receptive-field branch.

### Without Adaptive Fusion

```text
med_agcnet_no_fusion
```

Replaces adaptive fusion with non-adaptive branch aggregation.

### Local Only

```text
med_agcnet_local_only
```

Retains only local convolutional feature modeling.

These variants allow the individual contribution of each architectural component to be evaluated experimentally.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/mahmmooudian/Med-AGCNet.git
cd Med-AGCNet
```

---

## 2. Create a Virtual Environment

Recommended:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

The main dependencies are:

```text
Python
PyTorch
Torchvision
NumPy
Matplotlib
Pillow
Scikit-learn
```

---

# Usage

The entire research pipeline is available through a command-line interface.

General syntax:

```bash
python med_agcnet_research.py --mode MODE [OPTIONS]
```

Supported modes:

```text
validate
train
evaluate
baseline
ablation
infer
report
all
```

---

## Validate Dataset

```bash
python med_agcnet_research.py \
    --mode validate \
    --data pneumoniamnist.npz
```

This validates the dataset structure before training.

---

## Train Med-AGCNet

```bash
python med_agcnet_research.py \
    --mode train \
    --data pneumoniamnist.npz \
    --model med_agcnet_full \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.0001
```

Default model:

```text
med_agcnet_full
```

---

## Specify Device

Automatic device detection:

```bash
--device auto
```

CPU:

```bash
--device cpu
```

CUDA GPU:

```bash
--device cuda
```

Example:

```bash
python med_agcnet_research.py \
    --mode train \
    --data pneumoniamnist.npz \
    --device cuda
```

---

## Evaluate a Trained Model

```bash
python med_agcnet_research.py \
    --mode evaluate \
    --data pneumoniamnist.npz \
    --checkpoint outputs_med_agcnet/runs/med_agcnet_full/best_model.pth
```

---

## Run Baseline Comparison

```bash
python med_agcnet_research.py \
    --mode baseline \
    --data pneumoniamnist.npz \
    --comparison-epochs 10
```

This evaluates the reference models using a common experimental pipeline.

---

## Run Ablation Study

```bash
python med_agcnet_research.py \
    --mode ablation \
    --data pneumoniamnist.npz \
    --comparison-epochs 10
```

This evaluates the contribution of the principal Med-AGCNet architectural components.

---

## Run Inference + Grad-CAM

```bash
python med_agcnet_research.py \
    --mode infer \
    --checkpoint outputs_med_agcnet/runs/med_agcnet_full/best_model.pth \
    --image sample_image.png
```

The inference pipeline generates:

```text
Prediction
Class probabilities
Inference time
Grad-CAM visualization
Prediction metadata
```

---

## Generate Research Report

```bash
python med_agcnet_research.py \
    --mode report
```

The generated report is saved as:

```text
RESEARCH_REPORT.md
```

---

## Run the Complete Experimental Pipeline

Training, baseline comparison, ablation study, and report generation can be executed using:

```bash
python med_agcnet_research.py \
    --mode all \
    --data pneumoniamnist.npz
```

---

# Important Configuration Options

| Argument                 | Description                               | Default                |
| ------------------------ | ----------------------------------------- | ---------------------- |
| `--data`                 | Dataset path                              | `./pneumoniamnist.npz` |
| `--output`               | Output directory                          | `./outputs_med_agcnet` |
| `--model`                | Model architecture                        | `med_agcnet_full`      |
| `--epochs`               | Main training epochs                      | `20`                   |
| `--comparison-epochs`    | Baseline/Ablation epochs                  | `10`                   |
| `--batch-size`           | Batch size                                | `32`                   |
| `--image-size`           | Input image resolution                    | `224`                  |
| `--lr`                   | Learning rate                             | `1e-4`                 |
| `--weight-decay`         | Weight decay                              | `1e-4`                 |
| `--seed`                 | Random seed                               | `42`                   |
| `--device`               | Computing device                          | `auto`                 |
| `--imbalance-strategy`   | Class imbalance handling                  | `weighted_loss`        |
| `--fake-data`            | Enable synthetic testing data             | Disabled               |
| `--no-threshold-tuning`  | Disable validation threshold optimization | Disabled               |
| `--no-amp`               | Disable automatic mixed precision         | Disabled               |
| `--nondeterministic`     | Disable deterministic execution           | Disabled               |
| `--pretrained-baselines` | Use pretrained baseline models            | Disabled               |

---

# Generated Results

After training, the default output structure is:

```text
outputs_med_agcnet/
│
├── environment.json
│
├── runs/
│   └── med_agcnet_full/
│       ├── best_model.pth
│       ├── config.json
│       ├── metrics.json
│       ├── history.csv
│       ├── history.json
│       ├── test_predictions.csv
│       ├── classification_report.txt
│       ├── confusion_matrix.png
│       ├── training_loss.png
│       ├── training_accuracy.png
│       ├── roc_curve.png
│       └── pr_curve.png
│
├── comparisons/
│   │
│   ├── baseline/
│   │   ├── baseline_results.csv
│   │   ├── baseline_results.json
│   │   ├── balanced_accuracy.png
│   │   └── weighted_f1.png
│   │
│   └── ablation/
│       ├── ablation_results.csv
│       ├── ablation_results.json
│       ├── balanced_accuracy.png
│       └── weighted_f1.png
│
├── inference/
│   ├── image_prediction.json
│   └── image_gradcam.png
│
└── RESEARCH_REPORT.md
```

---

# Reproducibility

Reproducibility is a primary objective of this implementation.

The pipeline records and controls:

* Python random seed
* NumPy random seed
* PyTorch random seed
* CUDA random seed
* Deterministic CUDA execution
* Experiment configuration
* Training history
* Model checkpoint
* Environment metadata
* Evaluation predictions
* Classification threshold

Default seed:

```text
42
```

This facilitates more consistent reproduction and comparison of experimental results.

---

# Research Workflow

A recommended experimental workflow is:

```text
1. Validate Dataset
        ↓
2. Train Med-AGCNet
        ↓
3. Evaluate Test Performance
        ↓
4. Run Baseline Comparison
        ↓
5. Run Ablation Study
        ↓
6. Generate Grad-CAM Visualizations
        ↓
7. Export Metrics and Figures
        ↓
8. Generate Research Report
```

---

# Project Status

The current repository contains the research implementation of Med-AGCNet developed for experimental evaluation of adaptive local-global feature modeling in medical image classification.

The implementation is intended to support:

* Architecture validation
* Controlled experimental evaluation
* Baseline comparison
* Component ablation
* Explainability analysis
* Reproducibility
* Academic research

---

# Citation

If you use this implementation in academic work, please cite the associated Med-AGCNet research paper.

Citation information will be updated upon publication.

```bibtex
@article{medagcnet,
  title   = {Med-AGCNet},
  author  = {Mahmoudian, Amir Mohammad},
  journal = {To be updated},
  year    = {2026}
}
```

---

# Code Availability

The source code associated with the Med-AGCNet research project is publicly available through this repository.

The repository provides the implementation required for training, evaluation, baseline comparisons, ablation studies, explainability analysis, and reproduction of the proposed experimental workflow.

---

# Disclaimer

This project is provided **for research and educational purposes only**.

Med-AGCNet is not a certified medical device and should not be used for clinical diagnosis, treatment decisions, or direct patient care without appropriate clinical validation and regulatory approval.

---

# Author

**Amir Mohammad Mahmoudian**

Computer Science
Artificial Intelligence & Machine Learning Research

GitHub: `@mahmmooudian`

---

## License

Licensing information will be added to the repository separately.
