# Med-AGCNet

**Official PyTorch Implementation of Med-AGCNet: Adaptive Global Context Network for Medical Image Classification**

Med-AGCNet is a deep learning architecture designed for medical image classification by combining **local feature extraction**, **large receptive-field modeling**, and **global contextual information** within a unified convolutional framework.

The architecture introduces an **Adaptive Global Context Block (AGCB)** that integrates complementary feature branches and dynamically combines their representations.

This repository provides a complete research-oriented implementation for:

* Model training
* Validation and testing
* Baseline comparison
* Ablation studies
* Explainable AI using Grad-CAM
* Classification threshold optimization
* Automatic result generation
* Experimental reproducibility

---

## Architecture Overview

The Med-AGCNet architecture consists of a convolutional stem followed by multiple Adaptive Global Context Blocks.

```text
Input Medical Image
        │
        ▼
┌─────────────────────────────┐
│          CNN Stem           │
│   Conv + BN + GELU          │
│        + MaxPool            │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Adaptive Global Context     │
│ Block (AGCB)                │
│                             │
│  ┌───────────────────────┐  │
│  │ Local Convolution     │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │ Large Receptive Field │  │
│  └───────────────────────┘  │
│                             │
│  ┌───────────────────────┐  │
│  │ Global Context        │  │
│  │ Attention             │  │
│  └───────────────────────┘  │
│             │               │
│             ▼               │
│      Adaptive Fusion        │
│             │               │
│      Residual Connection    │
└─────────────────────────────┘
        │
        ▼
 Hierarchical AGCB Stages
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

---

## Adaptive Global Context Block

The proposed **Adaptive Global Context Block (AGCB)** contains three complementary branches.

### Local Convolution Branch

Captures fine-grained local patterns and spatial structures within medical images.

### Large Receptive Field Branch

Uses larger convolutional receptive fields to capture broader spatial relationships.

### Global Context Attention Branch

Models long-range dependencies and global contextual information across the feature map.

### Adaptive Fusion

The outputs of the branches are dynamically combined using an adaptive gating mechanism.

A residual connection is then applied to preserve the original feature representation and improve information flow.

---

## Main Features

### Deep Learning Architecture

* PyTorch implementation
* CNN-based feature extraction
* Adaptive Global Context Blocks
* Global Context Attention
* Large receptive-field modeling
* Adaptive feature fusion
* Residual learning
* Hierarchical feature extraction

### Training Pipeline

* AdamW optimizer
* Learning-rate scheduling
* Automatic device detection
* CPU support
* CUDA GPU support
* Apple MPS support where available
* Automatic Mixed Precision
* Best-model checkpointing
* Gradient clipping
* Early stopping
* Deterministic experiment mode
* Reproducible random seeds

### Class-Imbalance Handling

Supported strategies:

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

The implementation calculates a comprehensive set of classification metrics:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1-Score
* Macro F1
* Sensitivity
* Specificity
* Matthews Correlation Coefficient (MCC)
* ROC-AUC
* PR-AUC
* Confusion Matrix

For binary classification, the decision threshold can be automatically optimized using the validation dataset.

Supported threshold optimization objectives:

```text
balanced_accuracy
f1
```

---

## Explainable AI with Grad-CAM

Med-AGCNet includes **Gradient-weighted Class Activation Mapping (Grad-CAM)** for model interpretability.

Grad-CAM highlights image regions that contribute most strongly to the model prediction.

The inference pipeline provides:

* Predicted class
* Confidence score
* Class probabilities
* Inference time
* Grad-CAM heatmap
* Grad-CAM overlay
* Prediction metadata

This functionality enables qualitative investigation of the spatial regions influencing model decisions.

---

## Supported Dataset Formats

### PneumoniaMNIST NPZ

The implementation directly supports PneumoniaMNIST-style `.npz` files.

The dataset must contain:

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

Grayscale medical images are automatically converted into three-channel RGB representations for compatibility with the network architecture.

---

### ImageFolder Dataset

Custom datasets can also use the standard ImageFolder structure:

```text
dataset/
│
├── train/
│   ├── class_0/
│   │   ├── image_001.png
│   │   ├── image_002.png
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

Class names are automatically detected from the directory structure.

---

## Synthetic Data Mode

Synthetic data support is included only for software testing and pipeline verification.

Example:

```bash
python med_agcnet_research.py --mode train --fake-data --epochs 1
```

> **Important:** Results obtained using synthetic or FakeData must not be interpreted as scientific or medical evidence.

Real medical imaging datasets must be used for publishable experimental results.

---

## Baseline Models

The implementation supports comparison with several reference architectures:

* SimpleCNN
* ResNet-18
* EfficientNet-B0
* Med-AGCNet

Baseline experiments evaluate:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1-Score
* Parameter count
* Runtime

This enables consistent comparison between the proposed architecture and conventional CNN-based models.

---

## Ablation Study

Several Med-AGCNet variants are provided for component-level evaluation.

### Full Model

```text
med_agcnet_full
```

Uses all proposed architectural components.

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

Replaces adaptive branch fusion with non-adaptive feature aggregation.

### Local-Only Variant

```text
med_agcnet_local_only
```

Retains only local convolutional feature modeling.

These configurations enable analysis of the contribution of each proposed architectural component.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/mahmmooudian/Med-AGCNet.git
cd Med-AGCNet
```

---

## Create a Virtual Environment

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

## Install Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

Main dependencies:

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Pillow
* Scikit-learn

---

# Usage

The research pipeline is accessible through a command-line interface.

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

The default architecture is:

```text
med_agcnet_full
```

---

## Select Computing Device

Automatic detection:

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

## Baseline Comparison

```bash
python med_agcnet_research.py \
    --mode baseline \
    --data pneumoniamnist.npz \
    --comparison-epochs 10
```

---

## Ablation Study

```bash
python med_agcnet_research.py \
    --mode ablation \
    --data pneumoniamnist.npz \
    --comparison-epochs 10
```

---

## Inference and Grad-CAM

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
Confidence
Inference time
Grad-CAM visualization
Prediction metadata
```

---

## Generate Research Report

```bash
python med_agcnet_research.py --mode report
```

The report is generated as:

```text
RESEARCH_REPORT.md
```

---

## Run Complete Experimental Pipeline

```bash
python med_agcnet_research.py \
    --mode all \
    --data pneumoniamnist.npz
```

This can execute the main research workflow including training, comparison experiments, ablation analysis, and report generation.

---

# Configuration

| Argument                 | Description                     | Default                |
| ------------------------ | ------------------------------- | ---------------------- |
| `--data`                 | Dataset path                    | `./pneumoniamnist.npz` |
| `--output`               | Output directory                | `./outputs_med_agcnet` |
| `--model`                | Model architecture              | `med_agcnet_full`      |
| `--epochs`               | Training epochs                 | `20`                   |
| `--comparison-epochs`    | Baseline/Ablation epochs        | `10`                   |
| `--batch-size`           | Batch size                      | `32`                   |
| `--image-size`           | Image resolution                | `224`                  |
| `--lr`                   | Learning rate                   | `1e-4`                 |
| `--weight-decay`         | Weight decay                    | `1e-4`                 |
| `--seed`                 | Random seed                     | `42`                   |
| `--device`               | Computing device                | `auto`                 |
| `--imbalance-strategy`   | Class imbalance strategy        | `weighted_loss`        |
| `--fake-data`            | Enable synthetic data           | Disabled               |
| `--no-threshold-tuning`  | Disable threshold optimization  | Disabled               |
| `--no-amp`               | Disable mixed precision         | Disabled               |
| `--nondeterministic`     | Disable deterministic execution | Disabled               |
| `--pretrained-baselines` | Use pretrained baseline weights | Disabled               |

---

# Generated Outputs

After training, the pipeline can generate:

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

The implementation includes several mechanisms to improve experimental reproducibility.

The pipeline records and controls:

* Python random seed
* NumPy random seed
* PyTorch random seed
* CUDA random seed
* Deterministic execution settings
* Model configuration
* Training history
* Best checkpoint
* Environment metadata
* Test-set predictions
* Classification threshold

Default random seed:

```text
42
```

---

# Recommended Research Workflow

```text
Dataset Validation
        │
        ▼
Med-AGCNet Training
        │
        ▼
Validation and Threshold Selection
        │
        ▼
Test Evaluation
        │
        ▼
Baseline Comparison
        │
        ▼
Ablation Study
        │
        ▼
Grad-CAM Analysis
        │
        ▼
Metrics and Figure Export
        │
        ▼
Research Report
```

---

# Repository Structure

```text
Med-AGCNet/
│
├── med_agcnet_research.py
└── README.md
```

Additional experimental outputs are automatically generated when the research pipeline is executed.

---

# Code Availability

The implementation of Med-AGCNet is publicly available in this repository:

https://github.com/mahmmooudian/Med-AGCNet

The repository provides the source code required for:

* Model training
* Evaluation
* Baseline comparison
* Ablation experiments
* Explainability analysis
* Experimental reproduction

---

# Citation

If you use Med-AGCNet in academic research, please cite the associated research paper.

Publication information will be added after formal publication.

```bibtex
@article{medagcnet2026,
  title   = {Med-AGCNet: Adaptive Global Context Network for Medical Image Classification},
  author  = {Mahmoudian, Amir Mohammad},
  journal = {To be updated},
  year    = {2026}
}
```

---

# Disclaimer

This project is intended for **research and educational purposes only**.

Med-AGCNet is not a certified medical device and should not be used directly for clinical diagnosis, treatment decisions, or patient care without appropriate clinical validation and regulatory approval.

---

# Author

**Amir Mohammad Mahmoudian**

Computer Science
Artificial Intelligence & Machine Learning Research

GitHub: [@mahmmooudian](https://github.com/mahmmooudian)

---

## License

License information will be added separately.
