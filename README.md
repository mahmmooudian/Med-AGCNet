<div align="center">

# Med-AGCNet

### Adaptive Global Context Network for Medical Image Classification

**Deep Learning · Medical Imaging · Computer Vision · Explainable AI · PyTorch**

A research-oriented deep learning framework designed to combine **local visual patterns, large receptive-field information, and global contextual representations** through adaptive feature fusion.

</div>

---

## Overview

**Med-AGCNet** is a convolutional neural network architecture for medical image classification built around the **Adaptive Global Context Block (AGCB)**.

The architecture is designed to capture complementary information at multiple spatial scales:

- Fine-grained local features
- Wider spatial relationships
- Global contextual dependencies

These representations are dynamically combined through an adaptive fusion mechanism and reinforced with residual learning.

The repository provides an end-to-end research workflow covering:

- Model training
- Validation and testing
- Baseline comparison
- Ablation studies
- Classification-threshold optimization
- Class-imbalance handling
- Grad-CAM explainability
- Automatic metric and figure generation
- Experimental reproducibility
- Single-image inference

---

## Key Capabilities

### Deep Learning Architecture

- Custom PyTorch implementation
- Adaptive Global Context Blocks
- Multi-branch feature extraction
- Local convolutional modeling
- Large receptive-field modeling
- Global context attention
- Adaptive feature fusion
- Residual feature refinement
- Hierarchical representation learning

### Training Pipeline

- AdamW optimization
- Learning-rate scheduling
- Automatic Mixed Precision
- Gradient clipping
- Early stopping
- Best-checkpoint selection
- Automatic CPU / CUDA / MPS device selection
- Deterministic experiment mode
- Reproducible random seeds

### Evaluation

The pipeline supports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- Macro F1
- Sensitivity
- Specificity
- Matthews Correlation Coefficient
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Explainable AI

Grad-CAM support provides:

- Prediction probabilities
- Confidence scores
- Inference timing
- Activation heatmaps
- Image overlays
- Prediction metadata

---

# High-Level Architecture

```mermaid
flowchart TD
    A[Medical Image] --> B[Input Preprocessing]

    B --> C[CNN Stem<br/>Conv + BatchNorm + GELU + Pooling]

    C --> D[Adaptive Global Context Block]

    D --> E1[Local Convolution Branch]
    D --> E2[Large Receptive Field Branch]
    D --> E3[Global Context Attention Branch]

    E1 --> F[Adaptive Feature Fusion]
    E2 --> F
    E3 --> F

    F --> G[Residual Refinement]

    G --> H[Hierarchical AGCB Stages]

    H --> I[Global Average Pooling]

    I --> J[Classification Head]

    J --> K[Class Prediction]

    J --> L[Evaluation Metrics]

    H --> M[Grad-CAM Explainability]
```

The central design idea is to avoid relying on a single receptive-field scale.

Instead, each AGCB extracts complementary representations and combines them adaptively before passing the refined features to the next stage.

---

## Adaptive Global Context Block

The **AGCB** is the core architectural component of Med-AGCNet.

It contains three complementary branches.

### Local Convolution Branch

Captures fine-grained spatial patterns and local texture information.

This branch is responsible for preserving detailed features that may be important for medical-image discrimination.

### Large Receptive-Field Branch

Models broader spatial relationships by expanding the effective receptive field.

This allows the network to capture structural information extending beyond small local neighborhoods.

### Global Context Attention Branch

Aggregates global contextual information across the feature map.

This branch helps the architecture model long-range relationships that may not be captured effectively by local convolutions alone.

### Adaptive Fusion

The outputs of all branches are dynamically combined using a learned gating mechanism.

Conceptually:

```text
Local Features
      │
      ├──────────────┐
      │              │
Large-RF Features ───┼──► Adaptive Fusion ──► Residual Refinement
      │              │
Global Context ──────┘
```

The residual connection preserves the original representation and improves information flow through the network.

---

## Research Workflow

```mermaid
flowchart LR
    A[Dataset Validation] --> B[Training]
    B --> C[Validation]
    C --> D[Threshold Selection]
    D --> E[Test Evaluation]
    E --> F[Baseline Comparison]
    F --> G[Ablation Study]
    G --> H[Grad-CAM Analysis]
    H --> I[Metrics & Figure Export]
    I --> J[Research Report]
```

This workflow separates model development, threshold selection, final testing, comparative experiments, and interpretability analysis.

---

## Reference Experiment

The best recorded PneumoniaMNIST experiment achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **92.47%** |
| Balanced Accuracy | **90.98%** |
| Weighted F1 | **92.38%** |
| ROC-AUC | **0.9758** |
| Decision Threshold | **0.57** |

The experiment used a fixed training/validation/test protocol and threshold selection based on validation performance.

### Dataset Split

| Split | Samples |
|---|---:|
| Training | 4,708 |
| Validation | 524 |
| Test | 624 |

> Results are research outcomes for the specified experimental configuration and dataset. They should not be interpreted as clinical performance guarantees.

---

## Supported Dataset Formats

### PneumoniaMNIST NPZ

The implementation directly supports PneumoniaMNIST-style `.npz` datasets containing:

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

Grayscale images are automatically converted to three-channel representations for compatibility with the network.

---

### ImageFolder

Custom datasets can also use the standard PyTorch ImageFolder structure:

```text
dataset/
│
├── train/
│   ├── class_0/
│   └── class_1/
│
├── val/
│   ├── class_0/
│   └── class_1/
│
└── test/
    ├── class_0/
    └── class_1/
```

Class names are automatically inferred from the directory structure.

---

## Class-Imbalance Handling

Medical datasets frequently contain unequal class distributions.

Med-AGCNet supports multiple strategies:

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

This allows the training pipeline to account for class imbalance without modifying the underlying dataset.

---

## Classification Threshold Optimization

Binary classification performance can depend significantly on the decision threshold.

Instead of always assuming:

```text
threshold = 0.50
```

the validation pipeline can automatically search for a better threshold.

Supported optimization objectives include:

```text
balanced_accuracy
f1
```

The selected threshold is then used for final test evaluation.

The test set is not used for threshold selection.

---

## Baseline Models

The research pipeline supports comparison against multiple reference architectures:

| Model | Purpose |
|---|---|
| `SimpleCNN` | Lightweight convolutional baseline |
| `ResNet-18` | Standard residual CNN baseline |
| `EfficientNet-B0` | Efficient modern CNN baseline |
| `Med-AGCNet` | Proposed architecture |

Baseline experiments can compare:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- Parameter count
- Runtime

This provides a consistent framework for evaluating the proposed architecture against conventional CNN models.

---

## Ablation Study

Multiple variants are available to analyze the contribution of individual architectural components.

| Variant | Description |
|---|---|
| `med_agcnet_full` | Full architecture |
| `med_agcnet_no_global` | Removes Global Context Attention |
| `med_agcnet_no_large_rf` | Removes large receptive-field modeling |
| `med_agcnet_no_fusion` | Replaces adaptive fusion |
| `med_agcnet_local_only` | Retains only local feature modeling |

The objective of these experiments is to determine how each architectural component contributes to the final model behavior.

---

## Explainable AI with Grad-CAM

Med-AGCNet integrates **Gradient-weighted Class Activation Mapping** for qualitative model interpretation.

The inference workflow can generate:

```text
Input Image
     │
     ▼
Med-AGCNet
     │
     ├──► Class Prediction
     ├──► Probability
     ├──► Confidence
     └──► Grad-CAM
              │
              ▼
       Activation Heatmap
              │
              ▼
        Image Overlay
```

Grad-CAM visualizations help identify image regions that most strongly influence the model prediction.

> Grad-CAM is an interpretability aid and should not be considered a clinical explanation or diagnostic localization method.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahmmooudian/Med-AGCNet.git
cd Med-AGCNet
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install torch torchvision numpy matplotlib pillow scikit-learn
```

---

## Usage

The complete research workflow is accessible through:

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

## Dataset Validation

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

Default model:

```text
med_agcnet_full
```

---

## Device Selection

Automatic selection:

```bash
--device auto
```

CPU:

```bash
--device cpu
```

CUDA:

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

---

## Run Ablation Study

```bash
python med_agcnet_research.py \
    --mode ablation \
    --data pneumoniamnist.npz \
    --comparison-epochs 10
```

---

## Inference & Grad-CAM

```bash
python med_agcnet_research.py \
    --mode infer \
    --checkpoint outputs_med_agcnet/runs/med_agcnet_full/best_model.pth \
    --image sample_image.png
```

Generated inference information includes:

- Predicted class
- Class probabilities
- Confidence
- Inference time
- Grad-CAM visualization
- Prediction metadata

---

## Run the Complete Research Pipeline

```bash
python med_agcnet_research.py \
    --mode all \
    --data pneumoniamnist.npz
```

This mode can execute the primary research workflow including:

```text
Training
   ↓
Evaluation
   ↓
Baseline Comparison
   ↓
Ablation Study
   ↓
Result Generation
   ↓
Research Report
```

---

## Configuration

| Argument | Description | Default |
|---|---|---|
| `--data` | Dataset path | `./pneumoniamnist.npz` |
| `--output` | Output directory | `./outputs_med_agcnet` |
| `--model` | Architecture | `med_agcnet_full` |
| `--epochs` | Training epochs | `20` |
| `--comparison-epochs` | Comparison epochs | `10` |
| `--batch-size` | Batch size | `32` |
| `--image-size` | Image resolution | `224` |
| `--lr` | Learning rate | `1e-4` |
| `--weight-decay` | Weight decay | `1e-4` |
| `--seed` | Random seed | `42` |
| `--device` | Compute device | `auto` |
| `--imbalance-strategy` | Class-imbalance strategy | `weighted_loss` |
| `--fake-data` | Synthetic-data mode | Disabled |
| `--no-threshold-tuning` | Disable threshold optimization | Disabled |
| `--no-amp` | Disable mixed precision | Disabled |
| `--nondeterministic` | Disable deterministic execution | Disabled |
| `--pretrained-baselines` | Enable pretrained baselines | Disabled |

---

## Generated Outputs

A complete experiment can generate:

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
│   └── ablation/
│
├── inference/
│   ├── image_prediction.json
│   └── image_gradcam.png
│
└── RESEARCH_REPORT.md
```

This makes experiments easier to audit, compare, and reproduce.

---

## Reproducibility

The implementation controls or records:

- Python random seed
- NumPy random seed
- PyTorch random seed
- CUDA random seed
- Deterministic execution settings
- Model configuration
- Training history
- Best checkpoint
- Environment metadata
- Test-set predictions
- Classification threshold

Default seed:

```text
42
```

Reproducibility is treated as part of the research pipeline rather than an optional post-processing step.

---

## Synthetic Data Mode

Synthetic data can be used for software and pipeline validation:

```bash
python med_agcnet_research.py \
    --mode train \
    --fake-data \
    --epochs 1
```

> Synthetic-data results are intended only for software validation and must not be interpreted as medical or scientific evidence.

---

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Vision | Torchvision |
| Machine Learning | Scikit-learn |
| Scientific Computing | NumPy |
| Visualization | Matplotlib |
| Image Processing | Pillow |
| Explainability | Grad-CAM |
| Acceleration | CUDA / AMP |
| Research Workflow | CLI-based experiment pipeline |

---

## Project Structure

```text
Med-AGCNet/
│
├── CITATION.cff
├── LICENSE
├── README.md
└── med_agcnet_research.py
```

The current repository intentionally keeps the complete research implementation in a single executable research pipeline.

Generated experimental outputs are created separately during execution.

---

## Engineering & Research Principles

The project emphasizes:

- **Reproducibility**
- **Transparent evaluation**
- **Validation-based threshold selection**
- **Explicit class-imbalance handling**
- **Baseline comparison**
- **Component-level ablation**
- **Explainability**
- **Separation of validation and test decisions**
- **Automatic experiment artifact generation**

The objective is not only to train a classifier, but to provide a structured workflow for investigating model behavior and architectural design decisions.

---

## Limitations

Med-AGCNet is a research implementation and currently has several important limitations:

- Experimental performance is dataset-dependent.
- External clinical validation has not been performed.
- The model has not been evaluated prospectively in a clinical workflow.
- Grad-CAM does not provide clinically validated lesion localization.
- Results should not be generalized to other imaging modalities without additional validation.
- The current repository is research-oriented rather than a production inference service.
- Regulatory validation has not been performed.

These limitations are stated explicitly to separate **research performance** from **clinical applicability**.

---

## Roadmap

Potential future work includes:

- Evaluation on additional MedMNIST datasets
- External medical-imaging validation
- Expanded architectural ablation
- Statistical comparison across repeated runs
- Calibration analysis
- Additional explainability methods
- Model-efficiency benchmarking
- Packaged inference API
- Modularization of the research codebase
- Automated testing and CI
- Pretrained checkpoint release

---

## Citation

If you use Med-AGCNet in academic work, please cite the repository using the included:

```text
CITATION.cff
```

Research-paper citation information will be updated after formal publication.

Current citation placeholder:

```bibtex
@article{medagcnet2026,
  title   = {Med-AGCNet: Adaptive Global Context Network for Medical Image Classification},
  author  = {Mahmoudian, Amir Mohammad},
  journal = {To be updated},
  year    = {2026}
}
```

---

## Disclaimer

This repository is intended for **research and educational purposes only**.

Med-AGCNet is not a certified medical device and must not be used directly for:

- Clinical diagnosis
- Treatment decisions
- Patient management
- Clinical triage

without appropriate clinical validation, regulatory approval, and expert medical oversight.

---

## Author

**Amir Mohammad Mahmoudian**

AI Engineer focused on deep learning, computer vision, applied AI, and machine-learning systems.

- [GitHub](https://github.com/mahmmooudian)
- [LinkedIn](https://www.linkedin.com/in/amirmohmmadmahmoudian)

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<div align="center">

### Deep Learning × Global Context × Explainable Medical AI

If this repository supports your research or work, consider giving it a ⭐.

</div>
