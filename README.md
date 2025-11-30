# Exomind: Spurious Correlation Removal via Sidecar Injection

This project implements a system to identify and remove spurious correlations in medical image classification using the Phikon foundation model and PatchCamelyon dataset.

## Overview

The system uses Partial Least Squares (PLS) analysis to identify directions in the feature space that correspond to spurious correlations (features that lead to confident but incorrect predictions). A "sidecar" model is then trained to adjust embeddings to reduce reliance on these spurious features.

## Project Structure

```
exomind/
├── utils/                      # Utility modules
│   ├── feature_extraction.py  # Extract features from Phikon
│   └── prediction_head.py     # MLP classifier training
├── PLS/                        # PLS analysis modules
│   ├── loss_pls.py            # PLS on prediction loss
│   ├── confidence_pls.py      # PLS on model confidence (entropy)
│   └── viz.py                 # Visualization tools
├── injector/                   # Sidecar injection system
│   └── sidecar.py             # ResNet18-based sidecar model
├── scripts/                    # Executable scripts
│   ├── run_analysis.py        # Run PLS analysis per class
│   └── train_injector.py      # Train sidecar to remove spurious features
├── figures/                    # Generated visualizations
└── models/                     # Saved model checkpoints
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data:**
    The PatchCamelyon dataset will be automatically downloaded on first run to `./data/`.

## Usage

### 1. PLS Analysis (Per-Class)

Analyze loss and confidence components for each class separately:

```bash
python scripts/run_analysis.py
```

**Outputs:**
- Component similarity heatmaps
- Top images for each component
- Predicted vs. actual loss/confidence plots
- Files saved to `figures/` with class-specific naming

### 2. Sidecar Training (Spurious Feature Removal)

Train a sidecar model to reduce sensitivity to spurious directions:

```bash
python scripts/train_injector.py
```

**Interactive Steps:**
1. Loads Phikon and trains a prediction head
2. Computes PLS components (Loss and Confidence)
3. Displays similarity matrix
4. **Prompts you to select** a spurious direction (L_i, C_j pair)
5. Trains two sidecar models:
   - **Without invariance penalty** (baseline sidecar)
   - **With invariance penalty** (spurious feature removal)
6. Compares performance and sensitivity

**Outputs:**
- Validation loss, accuracy, and sensitivity metrics
- Best models saved to `models/`
- Summary comparison table

## Key Concepts

### PLS Components
- **Loss Components**: Directions in feature space explaining variance in prediction error
- **Confidence Components**: Directions explaining variance in model confidence (negative entropy)
- **Positive Correlation**: High loss + high confidence = "Confidently Wrong" (spurious feature)

### Sidecar Injection
- A ResNet18 model that outputs 768-dim vectors (matching Phikon)
- Added to Phikon embeddings: `h_combined = h_phikon + h_sidecar`
- Trained with invariance penalty: `MSE(Head(h), Head(h - λv))` to ignore spurious direction `v`

### Sensitivity Metric
- Average L2 change in logits when perturbing embeddings along direction `v`
- Lower sensitivity = model is more invariant to the spurious feature

## Hyperparameters

Edit `scripts/train_injector.py` to tune:
- `num_samples`: Training set size (default: 10000)
- `val_samples`: Validation set size (default: 10000)
- `epochs`: Training epochs (default: 5)
- `lambda_val`: Perturbation strength (default: 1.0)
- `alpha_penalty`: Invariance penalty weight (default: 1.0)

## Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- torchvision
- scikit-learn
- matplotlib
- tqdm

See `requirements.txt` for full list.

## Citation

This project uses:
- **Phikon**: Owkin's foundation model for histopathology
- **PatchCamelyon**: Binary classification dataset for metastasis detection
