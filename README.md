# Vision Transformer Project

A PyTorch-based computer vision project implementing both **Vision Transformers (ViT)** and **Convolutional Neural Networks (CNNs)** for image classification on the CIFAR-10 dataset.

![Training Progress Animation](animation_vit.gif)

*Training progress visualization showing the convergence of our Vision Transformer over 10 epochs*

![Visualization](figures/attention_maps.png)

*Visualization of learned attention maps after training*

## Table of Contents

- [Analysis](#analysis)
  - [Model Comparison: Patch Embedding Size](#model-comparison-patch-embedding-size)
    - [Quantitative Results](#quantitative-results)
    - [Training Curves](#training-curves)
    - [Takeaways](#takeaways)
  - [Project Overview](#project-overview)
- [Code](#code)
  - [Project Structure](#project-structure)
  - [Features](#features)
    - [Neural Network Architectures](#-neural-network-architectures)
    - [Training Framework](#-training-framework)
    - [Data Management](#-data-management)
    - [Visualization & Training Progress](#-visualization--training-progress)
    - [Interactive Tutorials & Demos](#-interactive-tutorials--demos)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Training a Model](#training-a-model)
    - [Interactive Tutorials](#interactive-tutorials)
    - [Custom Training](#custom-training)
    - [Using Vision Transformer](#using-vision-transformer)
    - [Attention Visualization](#attention-visualization)
  - [Model Architecture](#model-architecture)
  - [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Analysis

### Model Comparison: Patch Embedding Size

We compared a baseline CNN against Vision Transformers (ViTs) trained from scratch on CIFAR-10 with varying patch sizes.  

### Quantitative Results
| Model           | Test Loss | Test Accuracy |
|-----------------|-----------|---------------|
| CNN             | 0.962     | **67.0%**     |
| ViT (patch=2)   | 1.272     | 63.1%         |
| ViT (patch=4)   | 1.394     | 60.1%         |
| ViT (patch=6)   | 1.675     | 56.6%         |
| ViT (patch=8)   | 1.872     | 55.3%         |

We also computed **bootstrapped accuracy differences** relative to the CNN baseline:

<p align="center">
  <img src="figures/model_comparison_bootstrap.png" width="900">
</p>

The CNN consistently outperforms all ViTs, and performance decreases steadily with larger patch sizes.

---

### Training Curves
<p align="center">
  <img src="figures/model_comparison_curves.png" width="900">
</p>

The curves highlight two trends:
- The CNN converges faster and reaches higher validation accuracy.
- Smaller patch sizes (e.g., 2×2) help ViTs retain more local detail and improve performance relative to larger patches, but they still lag behind CNNs.

---

### Takeaways
- **Inductive biases matter:** On small datasets like CIFAR-10, CNNs have strong locality and translation equivariance built in, giving them an advantage over ViTs.  
- **ViTs need more data or pretraining:** In low-data regimes, ViTs trained from scratch underperform CNNs, but with sufficient scale or augmentations they can surpass CNNs.  
- **Patch size tradeoff:** Smaller patches preserve detail and perform better, while larger patches discard local structure and hurt accuracy.  

In short: on CIFAR-10, a simple CNN beats a ViT trained from scratch, but the ViT results align with known trends from the literature.

### Project Overview

This project provides a complete training pipeline for image classification using PyTorch, featuring:
- **CNN Architecture**: A custom convolutional neural network designed for CIFAR-10
- **Vision Transformer (ViT)**: A fully functional transformer-based architecture with attention mechanisms
- **Training Framework**: A robust `Runner` class that handles training, validation, and testing
- **Data Management**: Automatic CIFAR-10 dataset download and preprocessing
- **Visualization**: Built-in plotting capabilities with animated training progress GIFs and attention map visualization
- **Model Persistence**: Save and load functionality for trained models
- **Interactive Demos**: Jupyter notebooks for tutorials and attention visualization

---

## Code

### Project Structure

```
vision-transformer/
├── network.py          # Neural network architectures (CNN, FeedForward, ViT)
├── runner.py           # Training orchestration and metrics tracking
├── train.py            # Main training script (updated for 20 epochs)
├── setup_cifar.py      # CIFAR-10 dataset setup utilities
├── utils.py            # Utility functions for GIF generation
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── data/               # CIFAR-10 dataset storage
│   └── cifar-10-batches-py/  # CIFAR-10 data files
├── demos/              # Interactive tutorials and demonstrations
│   ├── vit_tutorial.ipynb     # Vision Transformer code explanation
│   └── attention_maps.ipynb   # Attention visualization tutorial
├── figures/            # Generated plots and visualizations
│   └── attention_maps.png     # Example attention map visualization
├── saved/              # Trained model checkpoints
│   └── VisionTransformer.pt   # Pre-trained ViT model
├── animation_vit.gif   # Training progress animation
└── .venv/              # Virtual environment (excluded from git)
```

### Features

- **Neural Network Architectures**: CNN, FeedForward, and Vision Transformer (ViT) implementations
- **Training Framework**: Complete training pipeline with metrics tracking and visualization
- **Data Management**: Automatic CIFAR-10 dataset download and preprocessing
- **Visualization**: Training progress animations and attention map visualization
- **Interactive Demos**: Jupyter notebooks for tutorials and model exploration

### Installation

#### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Apple Silicon Mac (for MPS acceleration) or CUDA-capable GPU
- Jupyter Notebook (for interactive demos)

#### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vision-transformer
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter for interactive demos**
   ```bash
   pip install jupyter
   ```

### Usage

#### Training a Model

Run the main training script:
```bash
python train.py
```

This will:
1. Download CIFAR-10 dataset (if not already present)
2. Train a Vision Transformer model for 20 epochs
3. Display training progress with live metrics
4. Generate training plots and animated GIF
5. Save the trained model as `VisionTransformer.pt`
6. Report final test performance

#### Interactive Tutorials

**Start Jupyter Notebook server:**
```bash
cd demos
jupyter notebook
```

**Available Tutorials:**
- **ViT Tutorial**: Learn about Vision Transformer architecture and components
- **Attention Maps**: Visualize attention weights and understand model decisions

#### Custom Training

Modify `train.py` to:
- Change the number of epochs (currently set to 20)
- Use different models (CNN, FeedForward, or VisionTransformer)
- Adjust hyperparameters (learning rate, batch size, model architecture)
- Change the validation split ratio

#### Using Vision Transformer

To train with the Vision Transformer architecture:

```python
from network import VisionTransformer

# Create ViT model with custom parameters
model = VisionTransformer(
    img_size=32,        # CIFAR-10 image size
    hidden_size=64,     # Embedding dimension (updated from 128)
    output_size=10,     # Number of classes
    num_heads=4,        # Number of attention heads (updated from 8)
    num_blocks=4        # Number of transformer blocks (updated from 6)
)

# Use with existing training pipeline
runner = Runner(model, optimizer, criterion, device)
runner.train(train_loader, val_loader, epochs=20)  # Updated to 20 epochs
```

#### Attention Visualization

After training a model, use the attention maps tutorial:

```python
# Load trained model
model = VisionTransformer.load("saved/VisionTransformer.pt")

# Extract attention weights
with torch.no_grad():
    # Forward pass to get attention weights
    output = model(images, return_attention=True)
    attention_weights = output['attention_weights']
    
# Visualize attention maps
visualize_attention_maps(images, attention_weights)
```

### Model Architecture

**CNN Architecture Details:**
```
Input: 3x32x32 (RGB image)
├── Conv2d(3→16, kernel=3x3, padding=1) + ReLU
├── MaxPool2d(2x2)
├── Conv2d(16→32, kernel=3x3, padding=1) + ReLU
├── MaxPool2d(2x2)
├── Flatten: 32×8×8 → 2048
└── Linear(2048→10) → Output
```

**Vision Transformer (ViT) Architecture Details:**
```
Input: 3x32x32 (RGB image)
├── Patch Embedding:
│   ├── Conv2d(3→64, kernel=4x4, stride=4) → 8x8 patches
│   ├── Flatten patches → 64 patches × 64 dimensions
│   ├── Add CLS token → 65 patches × 64 dimensions
│   └── Add positional embeddings
├── Transformer Blocks (4 blocks):
│   ├── LayerNorm + Multi-Head Attention (4 heads)
│   ├── Residual connection
│   ├── LayerNorm + Feed-Forward Network (GELU activation)
│   └── Residual connection
├── Extract CLS token representation
└── Linear(64→10) → Output
```

**Key ViT Components:**
- **Patch Embedding**: Divides 32×32 images into 4×4 patches (64 patches total)
- **CLS Token**: Learnable classification token prepended to patch sequence
- **Positional Embeddings**: Learnable positional information for each patch + CLS token
- **Multi-Head Attention**: 4 attention heads for different feature aspects
- **Transformer Blocks**: Stack of 4 self-attention and feed-forward layers with residual connections
- **Layer Normalization**: Stabilizes training and improves convergence
- **GELU Activation**: Smooth activation function used in modern transformers

### Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework
- **TorchVision** (≥0.15.0): Computer vision utilities
- **Matplotlib** (≥3.5.0): Plotting and visualization
- **NumPy** (≥1.21.0): Numerical computing
- **Pillow** (≥8.0.0): Image processing
- **Jupyter**: Interactive notebook environment
- **tqdm**: Progress bars (installed separately)

### License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Jordan Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Acknowledgments

- CIFAR-10 dataset creators
- PyTorch development team
- Apple Silicon MPS support
- Jupyter project contributors

---

**Last Updated**: December 2024
**Status**: Active Development
**Version**: 1.1.0
