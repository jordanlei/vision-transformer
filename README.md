# Vision Transformer Project

A PyTorch-based computer vision project implementing convolutional neural networks (CNNs) for image classification on the CIFAR-10 dataset.

## Project Overview

This project provides a complete training pipeline for image classification using PyTorch, featuring:
- **CNN Architecture**: A custom convolutional neural network designed for CIFAR-10
- **Training Framework**: A robust `Runner` class that handles training, validation, and testing
- **Data Management**: Automatic CIFAR-10 dataset download and preprocessing
- **Visualization**: Built-in plotting capabilities for training metrics
- **Model Persistence**: Save and load functionality for trained models

## Project Structure

```
vision-transformer/
├── network.py          # Neural network architectures (CNN, FeedForward)
├── runner.py           # Training orchestration and metrics tracking
├── train.py            # Main training script
├── setup_cifar.py      # CIFAR-10 dataset setup utilities
├── utils.py            # Utility functions (currently empty)
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── data/               # CIFAR-10 dataset storage
├── figures/            # Generated plots and visualizations
└── .venv/              # Virtual environment (excluded from git)
```

## Features

### 🧠 Neural Network Architectures

**CNN (Convolutional Neural Network)**
- 2 convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Fully connected output layer for 10-class classification
- Optimized for CIFAR-10 (32x32x3 images)

**FeedForward Network**
- Simple multi-layer perceptron
- Configurable input, hidden, and output sizes
- ReLU activation function

### 🚀 Training Framework

**Runner Class**
- Comprehensive training loop with progress bars
- Real-time metrics tracking (loss, accuracy)
- Validation during training
- Automatic device detection (MPS/GPU/CPU)
- Built-in plotting and visualization
- Model checkpointing and restoration

**Key Features**
- Automatic device selection (Apple Silicon MPS, CUDA, or CPU)
- Progress bars with live metrics display
- Comprehensive metrics logging
- Validation at each epoch
- Final test evaluation

### 📊 Data Management

**CIFAR-10 Dataset**
- Automatic download and setup
- Train/validation split (80/20)
- Data augmentation with normalization
- Batch processing with configurable batch sizes

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Apple Silicon Mac (for MPS acceleration) or CUDA-capable GPU

### Setup

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

## Usage

### Training a Model

Run the main training script:
```bash
python train.py
```

This will:
1. Download CIFAR-10 dataset (if not already present)
2. Train a CNN model for 1 epoch
3. Display training progress with live metrics
4. Generate training plots
5. Save the trained model as `CNN.pth`
6. Report final test performance

### Custom Training

Modify `train.py` to:
- Change the number of epochs
- Use different models (CNN or FeedForward)
- Adjust hyperparameters (learning rate, batch size)
- Change the validation split ratio

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

## Performance

The current CNN implementation achieves:
- **Training Accuracy**: ~60% (after 1 epoch)
- **Test Accuracy**: ~52% (after 1 epoch)
- **Training Time**: ~20 seconds per epoch on Apple Silicon M1

*Note: Performance improves significantly with more training epochs*

## Dependencies

- **PyTorch** (≥2.0.0): Deep learning framework
- **TorchVision** (≥0.15.0): Computer vision utilities
- **Matplotlib** (≥3.5.0): Plotting and visualization
- **NumPy** (≥1.21.0): Numerical computing
- **Pillow** (≥8.0.0): Image processing
- **tqdm**: Progress bars (installed separately)

## Development Status

### ✅ Completed
- CNN architecture implementation
- Training framework with metrics tracking
- CIFAR-10 data pipeline
- Model saving/loading
- Training visualization
- Virtual environment setup
- Comprehensive .gitignore

### 🔄 In Progress
- Model performance optimization
- Additional architectures (Vision Transformer planned)

### 📋 Planned Features
- Vision Transformer implementation
- Advanced data augmentation
- Learning rate scheduling
- Early stopping
- Cross-validation
- Model ensemble methods
- Transfer learning support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- CIFAR-10 dataset creators
- PyTorch development team
- Apple Silicon MPS support

---

**Last Updated**: August 2024
**Status**: Active Development
**Version**: 1.0.0
