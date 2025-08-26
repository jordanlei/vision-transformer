import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Set up transforms for CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load CIFAR10 training dataset
    print("Downloading CIFAR10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # CIFAR10 class names
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Create a 4x5 grid of random images
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('CIFAR10 Dataset Examples', fontsize=16, fontweight='bold')
    
    # Get 20 random indices
    indices = np.random.choice(len(trainset), 20, replace=False)
    
    for i, idx in enumerate(indices):
        # Get image and label
        img, label = trainset[idx]
        
        # Convert from normalized tensor back to displayable format
        img = img / 2 + 0.5  # Unnormalize
        img = img.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        
        # Plot in the grid
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{classes[label]}', fontsize=10)
        axes[row, col].axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed 20 random examples from CIFAR10 dataset")
    print(f"Dataset size: {len(trainset)} training images")
    print(f"Image shape: {img.shape}")
    print(f"Classes: {classes}")

if __name__ == "__main__":
    main()
