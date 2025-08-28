import torch
import torch.nn as nn
from network import FeedForward, CNN, VisionTransformer
from runner import Runner
import torchvision
import torchvision.transforms as transforms
from utils import write_to_gif
import shutil
import os

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def main(): 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load CIFAR10 training dataset
    print("Downloading CIFAR10 dataset...")
    train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )

    train, val = torch.utils.data.random_split(train, [0.8, 0.2])

    test = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False)
    
    model = VisionTransformer(
        img_size = 32,
        hidden_size=64,
        output_size=10,
        num_heads=4,
        num_blocks=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    runner = Runner(model, optimizer, criterion, device, plot_path="temp")
    runner.train(train_loader, val_loader, epochs=20)
    write_to_gif("temp")

    # Clean up temporary files
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        print("Cleaned up temporary files")


    test_loss, test_accuracy = runner.test(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

    runner.save(f'saved/{model.__class__.__name__}.pt')

if __name__ == "__main__":
    main()    