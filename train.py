import torch
import torch.nn as nn
from network import FeedForward, CNN, VisionTransformer
from runner import Runner
import torchvision
import torchvision.transforms as transforms
from utils import write_to_gif
import shutil
import os
import argparse

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit', 'mlp'],
                      help='Model architecture to use (cnn, vit, or mlp)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_heads', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=10, help='Number of transformer blocks')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size')
    args = parser.parse_args()
    
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
    
    if args.model == 'vit':
        model = VisionTransformer(
            img_size = 32,
            hidden_size=args.hidden_size,
            output_size=10,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            patch_size=args.patch_size
        )
    elif args.model == 'mlp':
        model = FeedForward()
    elif args.model == 'cnn':
        model = CNN()
    else:
        raise ValueError(f"Invalid model: {args.model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    runner = Runner(model, optimizer, criterion, device)
    runner.train(train_loader, val_loader, epochs=20)
    write_to_gif("temp")

    # Clean up temporary files
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        print("Cleaned up temporary files")

    test_loss, test_accuracy = runner.test(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

    if args.model == 'vit':
        runner.save_model(f'saved/{args.model}_hiddensize{args.hidden_size}_numheads{args.num_heads}_numblocks{args.num_blocks}_patchsize{args.patch_size}.pt')
        runner.save(f'saved/{args.model}_hiddensize{args.hidden_size}_numheads{args.num_heads}_numblocks{args.num_blocks}_patchsize{args.patch_size}.pkl')
    else:
        runner.save_model(f'saved/{args.model}.pt')
        runner.save(f'saved/{args.model}.pkl')

if __name__ == "__main__":
    main()    