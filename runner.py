import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

class Runner:
    """Training/testing runner class for PyTorch models"""
    def __init__(self, model, optimizer, criterion, device):
        # Initialize runner with model, optimizer, loss function and device
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = defaultdict(list)

    def train(self, train_loader, val_loader, epochs):
        # Train model for specified number of epochs
        train_step = 0
        for epoch in range(epochs): 
            val_loss, val_accuracy = self.test(val_loader)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_accuracy'].append(val_accuracy)
            self.metrics['val_step'].append(train_step)
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
            
            for _, (x, y) in progress_bar:
                # Move data to device and get predictions
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()

                # Optimization step
                self.optimizer.zero_grad()
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                self.metrics['train_loss'].append(loss.item())
                self.metrics['train_accuracy'].append(accuracy.item())
                self.metrics['train_step'].append(train_step)
                train_step += 1
                progress_bar.set_description(f"Epoch {epoch}, Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.4f}, Train Loss {loss.item():.4f}, Train Accuracy {accuracy.item():.4f}")
        
        # Final validation step
        final_loss, final_accuracy = self.test(val_loader)
        self.metrics['val_loss'].append(final_loss)
        self.metrics['val_accuracy'].append(final_accuracy)
        self.metrics['val_step'].append(train_step)

    def test(self, test_loader):
        # Evaluate model on test set
        self.model.eval()
        loss = 0
        accuracy = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss += self.criterion(y_pred, y).item()
                accuracy += (y_pred.argmax(dim=1) == y).float().mean().item()

        return loss / len(test_loader), accuracy / len(test_loader)

    def save(self, path):
        # Save model, optimizer, criterion state and metrics
        metadata = {
            'model': {
                'class': self.model.__class__.__name__,
                'dict': self.model.state_dict()
            },
            'optimizer': {
                'class': self.optimizer.__class__.__name__,
                'dict': self.optimizer.state_dict()
            },
            'criterion': {
                'class': self.criterion.__class__.__name__,
                'dict': self.criterion.state_dict()
            },
            'device': self.device,
            'metrics': self.metrics
        }
        torch.save(metadata, path)

    @classmethod
    def load(cls, path):
        # Load saved model state and recreate runner
        metadata = torch.load(path)     
        model = metadata['model']['class'].load_state_dict(metadata['model']['dict'])
        optimizer = metadata['optimizer']['class'].load_state_dict(metadata['optimizer']['dict'])
        criterion = metadata['criterion']['class'].load_state_dict(metadata['criterion']['dict'])
        runner = cls(model, optimizer, criterion, metadata['device'])
        return runner

    def plot(self, save_path=None):
        """Plot training and validation metrics"""
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training and validation metrics
        axs[0].plot(self.metrics['train_loss'], label='Train Loss')
        axs[0].plot(self.metrics['val_loss'], label='Validation Loss')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].legend()
        
        axs[1].plot(self.metrics['train_accuracy'], label='Train Accuracy')
        axs[1].plot(self.metrics['val_accuracy'], label='Validation Accuracy')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        




