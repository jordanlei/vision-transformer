import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

class Runner:
    """Training/testing runner class for PyTorch models"""
    def __init__(self, model, optimizer, criterion, device, plot_path = None):
        # Initialize runner with model, optimizer, loss function and device
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = defaultdict(list)
        self.plot_path = plot_path

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

                if train_step % 100 == 0:
                    progress_bar.set_description(f"Epoch {epoch}, Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.4f}, Train Loss {loss.item():.4f}, Train Accuracy {accuracy.item():.4f}")
                    if self.plot_path:
                        self.plot(val_loader, save_path=f'{self.plot_path}/{self.model.__class__.__name__}_{train_step:08d}.png')   

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
        self.model.save(path)

    def plot(self, val_loader = None, save_path=None):
        """Plot training and validation metrics"""
        fig = plt.figure(figsize=(8, 10))
        
        # Create subplot mosaic layout
        if val_loader is not None:
            mosaic = """
            AB
            CC
            """
            height_ratios = [0.8, 1]
        else:
            mosaic = "AB"
            height_ratios = [1, 1]
            
        ax_dict = fig.subplot_mosaic(mosaic, height_ratios=height_ratios)
        
        # Plot training and validation metrics
        ax_dict['A'].plot(self.metrics['train_step'], self.metrics['train_loss'], label='Train Loss')
        ax_dict['A'].plot(self.metrics['val_step'], self.metrics['val_loss'], label='Validation Loss')
        ax_dict['A'].set_xlabel('Step')
        ax_dict['A'].set_ylabel('Loss')
        ax_dict['A'].set_title('Training and Validation Loss')
        ax_dict['A'].legend()
        
        ax_dict['B'].plot(self.metrics['train_step'], self.metrics['train_accuracy'], label='Train Accuracy')
        ax_dict['B'].plot(self.metrics['val_step'], self.metrics['val_accuracy'], label='Validation Accuracy')
        ax_dict['B'].set_xlabel('Step')
        ax_dict['B'].set_ylabel('Accuracy')
        ax_dict['B'].set_title('Training and Validation Accuracy')
        ax_dict['B'].legend()
        plt.subplots_adjust(hspace=0.4)

        if val_loader is not None:
            # Get a batch of validation images
            images, labels = next(iter(val_loader))
            images = images.to(self.device)
            
            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            # Create grid for sample images in bottom panel
            gs = ax_dict['C'].get_gridspec()
            ax_dict['C'].remove()
            axs = gs[1, 0:].subgridspec(2, 3, hspace=0.5).subplots()
            
            # Plot first 6 images with their predictions
            classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')
            
            for i, ax in enumerate(axs.flat[:6]):
                # Display image
                img = images[i].cpu() / 2 + 0.5  # Unnormalize
                img = img.permute(1, 2, 0)  # Convert from (C,H,W) to (H,W,C)
                ax.imshow(img)
                ax.axis('off')
                
                # Add predictions as title
                actual_label = classes[labels[i]]
                predictions = [classes[idx] for idx in top3_idx[i]]
                title = f"actual: {actual_label}\n{predictions[0]} ({top3_prob[i][0]*100:.1f}%)\n{predictions[1]} ({top3_prob[i][1]*100:.1f}%)\n{predictions[2]} ({top3_prob[i][2]*100:.1f}%)"
                ax.set_title(title, fontsize=8, pad=10)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        




