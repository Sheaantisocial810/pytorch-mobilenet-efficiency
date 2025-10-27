"""
Utility functions for plotting results.
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def imshow_grid(data_loader, mean, std, title=""):
    """
    Shows a grid of augmented images from a data loader.
    
    Args:
        data_loader: The torch.utils.data.DataLoader.
        mean (list): The mean used for normalization.
        std (list): The std dev used for normalization.
        title (str): Title for the plot.
    """
    try:
        dataiter = iter(data_loader)
        images, labels = next(dataiter)
        
        # Make a grid and display
        img_grid = torchvision.utils.make_grid(images[:32])
        
        mean_arr = np.array(mean)
        std_arr = np.array(std)
        
        img = img_grid.numpy().transpose((1, 2, 0))
        img = img * std_arr + mean_arr
        img = np.clip(img, 0, 1)
        
        plt.figure(figsize=(15, 5))
        plt.imshow(img)
        plt.title(title)
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        save_path = log_dir / f"{title.lower().replace(' ', '_')}_batch_preview.png"
        plt.savefig(save_path)
        logger.info(f"Saved data preview image to {save_path}")
        plt.show()

    except Exception as e:
        logger.warning(f"Could not display image grid: {e}")


def plot_losses(train_losses, val_losses, title="Training and Validation Loss", save_name="loss_plot.png"):
    """
    Plots training and validation losses.
    
    Args:
        train_losses (list): List of training loss values (per-batch).
        val_losses (list): List of validation loss values (per-epoch).
        title (str): Title for the plot.
        save_name (str): Filename to save the plot as in the /logs directory.
    """
    try:
        c_window = 10
        epochs = len(val_losses)
        train_losses_c = np.convolve(np.array(train_losses),
                                     np.ones(c_window)/c_window, mode='valid')

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses_c, label='Smoothed Train Loss')
        plt.plot([i * len(train_losses) / epochs for i in range(1, epochs + 1)],
                 np.array(val_losses),
                 label='Validation Loss', marker='o', markersize=8)
        
        plt.title(title)
        plt.xlabel("Training Batches")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        save_path = log_dir / save_name
        plt.savefig(save_path)
        logger.info(f"Saved loss plot to {save_path}")
        plt.show()
        
    except Exception as e:
        logger.warning(f"Could not plot losses: {e}")
