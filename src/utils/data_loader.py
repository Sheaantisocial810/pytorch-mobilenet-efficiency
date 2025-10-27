"""
Handles loading and transforming CIFAR-10 and CIFAR-100 datasets.
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Re-usable transforms
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761]),
])


def get_cifar10_loaders(batch_size: int = 128, data_root: str = './data', train_subset_size: int = 20000):
    """
    Gets CIFAR-10 train and validation loaders.
    
    Args:
        batch_size (int): Number of samples per batch.
        data_root (str): Path to download/store data.
        train_subset_size (int): Number of training samples to use. 
                                 Set to None to use the full dataset.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    logger.info(f"Loading CIFAR-10 dataset from {data_root}")
    try:
        train_set_full = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=TRANSFORM_TRAIN)

        if train_subset_size:
            logger.info(f"Using a subset of {train_subset_size} training images.")
            indices = np.arange(train_subset_size)
            train_set = Subset(train_set_full, indices)
        else:
            logger.info("Using the full training dataset.")
            train_set = train_set_full

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        val_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=TRANSFORM_TEST)

        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        logger.info("CIFAR-10 data loaders created successfully.")
        return train_loader, val_loader

    except Exception as e:
        logger.error(f"Failed to load CIFAR-10 data: {e}")
        raise


def get_cifar100_loaders(batch_size: int = 128, data_root: str = './data', train_subset_size: int = 20000):
    """
    Gets CIFAR-100 train and validation loaders for transfer learning.
    
    Args:
        batch_size (int): Number of samples per batch.
        data_root (str): Path to download/store data.
        train_subset_size (int): Number of training samples to use.
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    logger.info(f"Loading CIFAR-100 dataset from {data_root}")
    try:
        train_set_full = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=TRANSFORM_TRAIN)

        if train_subset_size:
            logger.info(f"Using a subset of {train_subset_size} training images.")
            indices = np.arange(train_subset_size)
            train_set = Subset(train_set_full, indices)
        else:
            logger.info("Using the full training dataset.")
            train_set = train_set_full
            
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        val_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=TRANSFORM_TEST)

        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=2)
        
        logger.info("CIFAR-100 data loaders created successfully.")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Failed to load CIFAR-100 data: {e}")
        raise
