"""
Core training and evaluation loops.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def fit_epoch(model, data_loader, criterion, optimizer, device, train_mode=False):
    """
    Processes one epoch of training or validation.
    
    Args:
        model (nn.Module): The PyTorch model.
        data_loader (DataLoader): The data loader.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device (cpu or cuda).
        train_mode (bool): Whether this is a training pass (to enable grads and optimizer).
        
    Returns:
        tuple: (list of batch losses, average epoch loss, average epoch accuracy)
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_losses = []
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    pbar = tqdm(data_loader, desc="Training" if train_mode else "Validation")
    for inputs, labels in pbar:
        try:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            if train_mode:
                optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(train_mode):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass + optimize only if in training mode
                if train_mode:
                    loss.backward()
                    optimizer.step()
                
                preds = torch.argmax(outputs, 1)

            # Statistics
            epoch_losses.append(loss.item())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)

            pbar.set_postfix(
                loss=f"{running_loss/processed_data:.4f}",
                acc=f"{running_corrects.cpu().numpy()/processed_data:.4f}"
            )

        except Exception as e:
            logger.error(f"Error during {'training' if train_mode else 'validation'} step: {e}")
            continue # Skip batch on error

    epoch_loss = running_loss / processed_data
    epoch_acc = running_corrects.cpu().numpy() / processed_data

    if train_mode:
        logger.info(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    else:
        logger.info(f"  Val Loss: {epoch_loss:.4f} |   Val Acc: {epoch_acc:.4f}")

    return epoch_losses, epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, device, epochs, lr, model_save_path):
    """
    Full training loop for a model.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): The device (cpu or cuda).
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        model_save_path (str): Path to save the best model.
        
    Returns:
        tuple: (all_train_losses, all_val_losses)
    """
    all_train_losses, all_val_losses = [], []
    best_val_loss = float('inf')

    # Ensure model save directory exists
    model_dir = Path(model_save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Device: {device}, LR: {lr}")
    
    start_time = time.time()
    try:
        for epoch in range(epochs):
            logger.info(f"--- EPOCH {epoch+1}/{epochs} ---")
            
            # Training epoch
            train_batch_losses, _, _ = fit_epoch(
                model, train_loader, criterion, optimizer, device, train_mode=True
            )
            all_train_losses.extend(train_batch_losses)
            
            # Validation epoch
            _, val_loss, _ = fit_epoch(
                model, val_loader, criterion, optimizer, device, train_mode=False
            )
            all_val_losses.append(val_loss)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"New best model saved to {model_save_path} (Val Loss: {val_loss:.4f})")
                
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
    
    end_time = time.time()
    logger.info(f"Training complete. Total time: {(end_time - start_time) / 60:.2f} minutes.")
    
    return all_train_losses, all_val_losses


def get_accuracy(model, data_loader, device):
    """
    Calculates the accuracy of a model on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader.
        device (torch.device): The device (cpu or cuda).
        
    Returns:
        float: The accuracy.
    """
    model.to(device)
    model.eval()
    
    running_corrects = 0
    processed_data = 0
    
    logger.info(f"Calculating accuracy on {len(data_loader.dataset)} samples...")
    start_time = time.time()

    with torch.set_grad_enabled(False):
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                processed_data += inputs.size(0)
            except Exception as e:
                logger.error(f"Error during evaluation step: {e}")
                continue

    end_time = time.time()
    epoch_acc = running_corrects.cpu().numpy() / processed_data
    
    logger.info(f"Validation Accuracy: {epoch_acc:.4f}")
    logger.info(f"Validation Time: {end_time - start_time:.4f} s")
    return epoch_acc
