"""
Script 5: Analyze MobileNetV2 width_multiplier and train a wm=0.5 model.
This script fulfills the requirement from [CELL 59-60] to train the 
wm=0.5 model instead of loading a pre-trained one.
"""
import torch
import logging
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import train_model, get_accuracy
from src.models.mobilenet_v2 import MobileNetV2

# --- Configuration ---
LOG_FILE = "run_05_mobilenet_v2_hyperparams.log"
MODEL_SAVE_PATH = "models/mobilenet_v2_wm0.5_cifar10.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 15  # As per notebook description [CELL 59]
LR = 1e-3
BATCH_SIZE = 128
WIDTH_MULTIPLIER = 0.5
# --- End Configuration ---

def analyze_parameter_counts(logger):
    """Logs the parameter counts for different width multipliers."""
    logger.info("--- Analyzing Width Multiplier vs. Parameter Count ---")
    for i in range(1, 11):
        wm = i / 10.0
        model = MobileNetV2(n_class=10, width_multiplier=wm)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Width Multiplier: {wm:<3.1f} | Parameters: {total_params}")
    logger.info("--- End of Parameter Analysis ---")

def main():
    logger = setup_logging(LOG_FILE)
    logger.info(f"--- Starting Run 05: MobileNetV2 Hyperparameter Analysis (WM={WIDTH_MULTIPLIER}) ---")

    try:
        # 1. Analyze Parameter Counts
        analyze_parameter_counts(logger)

        # 2. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

        # 3. Initialize Model (WM=0.5)
        model = MobileNetV2(n_class=10, width_multiplier=WIDTH_MULTIPLIER).to(DEVICE)
        logger.info(f"Model MobileNetV2 initialized. WM={WIDTH_MULTIPLIER}, "
                    f"Parameters: {sum(p.numel() for p in model.parameters())}")

        # 4. Train Model
        logger.info(f"Training model for {EPOCHS} epochs (as per notebook [CELL 59])...")
        train_model(
            model, train_loader, val_loader, DEVICE, EPOCHS, LR, MODEL_SAVE_PATH
        )

        # 5. Load best model and evaluate
        logger.info(f"Loading best model (WM={WIDTH_MULTIPLIER}) for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        get_accuracy(model, val_loader, DEVICE)

        logger.info(f"--- Run 05: MobileNetV2 Hyperparameter Analysis (WM={WIDTH_MULTIPLIER}) Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 05: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
