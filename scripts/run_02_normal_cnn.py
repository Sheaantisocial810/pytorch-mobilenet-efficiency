"""
Script 2: Train NormalCNN on CIFAR-10 for comparison.
"""
import torch
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import train_model, get_accuracy
from src.models.normal_cnn import NormalCNN

# --- Configuration ---
LOG_FILE = "run_02_normal_cnn.log"
MODEL_SAVE_PATH = "models/normal_cnn_cifar10.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 2  # As per notebook, just for speed comparison
LR = 1e-3
BATCH_SIZE = 128
# --- End Configuration ---

def main():
    logger = setup_logging(LOG_FILE)
    logger.info("--- Starting Run 02: NormalCNN Training Comparison ---")

    try:
        # 1. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

        # 2. Initialize Model
        model = NormalCNN(n_class=10).to(DEVICE)
        logger.info(f"Model NormalCNN initialized. Parameters: {sum(p.numel() for p in model.parameters())}")

        # 3. Train Model
        train_model(
            model, train_loader, val_loader, DEVICE, EPOCHS, LR, MODEL_SAVE_PATH
        )

        # 4. Load best model and evaluate
        logger.info("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        get_accuracy(model, val_loader, DEVICE)

        logger.info("--- Run 02: NormalCNN Training Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 02: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
