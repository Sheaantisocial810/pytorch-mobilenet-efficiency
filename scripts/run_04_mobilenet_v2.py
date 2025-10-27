"""
Script 4: Train MobileNetV2 (width_multiplier=1.0) on CIFAR-10.
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
from src.utils.plotting import plot_losses
from src.models.mobilenet_v2 import MobileNetV2

# --- Configuration ---
LOG_FILE = "run_04_mobilenet_v2.log"
MODEL_SAVE_PATH = "models/mobilenet_v2_wm1.0_cifar10.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 128
WIDTH_MULTIPLIER = 1.0
# --- End Configuration ---

def main():
    logger = setup_logging(LOG_FILE)
    logger.info(f"--- Starting Run 04: MobileNetV2 (WM={WIDTH_MULTIPLIER}) Training ---")

    try:
        # 1. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

        # 2. Initialize Model
        model = MobileNetV2(n_class=10, width_multiplier=WIDTH_MULTIPLIER).to(DEVICE)
        logger.info(f"Model MobileNetV2 initialized. WM={WIDTH_MULTIPLIER}, "
                    f"Parameters: {sum(p.numel() for p in model.parameters())}")

        # 3. Train Model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, DEVICE, EPOCHS, LR, MODEL_SAVE_PATH
        )

        # 4. Plot Losses
        plot_losses(train_losses, val_losses, 
                    title=f"MobileNetV2 (WM={WIDTH_MULTIPLIER}) CIFAR-10 Training", 
                    save_name="run_04_mobilenet_v2_loss.png")

        # 5. Load best model and evaluate
        logger.info("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        get_accuracy(model, val_loader, DEVICE)

        logger.info(f"--- Run 04: MobileNetV2 (WM={WIDTH_MULTIPLIER}) Training Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 04: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
