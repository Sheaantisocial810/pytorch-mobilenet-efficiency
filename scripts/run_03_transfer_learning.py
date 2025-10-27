"""
Script 3: Fine-tune MobileNetV1 (from Script 1) on CIFAR-100.
"""
import torch
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar100_loaders
from src.utils.training import train_model, get_accuracy
from src.utils.plotting import plot_losses
from src.models.mobilenet_v1 import MobileNet

# --- Configuration ---
LOG_FILE = "run_03_transfer_learning.log"
PRETRAINED_MODEL_PATH = "models/mobilenet_v1_cifar10.pt"
MODEL_SAVE_PATH = "models/mobilenet_v1_cifar100.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5  # Fine-tuning epochs
LR = 1e-4   # Lower LR for fine-tuning
BATCH_SIZE = 128
# --- End Configuration ---

def main():
    logger = setup_logging(LOG_FILE)
    logger.info("--- Starting Run 03: Transfer Learning (CIFAR-10 to CIFAR-100) ---")

    try:
        # 1. Check if pretrained model exists
        if not Path(PRETRAINED_MODEL_PATH).exists():
            logger.error(f"Pretrained model not found at {PRETRAINED_MODEL_PATH}.")
            logger.error("Please run 'scripts/run_01_mobilenet_v1.py' first.")
            sys.exit(1)

        # 2. Load Data (CIFAR-100)
        train_loader, val_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

        # 3. Initialize Model and load weights
        model = MobileNet(n_class=100).to(DEVICE) # Initialize with 100 classes
        
        pretrained_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        model_dict = model.state_dict()

        # Filter out FC layer weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info("Loaded pre-trained weights from CIFAR-10 model.")

        # 4. Freeze layers
        logger.info("Freezing initial 7 convolutional blocks...")
        for i, layer in enumerate(model.model[:7]): # Freeze first 7 blocks
            for param in layer.parameters():
                param.requires_grad = False
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters after freezing: {trainable_params}")

        # 5. Train Model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, DEVICE, EPOCHS, LR, MODEL_SAVE_PATH
        )

        # 6. Plot Losses
        plot_losses(train_losses, val_losses, 
                    title="MobileNetV1 Fine-Tuning on CIFAR-100", 
                    save_name="run_03_transfer_learning_loss.png")

        # 7. Load best model and evaluate
        logger.info("Loading best fine-tuned model for final evaluation...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        get_accuracy(model, val_loader, DEVICE)

        logger.info("--- Run 03: Transfer Learning Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 03: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
