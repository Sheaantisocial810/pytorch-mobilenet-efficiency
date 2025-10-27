"""
Script 1: Train MobileNetV1 on CIFAR-10.
"""
import torch
import logging
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import train_model, get_accuracy
from src.utils.plotting import plot_losses
from src.models.mobilenet_v1 import MobileNet

def parse_args():
    parser = argparse.ArgumentParser(description="Script 1: Train MobileNetV1 on CIFAR-10")
    parser.add_argument('--log_file', type=str, default="run_01_mobilenet_v1.log", help="Log file name")
    parser.add_argument('--model_save_path', type=str, default="models/mobilenet_v1_cifar10.pt", help="Path to save the best model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and validation")
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logging(args.log_file)
    logger.info("--- Starting Run 01: MobileNetV1 Training ---")
    logger.info(f"Using arguments: {args}")
    logger.info(f"Using device: {DEVICE}")

    try:
        # 1. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

        # 2. Initialize Model
        model = MobileNet(n_class=10).to(DEVICE)
        logger.info(f"Model MobileNetV1 initialized. Parameters: {sum(p.numel() for p in model.parameters())}")

        # 3. Train Model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, DEVICE, args.epochs, args.lr, args.model_save_path
        )

        # 4. Plot Losses
        plot_losses(train_losses, val_losses, 
                    title="MobileNetV1 CIFAR-10 Training", 
                    save_name="run_01_mobilenet_v1_loss.png")

        # 5. Load best model and evaluate
        logger.info("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(args.model_save_path))
        get_accuracy(model, val_loader, DEVICE)

        logger.info("--- Run 01: MobileNetV1 Training Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 01: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
