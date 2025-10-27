"""
Script 5: Analyze MobileNetV2 width_multiplier and train a wm=0.5 model.
This script fulfills the requirement from [CELL 59-60] to train the 
wm=0.5 model instead of loading a pre-trained one.
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
from src.models.mobilenet_v2 import MobileNetV2

def parse_args():
    parser = argparse.ArgumentParser(description="Script 5: Analyze MobileNetV2 width_multiplier and train a specific WM model")
    parser.add_argument('--log_file', type=str, default="run_05_mobilenet_v2_hyperparams.log", help="Log file name")
    parser.add_argument('--model_save_path', type=str, default="models/mobilenet_v2_wm0.5_cifar10.pt", help="Path to save the best model")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and validation")
    parser.add_argument('--width_multiplier', type=float, default=0.5, help="Width multiplier for MobileNetV2 to train")
    return parser.parse_args()

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
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logging(args.log_file)
    logger.info(f"--- Starting Run 05: MobileNetV2 Hyperparameter Analysis (Training WM={args.width_multiplier}) ---")
    logger.info(f"Using arguments: {args}")
    logger.info(f"Using device: {DEVICE}")

    try:
        # 1. Analyze Parameter Counts
        analyze_parameter_counts(logger)

        # 2. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

        # 3. Initialize Model (WM=0.5)
        model = MobileNetV2(n_class=10, width_multiplier=args.width_multiplier).to(DEVICE)
        logger.info(f"Model MobileNetV2 initialized. WM={args.width_multiplier}, "
                    f"Parameters: {sum(p.numel() for p in model.parameters())}")

        # 4. Train Model
        logger.info(f"Training model for {args.epochs} epochs...")
        train_model(
            model, train_loader, val_loader, DEVICE, args.epochs, args.lr, args.model_save_path
        )

        # 5. Load best model and evaluate
        logger.info(f"Loading best model (WM={args.width_multiplier}) for final evaluation...")
        model.load_state_dict(torch.load(args.model_save_path))
        get_accuracy(model, val_loader, DEVICE)

        logger.info(f"--- Run 05: MobileNetV2 Hyperparameter Analysis (WM={args.width_multiplier}) Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 05: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
