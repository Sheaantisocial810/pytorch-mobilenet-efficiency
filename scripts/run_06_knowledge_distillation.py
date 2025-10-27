"""
Script 6: Run Knowledge Distillation from ResNet-18 to MobileNetV2.
"""
import torch
import timm
import logging
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import get_accuracy
from src.models.mobilenet_v2 import MobileNetV2
from src.distillation import train_student

def parse_args():
    parser = argparse.ArgumentParser(description="Script 6: Knowledge Distillation (ResNet-18 to MobileNetV2)")
    parser.add_argument('--log_file', type=str, default="run_06_knowledge_distillation.log", help="Log file name")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training and validation")
    parser.add_argument('--temperature', type=float, default=3.0, help="Temperature for distillation")
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for distillation loss (0.0 to 1.0)")
    parser.add_argument('--student_wm', type=float, default=1.0, help="Width multiplier for the student MobileNetV2")
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logging(args.log_file)
    logger.info("--- Starting Run 06: Knowledge Distillation ---")
    logger.info(f"Using arguments: {args}")
    logger.info(f"Using device: {DEVICE}")

    try:
        # 1. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

        # 2. Initialize Teacher Model
        logger.info("Loading pre-trained ResNet-18 (teacher) model...")
        teacher_model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        teacher_model = teacher_model.to(DEVICE)
        
        logger.info("Evaluating teacher model on CIFAR-10 (expect low accuracy as it's ImageNet pre-trained)...")
        get_accuracy(teacher_model, val_loader, DEVICE)

        # 3. Initialize Student Model
        student_model = MobileNetV2(n_class=10, width_multiplier=args.student_wm).to(DEVICE)
        logger.info(f"Student model MobileNetV2 (WM={args.student_wm}) initialized. "
                    f"Parameters: {sum(p.numel() for p in student_model.parameters())}")

        # 4. Train Student with Distillation
        train_student(
            teacher=teacher_model,
            student=student_model,
            train_loader=train_loader,
            device=DEVICE,
            epochs=args.epochs,
            lr=args.lr,
            T=args.temperature,
            alpha=args.alpha
        )

        # 5. Evaluate Student
        logger.info("Evaluating student model after distillation...")
        get_accuracy(student_model, val_loader, DEVICE)

        logger.info("--- Run 06: Knowledge Distillation Finished ---")

    except Exception as e:
        logger.error(f"An error occurred in Run 06: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
