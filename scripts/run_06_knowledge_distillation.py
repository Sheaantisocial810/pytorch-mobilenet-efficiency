"""
Script 6: Run Knowledge Distillation from ResNet-18 to MobileNetV2.
"""
import torch
import timm
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging
from src.utils.data_loader import get_cifar10_loaders
from src.utils.training import get_accuracy
from src.models.mobilenet_v2 import MobileNetV2
from src.distillation import train_student

# --- Configuration ---
LOG_FILE = "run_06_knowledge_distillation.log"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10  # Increased from 1 for a more meaningful run
LR = 1e-3
BATCH_SIZE = 128
TEMPERATURE = 3.0
ALPHA = 0.5  # Balance between KD loss and CE loss
# --- End Configuration ---

def main():
    logger = setup_logging(LOG_FILE)
    logger.info("--- Starting Run 06: Knowledge Distillation ---")

    try:
        # 1. Load Data
        train_loader, val_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

        # 2. Initialize Teacher Model
        logger.info("Loading pre-trained ResNet-18 (teacher) model...")
        teacher_model = timm.create_model('resnet18', pretrained=True, num_classes=10)
        teacher_model = teacher_model.to(DEVICE)
        
        # Evaluate teacher (as in notebook)
        logger.info("Evaluating teacher model on CIFAR-10 (expect low accuracy as it's ImageNet pre-trained)...")
        get_accuracy(teacher_model, val_loader, DEVICE)

        # 3. Initialize Student Model
        student_model = MobileNetV2(n_class=10).to(DEVICE)
        logger.info(f"Student model MobileNetV2 initialized. Parameters: {sum(p.numel() for p in student_model.parameters())}")

        # 4. Train Student with Distillation
        train_student(
            teacher=teacher_model,
            student=student_model,
            train_loader=train_loader,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LR,
            T=TEMPERATURE,
            alpha=ALPHA
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
