"""
Implements the Knowledge Distillation loss and training loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Custom loss for Knowledge Distillation.
    Combines KL Divergence loss (student vs. teacher) and 
    Cross-Entropy loss (student vs. hard labels).
    
    Args:
        temperature (float): Softening parameter for logits.
        alpha (float): Weight for the distillation (KLDiv) loss.
                       (1 - alpha) will be the weight for the CE loss.
    """
    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.
        """
        # Distillation loss (KLDiv)
        soft_targets = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Scale KLDiv loss by T^2 as recommended in the original paper
        distillation_loss = self.kl_div(soft_targets, soft_teacher) * (self.temperature ** 2)

        # Standard classification loss (Cross-Entropy)
        classification_loss = self.ce_loss(student_logits, labels)

        # Weighted combination
        return (self.alpha * distillation_loss) + ((1 - self.alpha) * classification_loss)


def train_student(teacher, student, train_loader, device, epochs, lr, T, alpha):
    """
    Full training loop for the student model using distillation.
    
    Args:
        teacher (nn.Module): The pre-trained teacher model.
        student (nn.Module): The student model to train.
        train_loader (DataLoader): Training data loader.
        device (torch.device): The device (cpu or cuda).
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        T (float): Temperature for distillation.
        alpha (float): Weight for distillation loss.
    """
    teacher.to(device).eval()  # Teacher is in eval mode
    student.to(device)

    criterion = DistillationLoss(temperature=T, alpha=alpha)
    optimizer = optim.Adam(student.parameters(), lr=lr)

    logger.info(f"Starting Knowledge Distillation for {epochs} epochs...")
    logger.info(f"Teacher: {teacher.__class__.__name__}, Student: {student.__class__.__name__}")
    logger.info(f"Device: {device}, LR: {lr}, Temp: {T}, Alpha: {alpha}")

    start_time = time.time()
    for epoch in range(epochs):
        logger.info(f"--- EPOCH {epoch+1}/{epochs} ---")
        student.train()
        running_loss = 0.0
        processed_data = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for inputs, labels in pbar:
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                # Get teacher logits (no gradient)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                
                # Get student logits (with gradient)
                student_logits = student(inputs)

                # Compute combined loss
                loss = criterion(student_logits, teacher_logits, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                processed_data += inputs.size(0)
                pbar.set_postfix(loss=f"{running_loss/processed_data:.4f}")
            
            except Exception as e:
                logger.error(f"Error during distillation step: {e}")
                continue
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    end_time = time.time()
    logger.info(f"Distillation training complete. Total time: {(end_time - start_time) / 60:.2f} minutes.")
