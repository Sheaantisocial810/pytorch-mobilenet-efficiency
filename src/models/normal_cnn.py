"""
Implementation of a standard CNN with the same layer structure as MobileNetV1
for comparison.
"""
import torch.nn as nn
from .mobilenet_v1 import conv_bn  # Re-use the conv_bn helper


class NormalCNN(nn.Module):
    """
    Standard CNN for comparison.
    
    Args:
        n_class (int): Number of output classes.
    """
    def __init__(self, n_class=10):
        super().__init__()

        self.model = nn.Sequential(
            conv_bn(  3,   32, 2),
            conv_bn( 32,   64, 1),
            conv_bn( 64,  128, 2),
            conv_bn(128,  128, 1),
            conv_bn(128,  256, 2),
            conv_bn(256,  256, 1),
            conv_bn(256,  512, 2),
            conv_bn(512,  512, 1),
            conv_bn(512,  512, 1),
            conv_bn(512,  512, 1),
            conv_bn(512,  512, 1),
            conv_bn(512,  512, 1),
            conv_bn(512, 1024, 2),
            conv_bn(1024, 1024, 1),
            nn.AvgPool2d(7), # Adjusted for 224x224 input
        )
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
