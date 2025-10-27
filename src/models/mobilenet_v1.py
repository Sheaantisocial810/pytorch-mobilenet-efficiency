"""
Implementation of the MobileNetV1 architecture.
"""
import torch.nn as nn


def conv_bn(inp, oup, stride):
    """
    Standard 3x3 convolution with Batch Norm and ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # 3x3 conv
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    """
    Depthwise Separable Convolution block.
    """
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        # Pointwise
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    """
    MobileNetV1 model class.
    
    Args:
        n_class (int): Number of output classes.
    """
    def __init__(self, n_class=10):
        super().__init__()

        self.model = nn.Sequential(
            conv_bn(  3,   32, 2),
            conv_dw( 32,   64, 1),
            conv_dw( 64,  128, 2),
            conv_dw(128,  128, 1),
            conv_dw(128,  256, 2),
            conv_dw(256,  256, 1),
            conv_dw(256,  512, 2),
            conv_dw(512,  512, 1),
            conv_dw(512,  512, 1),
            conv_dw(512,  512, 1),
            conv_dw(512,  512, 1),
            conv_dw(512,  512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7), # Adjusted for 224x224 input
        )
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
