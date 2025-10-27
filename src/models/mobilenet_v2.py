"""
Implementation of the MobileNetV2 architecture.
"""
import torch.nn as nn


def conv_bn_v2(inp, oup, stride):
    """
    Standard 3x3 convolution with Batch Norm and ReLU6 (for V2).
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    """
    1x1 pointwise convolution with Batch Norm and ReLU6.
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    """
    Inverted Residual block for MobileNetV2.
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (stride == 1) and (inp == oup)
        hidden_dim = int(inp * expand_ratio)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model class.
    
    Args:
        n_class (int): Number of output classes.
        width_multiplier (float): Controls the width of the network.
    """
    def __init__(self, n_class=10, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()

        input_channel = int(32 * width_multiplier)
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        last_channel = int(1280 * width_multiplier) if width_multiplier > 1.0 else 1280

        # First layer
        self.features = [conv_bn_v2(3, input_channel, 2)]

        # Inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Last layers
        self.features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*self.features)

        # Classifier
        self.classifier = nn.Linear(last_channel, n_class)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
