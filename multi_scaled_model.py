import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class BasicConvBlock(nn.Module):
    """
    A small block with Conv -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiScaleEmotionCNN(nn.Module):
    """
    Multi-scale CNN that processes an image at 48x48 and 24x24, then fuses the features.
    """
    def __init__(self, num_classes=6):
        super(MultiScaleEmotionCNN, self).__init__()

        # Branch 1: processes 48x48
        self.branch1_conv1 = BasicConvBlock(in_channels=3, out_channels=32)
        self.branch1_conv2 = BasicConvBlock(in_channels=32, out_channels=64)
        self.branch1_pool = nn.AdaptiveAvgPool2d((6, 6))  # reduce spatial dimension

        # Branch 2: processes 24x24
        # We'll do the resizing inside forward() or in a transform.
        # For now, we assume we do it in forward for clarity.
        self.branch2_conv1 = BasicConvBlock(in_channels=3 , out_channels=32)
        self.branch2_conv2 = BasicConvBlock(in_channels=32, out_channels=64)
        self.branch2_pool = nn.AdaptiveAvgPool2d((3, 3))

        # Fully-connected layers after concatenation
        # The first branch outputs 64 * 6 * 6 = 2304, the second branch 64 * 3 * 3 = 576
        # Combined = 2880
        self.fc1 = nn.Linear(2880, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x is assumed to be (N, 1, 48, 48) for grayscale
        # If you have 3-channel images, change in_channels=1 -> in_channels=3 above.

        # Branch 1: 48x48
        b1 = self.branch1_conv1(x)      # (N, 32, 48, 48)
        b1 = self.branch1_conv2(b1)     # (N, 64, 48, 48)
        b1 = self.branch1_pool(b1)      # (N, 64, 6, 6)
        b1 = b1.view(b1.size(0), -1)    # (N, 64*6*6) = (N, 2304)

        # Branch 2: 24x24
        # Downsample the input manually or do it in a transform
        x_down = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
        b2 = self.branch2_conv1(x_down) # (N, 32, 24, 24)
        b2 = self.branch2_conv2(b2)     # (N, 64, 24, 24)
        b2 = self.branch2_pool(b2)      # (N, 64, 3, 3)
        b2 = b2.view(b2.size(0), -1)    # (N, 64*3*3) = (N, 576)

        # Concatenate
        fused = torch.cat([b1, b2], dim=1)  # (N, 2880)

        # Fully connected layers
        fused = self.fc1(fused)
        fused = self.bn1(fused)
        fused = self.relu1(fused)
        fused = self.drop1(fused)

        fused = self.fc2(fused)
        fused = self.bn2(fused)
        fused = self.relu2(fused)
        fused = self.drop2(fused)

        out = self.fc3(fused)
        return out
