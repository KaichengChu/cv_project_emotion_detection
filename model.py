import torch
import torch.nn as nn
import torchvision.models as models

# Use pretrained model, always fine-tune the model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()

        densenet = models.densenet169(pretrained=True)

        self.feature_extractor = densenet.features

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(1664, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        # Feature extraction
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        # Classification
        x = self.classifier(x)
        return x
