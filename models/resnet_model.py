from torchvision import models
import torch.nn as nn

def create_model(num_classes):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model