import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model.to(device).eval()