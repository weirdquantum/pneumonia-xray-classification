"""Final baseline CNN trained from scratch. Accepts (B, 3, 100, 100) and returns logits."""
import torch
from torch import nn


class BaselineCNN(nn.Module):
    """PyTorch reconstruction of the original Keras notebook architecture."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))



def build_model(name: str = 'baseline') -> nn.Module:
    if name != 'baseline':
        raise ValueError(f'Only the final baseline model is supported: {name}')
    return BaselineCNN()
