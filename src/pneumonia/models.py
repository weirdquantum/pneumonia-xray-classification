"""Model registry. Every model outputs a single logit (pneumonia vs. normal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision import models as tvm


class SimpleCNN(nn.Module):
    """From-scratch baseline: the v1 4-block CNN with BatchNorm and global average pooling."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def vit_tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """(B, 1 + H*W, C) ViT tokens -> (B, C, H, W) feature map, dropping the [CLS] token."""
    patches = tokens[:, 1:, :]
    side = int(patches.shape[1] ** 0.5)
    return patches.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2)


@dataclass
class ModelSpec:
    model: nn.Module
    head: nn.Module  # parameters trained in the frozen-backbone warm-up stage
    cam_layer: nn.Module  # Grad-CAM target layer
    cam_reshape: Callable[[torch.Tensor], torch.Tensor] | None = None
    pretrained: bool = True


def build_model(name: str, pretrained: bool = True) -> ModelSpec:
    if name == "simple_cnn":
        m = SimpleCNN()
        return ModelSpec(m, m.head, m.features[-2], pretrained=False)  # last ReLU before the final pool

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return ModelSpec(m, m.fc, m.layer4, pretrained=pretrained)

    if name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        return ModelSpec(m, m.classifier, m.features.denseblock4, pretrained=pretrained)  # norm5 feeds an in-place ReLU

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        return ModelSpec(m, m.classifier, m.features[-1], pretrained=pretrained)

    if name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
        return ModelSpec(m, m.heads, m.encoder.layers[-1].ln_1, vit_tokens_to_map, pretrained=pretrained)

    raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")


MODEL_NAMES = ("simple_cnn", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16")
