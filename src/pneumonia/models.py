"""Model registry. Every model outputs a single logit (pneumonia vs. normal)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn
from torchvision import models as tvm


class SimpleCNN(nn.Module):
    """From-scratch baseline: 4 double-conv blocks with BatchNorm and global average pooling."""

    def __init__(self, dropout: float = 0.3, batchnorm: bool = True):
        super().__init__()
        blocks = []
        channels = [3, 32, 64, 128, 256]
        norm = nn.BatchNorm2d if batchnorm else lambda c: nn.Identity()
        for c_in, c_out in pairwise(channels):
            blocks += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=not batchnorm),
                norm(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=not batchnorm),
                norm(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SimpleCNNv1(nn.Module):
    """PyTorch replica of the v1 Keras CNN (cnn.ipynb at git tag v1): 4 x (conv-pool-dropout), flatten, Dense(128)."""

    def __init__(self, image_size: int = 100):
        super().__init__()
        blocks, c_in = [], 3
        for c_out in (32, 32, 64, 64):
            blocks += [nn.Conv2d(c_in, c_out, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 3, image_size, image_size)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

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


def build_model(name: str, pretrained: bool = True, image_size: int = 224) -> ModelSpec:
    if name in ("simple_cnn", "simple_cnn_nobn"):
        m = SimpleCNN(batchnorm=name == "simple_cnn")
        return ModelSpec(m, m.head, m.features[-2], pretrained=False)  # last ReLU before the final pool

    if name == "simple_cnn_v1":
        m = SimpleCNNv1(image_size)
        return ModelSpec(m, m.head, m.features[-3], pretrained=False)  # last ReLU

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
ABLATION_MODEL_NAMES = ("simple_cnn_nobn", "simple_cnn_v1")
