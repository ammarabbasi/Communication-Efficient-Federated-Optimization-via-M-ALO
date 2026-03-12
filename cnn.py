"""
Neural Network Model Definitions
=================================
Lightweight CNNs suitable for MNIST, CIFAR-10, Rice Leaf Disease, and WaRP
in a federated setting where client-side compute is limited.

Models
------
  MNISTNet    : Simple 2-conv CNN for 28×28 grayscale images (10 classes)
  CIFAR10Net  : Slightly deeper CNN for 32×32 RGB images (10 classes)
  GeneralCNN  : Configurable CNN for domain-specific datasets (N classes)
               — used for Rice Leaf Disease (5 classes) and WaRP (28 classes)
               — accepts variable input resolution via adaptive pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MNIST Network
# ---------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """Compact CNN for MNIST (28×28 grayscale, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # → 14×14×32
        x = self.pool(F.relu(self.conv2(x)))   # → 7×7×64
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# CIFAR-10 Network
# ---------------------------------------------------------------------------

class CIFAR10Net(nn.Module):
    """Lightweight CNN for CIFAR-10 (32×32 RGB, 10 classes)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# General-purpose CNN  (Rice Leaf Disease / WaRP)
# ---------------------------------------------------------------------------

class GeneralCNN(nn.Module):
    """
    Flexible CNN for variable-resolution RGB images.

    Uses adaptive average pooling before the FC layers, so it accepts any
    input spatial size — no need to recompute FC dimensions manually.
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # always 4×4 regardless of input size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_model(dataset_name: str) -> nn.Module:
    """
    Return an appropriate model given the dataset name.

    Parameters
    ----------
    dataset_name : one of 'mnist', 'cifar10', 'rice', 'warp'
    """
    name = dataset_name.lower()
    if name == "mnist":
        return MNISTNet(num_classes=10)
    elif name in ("cifar10", "cifar-10"):
        return CIFAR10Net(num_classes=10)
    elif name in ("rice", "rice_leaf", "rice_leaf_disease"):
        return GeneralCNN(num_classes=5, in_channels=3)
    elif name == "warp":
        return GeneralCNN(num_classes=28, in_channels=3)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            "Choose from: mnist, cifar10, rice, warp."
        )
