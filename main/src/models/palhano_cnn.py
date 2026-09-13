"""Pruned Convolutional Baseline Model for NXP Cup Autonomous Driving.

Architecture:
    Compact 5-layer convolutional baseline with dense regression head.
    Exactly 886,037 parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PalhanoCNN(nn.Module):
    """Pruned convolutional network architecture for steering regression.

    Architecture (exactly 886,037 parameters for 66x200 input):
        - Conv1: 24 filters 5x5, stride 2 (output: 24x31x98)
        - Conv2: 36 filters 5x5, stride 2 (output: 36x14x47)
        - Conv3: 48 filters 5x5, stride 2 (output: 48x5x22)
        - Conv4: 64 filters 3x3, stride 1 (output: 64x3x20)
        - Conv5: 64 filters 3x3, stride 1 (output: 64x1x18)
        - Flatten: 1152 features
        - Linear 1: 1152 -> 512
        - Linear 2: 512 -> 256
        - Linear 3: 256 -> 128
        - Linear 4 (Output): 128 -> 1
    """

    def __init__(
        self,
        input_channels: int = 3,
        dropout: float = 0.5,
        output_dim: int = 1,
        activation: str = "elu",
    ) -> None:
        """Initialize convolutional regression model.

        Args:
            input_channels: Number of input image channels (default: 3 for RGB).
            dropout: Dropout probability in dense layers.
            output_dim: Number of output regression predictions (default: 1 for normalized angle).
            activation: Activation function name ('elu' or 'relu').
        """
        super().__init__()
        act_layer = nn.ELU if activation.lower() == "elu" else nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
            act_layer(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            act_layer(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            act_layer(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            act_layer(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            act_layer(),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(1152, 512),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 66, 200).

        Returns:
            Steering prediction tensor of shape (B, 1).
        """
        feat = self.features(x)
        flat = self.flatten(feat)
        out = self.classifier(flat)
        return out
