"""NVIDIA DAVE-2 Baseline Model Implementation.

Reference:
    Bojarski et al., "End to End Learning for Self-Driving Cars", 2016.
    arXiv:1604.07316
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DAVE2(nn.Module):
    """NVIDIA DAVE-2 end-to-end convolutional neural network for steering angle regression.

    Architecture:
        - 3 Conv layers with 5x5 kernels and 2x2 stride
        - 2 Conv layers with 3x3 kernels and 1x1 stride
        - Flatten (1152 features for 66x200 input)
        - 5 Linear layers: 1152 -> 1164 -> 100 -> 50 -> 10 -> 1
    """

    def __init__(
        self,
        input_channels: int = 3,
        dropout: float = 0.5,
        output_dim: int = 1,
        activation: str = "elu",
    ) -> None:
        """Initialize DAVE2 model.

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
            nn.Linear(1152, 1164),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(1164, 100),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 50),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(50, 10),
            act_layer(),
            nn.Linear(10, output_dim),
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
