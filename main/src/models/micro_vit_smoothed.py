"""MicroViT with Online Temporal Smoothing (Moving Median + EMA).

Acts as the 4th neural network in the NXP Cup suite:
- Backbone: MicroViT (359k parameters)
- Post-processing: Online Moving Median (K=3) + Exponential Moving Average (alpha=0.65)
- Purpose: Attenuates high-frequency steering jitter and enhances resilience against
  sudden contrast drops and barrier cast shadows without latency penalty (<0.05 ms).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from main.src.models.micro_vit import MicroViT


class MedianEMASmoother:
    """Online Moving Median + Exponential Moving Average (EMA) filter.

    Designed for real-time edge vehicle control:
    - Moving Median (window_size=3) effectively filters non-linear transient noise,
      spurious reflections, and single-frame dropouts caused by cast shadows.
    - EMA (alpha=0.65) ensures kinematic smoothness and bounded derivative (d_delta/dt).
    """

    def __init__(
        self,
        window_size: int = 3,
        alpha: float = 0.65,
        max_angle_deg: float = 30.0,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_angle_deg = float(max_angle_deg)
        self.buffer: list[float] = []
        self.ema_val: float | None = None

    def reset(self) -> None:
        """Resets temporal filter state (e.g. at the start of a new video lap or run)."""
        self.buffer.clear()
        self.ema_val = None

    def update(self, raw_angle_deg: float) -> float:
        """Applies online median filtering and exponential smoothing to a new angle."""
        clamped = float(np.clip(raw_angle_deg, -self.max_angle_deg, self.max_angle_deg))
        self.buffer.append(clamped)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # 1. Non-linear median filter on local window
        median_val = float(np.median(self.buffer))

        # 2. Exponential Moving Average
        if self.ema_val is None:
            self.ema_val = median_val
        else:
            self.ema_val = self.alpha * median_val + (1.0 - self.alpha) * self.ema_val

        return float(np.clip(self.ema_val, -self.max_angle_deg, self.max_angle_deg))

    def smooth_sequence(self, angles_deg: list[float] | np.ndarray) -> np.ndarray:
        """Applies causal online smoothing across an entire sequence."""
        self.reset()
        smoothed = np.empty(len(angles_deg), dtype=np.float32)
        for i, a in enumerate(angles_deg):
            smoothed[i] = self.update(float(a))
        return smoothed


class MicroViTSmoothed(nn.Module):
    """MicroViT wrapper encapsulating online Median + EMA temporal smoothing."""

    def __init__(
        self,
        window_size: int = 3,
        alpha: float = 0.65,
        max_angle_deg: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.backbone = MicroViT(**kwargs)
        self.smoother = MedianEMASmoother(
            window_size=window_size,
            alpha=alpha,
            max_angle_deg=max_angle_deg,
        )

    def reset_smoother(self) -> None:
        """Resets online temporal state."""
        self.smoother.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MicroViT with optional eval-time temporal smoothing.

        During training (train mode): returns raw backbone predictions for unbiased gradient flow.
        During inference (eval mode) with batch_size=1: applies causal online smoothing.
        """
        raw_out = self.backbone(x)
        if self.training or x.shape[0] != 1:
            return raw_out

        raw_deg = float(raw_out[0, 0].item()) * self.smoother.max_angle_deg
        smoothed_deg = self.smoother.update(raw_deg)
        smoothed_norm = smoothed_deg / self.smoother.max_angle_deg
        return torch.tensor([[smoothed_norm]], dtype=raw_out.dtype, device=raw_out.device)
