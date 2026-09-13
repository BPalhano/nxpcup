"""Model Factory for NXP Cup Autonomous Driving Architectures.

Provides a unified interface for instantiating DAVE-2, PalhanoCNN, and MicroViT models.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from main.src.models.dave2 import DAVE2
from main.src.models.micro_vit import MicroViT
from main.src.models.micro_vit_smoothed import MicroViTSmoothed
from main.src.models.palhano_cnn import PalhanoCNN


class ModelFactory:
    """Factory Pattern class to dynamically create neural models."""

    _REGISTRY: dict[str, type[nn.Module]] = {
        "dave2": DAVE2,
        "nvidia_dave2": DAVE2,
        "palhano_cnn": PalhanoCNN,
        "palhano": PalhanoCNN,
        "micro_vit": MicroViT,
        "microvit": MicroViT,
        "micro_vit_smoothed": MicroViTSmoothed,
        "microvit_smoothed": MicroViTSmoothed,
    }

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Returns sorted list of registered model names."""
        return sorted(list(cls._REGISTRY.keys()))

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> nn.Module:
        """Instantiate a neural model by name.

        Args:
            name: Case-insensitive name of the architecture.
            **kwargs: Hyperparameters forwarded to the model constructor.

        Returns:
            nn.Module instance.

        Raises:
            ValueError: If name is not registered.
        """
        key = name.strip().lower()
        if key not in cls._REGISTRY:
            available = ", ".join(cls.list_available_models())
            raise ValueError(f"Unknown model name '{name}'. Available models: [{available}]")

        model_cls = cls._REGISTRY[key]
        return model_cls(**kwargs)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Counts total and trainable parameters in a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Dictionary with 'total' and 'trainable' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_model_summary_str(model: nn.Module, model_name: str) -> str:
    """Generates a formatted summary string for logging."""
    counts = count_parameters(model)
    return (
        f"Model: {model_name} | "
        f"Total Params: {counts['total']:,} | "
        f"Trainable Params: {counts['trainable']:,}"
    )
