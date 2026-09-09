"""PyTorch Dataset implementation for NXP Cup Autonomous Driving.

Features:
- Reads cleaned labels and 200x66 RGB road images.
- Implements Albumentations augmentation pipeline (RandomShadow, RandomBrightnessContrast).
- Handles steering angle geometric compensation on HorizontalFlip (reversing angle sign).
- Provides stratified train/val/test splitting and DataLoader factories.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Standard ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    brightness_contrast_p: float = 0.5,
    shadow_p: float = 0.4,
    shift_scale_rotate_p: float = 0.3,
) -> A.Compose:
    """Builds the Albumentations augmentation pipeline for training."""
    transforms_list = [
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=brightness_contrast_p,
        ),
        A.RandomShadow(
            p=shadow_p,
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
        ),
        A.Affine(
            scale=(0.96, 1.04),
            translate_percent=(-0.04, 0.04),
            rotate=(-4, 4),
            border_mode=cv2.BORDER_CONSTANT,
            p=shift_scale_rotate_p,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)


def get_val_transforms() -> A.Compose:
    """Builds the deterministic transformation pipeline for validation and testing."""
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class NXPCupDataset(Dataset):
    """PyTorch Dataset for NXP Cup camera frames and steering angles."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        images_dir: str | Path,
        transforms: A.Compose | None = None,
        is_train: bool = False,
        horizontal_flip_p: float = 0.5,
    ) -> None:
        """Initialize NXPCupDataset.

        Args:
            records: List of record dictionaries containing 'filename', 'angle_norm', 'angle_deg'.
            images_dir: Path to directory containing image files.
            transforms: Albumentations Compose pipeline.
            is_train: Whether this dataset is used for training (enables horizontal flip).
            horizontal_flip_p: Probability of applying horizontal flip with angle inversion.
        """
        self.records = records
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.is_train = is_train
        self.horizontal_flip_p = horizontal_flip_p

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        img_path = self.images_dir / rec["filename"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        angle_norm = float(rec["angle_norm"])
        angle_deg = float(rec["angle_deg"])

        # Manual Horizontal Flip with exact steering angle inversion
        # If the car flips horizontally, a left turn (-30°) becomes a right turn (+30°)
        if self.is_train and random.random() < self.horizontal_flip_p:
            image = np.ascontiguousarray(np.fliplr(image))
            angle_norm = -angle_norm
            angle_deg = -angle_deg

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image_tensor,
            "angle": torch.tensor(angle_norm, dtype=torch.float32),
            "angle_deg": torch.tensor(angle_deg, dtype=torch.float32),
            "next_step_steer_deg": torch.tensor(angle_deg, dtype=torch.float32),
            "next_step_steer_norm": torch.tensor(angle_norm, dtype=torch.float32),
            "lateral_offset": torch.tensor(
                float(rec.get("lateral_offset_px", 0.0)), dtype=torch.float32
            ),
            "curvature": torch.tensor(float(rec.get("curvature", 0.0)), dtype=torch.float32),
            "filename": rec["filename"],
            "class_name": rec.get("class_name", ""),
            "section_id": rec.get("section_id", ""),
        }


def load_dataset_records(csv_path: str | Path) -> list[dict[str, Any]]:
    """Loads records from labels CSV (supports both discrete and continuous formats)."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Prioritize next_step_steer_deg for centering control
            if "next_step_steer_deg" in row and row["next_step_steer_deg"] != "":
                angle_deg = float(row["next_step_steer_deg"])
                angle_norm = float(row.get("next_step_steer_norm", angle_deg / 30.0))
            else:
                angle_deg = float(row["angle_deg"])
                angle_norm = float(row.get("angle_norm", angle_deg / 30.0))

            # Derive class_name for stratified sampling
            class_name = row.get("class_name")
            if not class_name:
                mode = row.get("mode")
                if mode:
                    class_name = mode.lower()
                elif angle_deg < -5.0:
                    class_name = "left_curve"
                elif angle_deg > 5.0:
                    class_name = "right_curve"
                else:
                    class_name = "straight"

            records.append(
                {
                    "filename": row["filename"],
                    "angle_deg": angle_deg,
                    "angle_norm": angle_norm,
                    "next_step_steer_deg": angle_deg,
                    "next_step_steer_norm": angle_norm,
                    "class_name": class_name,
                    "mode": row.get("mode", class_name.upper()),
                    "section_id": row.get("section_id", ""),
                    "lateral_offset_px": float(row.get("lateral_offset_px", 0.0)),
                    "curvature": float(row.get("curvature", 0.0)),
                    "confidence": float(row.get("confidence", 1.0)),
                }
            )
    return records


def split_dataset_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits records into stratified train, val, and test subsets based on class_name."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group by class to ensure proportional distribution of turns
    class_groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        c = r["class_name"]
        class_groups.setdefault(c, []).append(r)

    train_recs: list[dict[str, Any]] = []
    val_recs: list[dict[str, Any]] = []
    test_recs: list[dict[str, Any]] = []

    for _, group in class_groups.items():
        rng.shuffle(group)
        n_total = len(group)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_recs.extend(group[:n_train])
        val_recs.extend(group[n_train : n_train + n_val])
        test_recs.extend(group[n_train + n_val :])

    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)

    logger.info(
        "Dataset Split -> Train: %d, Val: %d, Test: %d (Total: %d)",
        len(train_recs),
        len(val_recs),
        len(test_recs),
        len(records),
    )
    return train_recs, val_recs, test_recs


def get_dataloaders(
    csv_path: str | Path,
    images_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Factory function creating train, val, and test PyTorch DataLoaders."""
    records = load_dataset_records(csv_path)
    train_recs, val_recs, test_recs = split_dataset_records(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = NXPCupDataset(
        records=train_recs,
        images_dir=images_dir,
        transforms=get_train_transforms(),
        is_train=True,
    )
    val_dataset = NXPCupDataset(
        records=val_recs,
        images_dir=images_dir,
        transforms=get_val_transforms(),
        is_train=False,
    )
    test_dataset = NXPCupDataset(
        records=test_recs,
        images_dir=images_dir,
        transforms=get_val_transforms(),
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
