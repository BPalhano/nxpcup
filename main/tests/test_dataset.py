"""Unit tests for NXPCupDataset and data pipeline."""

from pathlib import Path

import pytest
import torch

from main.data.dataset import (
    NXPCupDataset,
    get_dataloaders,
    get_val_transforms,
    load_dataset_records,
    split_dataset_records,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = BASE_DIR / "main" / "data" / "processed" / "labels_continuous.csv"
IMAGES_DIR = BASE_DIR / "main" / "data" / "processed" / "images"


@pytest.fixture
def sample_records():
    assert CSV_PATH.exists(), f"Labels file not found at {CSV_PATH}"
    records = load_dataset_records(CSV_PATH)
    assert len(records) > 0, "Labels file is empty"
    return records


def test_load_dataset_records(sample_records):
    """Verifies records schema and expected fields."""
    first = sample_records[0]
    assert "filename" in first
    assert "angle_deg" in first
    assert "angle_norm" in first
    assert "class_name" in first
    assert -1.0 <= first["angle_norm"] <= 1.0


def test_split_dataset_records(sample_records):
    """Verifies that split partitions are disjoint and sum to total."""
    train_recs, val_recs, test_recs = split_dataset_records(
        sample_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    total_len = len(train_recs) + len(val_recs) + len(test_recs)
    assert total_len == len(sample_records)

    # Check disjointness
    train_files = {r["filename"] for r in train_recs}
    val_files = {r["filename"] for r in val_recs}
    test_files = {r["filename"] for r in test_recs}

    assert len(train_files.intersection(val_files)) == 0
    assert len(train_files.intersection(test_files)) == 0
    assert len(val_files.intersection(test_files)) == 0


def test_dataset_item_shape_and_dtype(sample_records):
    """Verifies that items returned by NXPCupDataset have correct tensor shapes and types."""
    dataset = NXPCupDataset(
        records=sample_records[:20],
        images_dir=IMAGES_DIR,
        transforms=get_val_transforms(),
        is_train=False,
    )

    assert len(dataset) == 20
    item = dataset[0]

    image = item["image"]
    angle = item["angle"]

    # Input shape must be (C, H, W) = (3, 66, 200)
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 66, 200)
    assert image.dtype == torch.float32

    # Angle must be scalar float32 in [-1.0, 1.0]
    assert isinstance(angle, torch.Tensor)
    assert angle.dtype == torch.float32
    assert -1.0 <= angle.item() <= 1.0


def test_horizontal_flip_angle_inversion():
    """Verifies that horizontal flip strictly inverts the sign of steering angles."""
    records = [
        {
            "filename": "frame_0001_corner_200x66.jpg",
            "angle_deg": 30.0,
            "angle_norm": 1.0,
            "class_name": "right_curve",
        }
    ]

    # With flip_p = 1.0, flip is guaranteed
    dataset_flipped = NXPCupDataset(
        records=records,
        images_dir=IMAGES_DIR,
        transforms=get_val_transforms(),
        is_train=True,
        horizontal_flip_p=1.0,
    )
    item_flipped = dataset_flipped[0]
    assert item_flipped["angle"].item() == pytest.approx(-1.0)
    assert item_flipped["angle_deg"].item() == pytest.approx(-30.0)

    # With flip_p = 0.0, no flip occurs
    dataset_normal = NXPCupDataset(
        records=records,
        images_dir=IMAGES_DIR,
        transforms=get_val_transforms(),
        is_train=True,
        horizontal_flip_p=0.0,
    )
    item_normal = dataset_normal[0]
    assert item_normal["angle"].item() == pytest.approx(1.0)
    assert item_normal["angle_deg"].item() == pytest.approx(30.0)


def test_dataloaders_batch_generation():
    """Verifies that get_dataloaders produces valid batches."""
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        batch_size=8,
        num_workers=0,  # 0 workers for fast synchronous test
    )

    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]

    assert images.shape == (8, 3, 66, 200)
    assert angles.shape == (8,)
    assert images.dtype == torch.float32
    assert angles.dtype == torch.float32
