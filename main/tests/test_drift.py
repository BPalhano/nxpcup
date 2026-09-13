"""Tests for Evidently AI Data Drift and Model Resilience pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from main.src.monitoring.drift_report import (
    apply_synthetic_drift,
    extract_image_features,
    generate_drift_report,
)

MAIN_DIR = Path(__file__).resolve().parents[1]


def test_apply_synthetic_drift_properties() -> None:
    """Verifies that synthetic drift reduces brightness and maintains image shape/dtype."""
    dummy_img = np.full((66, 200, 3), 180, dtype=np.uint8)
    drifted = apply_synthetic_drift(dummy_img, seed=42)

    assert drifted.shape == (66, 200, 3)
    assert drifted.dtype == np.uint8
    # Overall brightness must decrease under dimming and cast shadows
    assert np.mean(drifted) < np.mean(dummy_img)

    # Determinism
    drifted2 = apply_synthetic_drift(dummy_img, seed=42)
    np.testing.assert_array_equal(drifted, drifted2)


def test_extract_image_features_schema() -> None:
    """Verifies that extracted CV features have correct keys and valid ranges."""
    dummy_img = np.random.randint(0, 256, (66, 200, 3), dtype=np.uint8)
    feats = extract_image_features(dummy_img)

    expected_keys = [
        "mean_brightness",
        "std_contrast",
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_saturation",
        "laplacian_var",
        "edge_density",
        "dark_pixel_fraction",
    ]

    for k in expected_keys:
        assert k in feats
        assert isinstance(feats[k], float)

    assert 0.0 <= feats["dark_pixel_fraction"] <= 1.0
    assert 0.0 <= feats["edge_density"] <= 1.0


def test_generate_drift_report_smoke(tmp_path: Path) -> None:
    """Executes a fast smoke test of generate_drift_report on real dataset sample."""
    csv_path = MAIN_DIR / "data" / "processed" / "labels_continuous.csv"
    images_dir = MAIN_DIR / "data" / "processed" / "images"

    if not csv_path.exists() or not images_dir.exists():
        return

    html_path, json_path, dist_path, res_path = generate_drift_report(
        csv_path=csv_path,
        images_dir=images_dir,
        output_dir=tmp_path,
        max_samples=6,
        seed=123,
    )

    assert html_path.exists()
    assert html_path.stat().st_size > 10_000

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["samples_evaluated"] == 6
    assert "micro_vit" in summary["model_resilience"]
    assert "micro_vit_smoothed" in summary["model_resilience"]
    assert "dave2" in summary["model_resilience"]
    assert "palhano_cnn" in summary["model_resilience"]

    assert dist_path.exists()
    assert res_path.exists()
