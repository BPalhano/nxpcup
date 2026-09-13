"""Automated Unit Tests for NXP Cup FastAPI Steering Inference Service."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.src.api.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provides a TestClient context with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def synthetic_road_image_bytes() -> bytes:
    """Generates an in-memory JPEG road image (66x200) with center track and guide lines."""
    img = np.ones((66, 200, 3), dtype=np.uint8) * 240  # white canvas
    # Left and right black tape lines
    cv2.line(img, (30, 0), (10, 66), (20, 20, 20), 4)
    cv2.line(img, (170, 0), (190, 66), (20, 20, 20), 4)

    success, buffer = cv2.imencode(".jpg", img)
    assert success, "Failed to encode synthetic JPEG image"
    return buffer.tobytes()


def test_root_endpoint(client: TestClient) -> None:
    """Verifies root endpoint metadata and docs links."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs_url"]
    assert "/steer" in data["steer_url"]


def test_health_endpoint(client: TestClient) -> None:
    """Verifies service health check and telemetry."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["active_model"] == "micro_vit_smoothed"
    assert "micro_vit" in data["available_models"]
    assert "dave2" in data["available_models"]
    assert data["uptime_seconds"] >= 0.0


def test_models_endpoint(client: TestClient) -> None:
    """Verifies model catalog listing all four architectures."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 4
    names = [m["name"] for m in data["models"]]
    assert "micro_vit_smoothed" in names
    assert "micro_vit" in names
    assert "palhano_cnn" in names
    assert "dave2" in names


def test_steer_multipart_upload(client: TestClient, synthetic_road_image_bytes: bytes) -> None:
    """Tests POST /steer with multipart binary image file upload."""
    files = {"file": ("road_frame.jpg", io.BytesIO(synthetic_road_image_bytes), "image/jpeg")}
    resp = client.post("/steer", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["model"] == "micro_vit_smoothed"
    assert -30.0 <= data["steer_deg"] <= 30.0
    assert -1.0 <= data["steer_norm"] <= 1.0
    assert data["inference_time_ms"] > 0.0
    assert "timestamp" in data


def test_steer_base64_payload(client: TestClient, synthetic_road_image_bytes: bytes) -> None:
    """Tests POST /steer/base64 with JSON Base64 image payload."""
    b64_str = base64.b64encode(synthetic_road_image_bytes).decode("utf-8")
    payload = {
        "image_base64": b64_str,
        "model_name": "micro_vit",
        "backend": "onnx",
    }
    resp = client.post("/steer/base64", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["model"] == "micro_vit"
    assert -30.0 <= data["steer_deg"] <= 30.0


def test_steer_model_switching(client: TestClient, synthetic_road_image_bytes: bytes) -> None:
    """Tests switching between different model architectures dynamically."""
    models_to_test = ["palhano_cnn", "dave2", "micro_vit"]
    for m in models_to_test:
        files = {"file": ("test.jpg", io.BytesIO(synthetic_road_image_bytes), "image/jpeg")}
        resp = client.post(f"/steer?model_name={m}", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == m
        assert -30.0 <= data["steer_deg"] <= 30.0


def test_steer_invalid_image_data(client: TestClient) -> None:
    """Tests proper error handling when uploading corrupted image bytes."""
    corrupted_bytes = b"NOT_A_REAL_IMAGE_DATA_CORRUPTED"
    files = {"file": ("corrupt.jpg", io.BytesIO(corrupted_bytes), "image/jpeg")}
    resp = client.post("/steer", files=files)
    assert resp.status_code == 400
    assert "Invalid image content" in resp.json()["detail"]


def test_steer_invalid_model_name(client: TestClient, synthetic_road_image_bytes: bytes) -> None:
    """Tests proper error handling when requesting an unknown model name."""
    files = {"file": ("test.jpg", io.BytesIO(synthetic_road_image_bytes), "image/jpeg")}
    resp = client.post("/steer?model_name=unknown_robot_model", files=files)
    assert resp.status_code == 400
    assert "Invalid model" in resp.json()["detail"]


def test_reset_smoother_endpoint(client: TestClient) -> None:
    """Tests resetting online temporal filter state."""
    resp = client.post("/reset?model_name=micro_vit_smoothed")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
