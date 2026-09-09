"""Unit tests for the 3D Geometric Auto-Labeling and Video Sampling modules."""

from pathlib import Path

import cv2
import numpy as np

from main.data.geometry_estimator import TrackGeometryEstimator
from main.data.temporal_smoother import smooth_lap_angles
from main.data.video_sampler import sample_video_frames

BASE_DIR = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = BASE_DIR / "recording_dataset" / "WIN_20260202_15_54_34_Pro.mp4"


def test_track_geometry_estimator_output_ranges():
    """Verifies that geometry estimator outputs are within physical bounds."""
    estimator = TrackGeometryEstimator(max_steering_deg=30.0)

    # Read real sample frame
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    assert cap.isOpened(), f"Cannot open test video: {SAMPLE_VIDEO}"
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read frame from test video"

    result = estimator.estimate_steering(frame, return_visualization=True)

    assert -30.0 <= result.angle_deg <= 30.0, f"Angle {result.angle_deg} out of bounds"
    assert -1.0 <= result.angle_norm <= 1.0, f"Normalized angle {result.angle_norm} out of bounds"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of bounds"
    assert result.annotated_image is not None
    assert result.bev_image is not None


def test_track_geometry_estimator_blank_frame():
    """Verifies graceful handling of dark/blank/noisy frames."""
    estimator = TrackGeometryEstimator(max_steering_deg=30.0)
    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = estimator.estimate_steering(blank_frame)

    assert result.angle_deg == 0.0
    assert result.confidence <= 0.2
    assert not result.left_line_detected
    assert not result.right_line_detected


def test_track_geometry_estimator_transverse_straight():
    """Verifies that synthetic crossroad/pedestrian patterns force strictly 0.0°."""
    estimator = TrackGeometryEstimator()
    # Create synthetic frame with white road and horizontal black bars (pedestrian crossing)
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    # Add horizontal black bars across road area
    for y in range(400, 600, 30):
        frame[y : y + 10, :] = 20

    result = estimator.estimate_steering(frame)
    assert result.angle_deg == 0.0
    assert result.mode == "CROSSROAD_OR_PEDESTRIAN"



def test_temporal_smoother_outlier_interpolation():
    """Verifies that low-confidence outliers are interpolated and filtered."""
    records = [
        {"angle_deg": 10.0, "confidence": 1.0},
        {"angle_deg": 12.0, "confidence": 1.0},
        {"angle_deg": -50.0, "confidence": 0.1},  # Outlier with low confidence
        {"angle_deg": 14.0, "confidence": 1.0},
        {"angle_deg": 15.0, "confidence": 1.0},
    ]

    smoothed = smooth_lap_angles(records, confidence_threshold=0.35)

    # Outlier at index 2 should be interpolated between 12 and 14 (approx 13)
    outlier_angle = smoothed[2]["angle_deg"]
    assert 10.0 <= outlier_angle <= 16.0, f"Interpolated angle {outlier_angle} was not smooth"
    for r in smoothed:
        assert -30.0 <= r["angle_deg"] <= 30.0


def test_video_sampler_step_and_fps(tmp_path):
    """Verifies that 10 FPS sampling extracts approximately 1/3 of frames from 30 FPS video."""
    raw_out = tmp_path / "raw"
    resized_out = tmp_path / "resized"

    records = sample_video_frames(
        video_path=SAMPLE_VIDEO,
        raw_output_dir=raw_out,
        resized_output_dir=resized_out,
        target_fps=10.0,
        target_size=(200, 66),
    )

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Step is 3, so expected is approx total_native / 3
    expected_count = int(np.ceil(total_native / 3.0))
    assert abs(len(records) - expected_count) <= 2
    assert (resized_out / records[0]["resized_filename"]).exists()

    # Check shape of resized image
    first_img = cv2.imread(records[0]["resized_path"])
    assert first_img.shape == (66, 200, 3)
