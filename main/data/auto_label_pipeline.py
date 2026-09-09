"""End-to-End Pipeline for 10 FPS Video Sampling and 3D Geometric Auto-Labeling.

Processes all videos in `recording_dataset/`, estimates continuous steering angles
using Inverse Perspective Mapping (IPM) and Pure Pursuit, applies Savitzky-Golay
temporal filtering, and outputs `main/data/processed/labels_continuous.csv`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from main.data.geometry_estimator import TrackGeometryEstimator
from main.data.temporal_smoother import smooth_lap_angles
from main.data.video_sampler import sample_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_auto_label_pipeline(
    video_dir: str | Path,
    raw_frames_dir: str | Path,
    processed_images_dir: str | Path,
    output_csv_path: str | Path,
    qc_output_dir: str | Path | None = None,
    target_fps: float = 10.0,
    max_qc_per_video: int = 3,
) -> Path:
    """Executes the complete video sampling and geometric auto-labeling pipeline.

    Args:
        video_dir: Directory containing input .mp4 video files.
        raw_frames_dir: Directory to save extracted HD frames.
        processed_images_dir: Directory to save resized 200x66 model input images.
        output_csv_path: Path for output labels_continuous.csv.
        qc_output_dir: Directory to save diagnostic QC visual overlays.
        target_fps: Frame rate for extraction (default 10.0 FPS).
        max_qc_per_video: Number of sample annotated images to save per video for QC.

    Returns:
        Path: Path to generated CSV file.
    """
    video_dir = Path(video_dir)
    raw_frames_dir = Path(raw_frames_dir)
    processed_images_dir = Path(processed_images_dir)
    output_csv_path = Path(output_csv_path)

    raw_frames_dir.mkdir(parents=True, exist_ok=True)
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if qc_output_dir is not None:
        qc_output_dir = Path(qc_output_dir)
        qc_output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm")))
    logger.info("Starting Auto-Label pipeline across %d video files...", len(video_files))

    estimator = TrackGeometryEstimator()
    all_dataset_records: list[dict[str, Any]] = []

    for vid_idx, vid_file in enumerate(video_files):
        logger.info("[%d/%d] Processing video: %s", vid_idx + 1, len(video_files), vid_file.name)

        # 1. Sample video frames at 10 FPS
        sampled_frames = sample_video_frames(
            video_path=vid_file,
            raw_output_dir=raw_frames_dir,
            resized_output_dir=processed_images_dir,
            target_fps=target_fps,
            target_size=(200, 66),
        )

        lap_records: list[dict[str, Any]] = []

        # 2. Geometric 3D angle estimation
        qc_saved = 0
        for rec in sampled_frames:
            frame_raw = cv2.imread(rec["raw_path"])
            if frame_raw is None:
                continue

            want_qc = qc_output_dir is not None and qc_saved < max_qc_per_video
            res = estimator.estimate_steering(frame_raw, return_visualization=want_qc)

            rec["angle_deg"] = res.angle_deg
            rec["angle_norm"] = res.angle_norm
            rec["confidence"] = res.confidence
            rec["curvature"] = res.curvature
            rec["lateral_offset_px"] = res.lateral_offset_px
            rec["left_detected"] = res.left_line_detected
            rec["right_detected"] = res.right_line_detected
            rec["mode"] = res.mode

            # Save visual QC image if requested
            if want_qc and res.annotated_image is not None and qc_output_dir is not None:
                qc_path = qc_output_dir / f"qc_{rec['video_id']}_f{rec['frame_idx']:04d}.jpg"
                cv2.imwrite(str(qc_path), res.annotated_image)
                qc_saved += 1

            lap_records.append(rec)

        # 3. Temporal smoothing across the lap sequence
        smoothed_lap = smooth_lap_angles(lap_records)
        all_dataset_records.extend(smoothed_lap)

    # 4. Write master continuous labels CSV
    logger.info(
        "Writing continuous labels to %s (%d records)...", output_csv_path, len(all_dataset_records)
    )
    fieldnames = [
        "filename",
        "video_id",
        "frame_idx",
        "timestamp_s",
        "angle_deg",
        "angle_norm",
        "confidence",
        "curvature",
        "mode",
    ]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_dataset_records:
            writer.writerow(
                {
                    "filename": r["resized_filename"],
                    "video_id": r["video_id"],
                    "frame_idx": r["frame_idx"],
                    "timestamp_s": r["timestamp_s"],
                    "angle_deg": r["angle_deg"],
                    "angle_norm": r["angle_norm"],
                    "confidence": r["confidence"],
                    "curvature": r["curvature"],
                    "mode": r.get("mode", "STRAIGHT"),
                }
            )

    # Summary Statistics
    angles = np.array([r["angle_deg"] for r in all_dataset_records])
    left_turns = np.count_nonzero(angles < -5.0)
    straight = np.count_nonzero((angles >= -5.0) & (angles <= 5.0))
    right_turns = np.count_nonzero(angles > 5.0)

    logger.info("================ AUTO-LABELING SUMMARY ================")
    logger.info("Total Frames Generated: %d", len(all_dataset_records))
    logger.info(
        "Steering Angle Min: %.2f deg | Max: %.2f deg", float(np.min(angles)), float(np.max(angles))
    )
    logger.info(
        "Steering Angle Mean: %.2f deg | Std: %.2f deg",
        float(np.mean(angles)),
        float(np.std(angles)),
    )
    logger.info(
        "Class Balance -> Left (< -5°): %d (%.1f%%)", left_turns, 100 * left_turns / len(angles)
    )
    logger.info(
        "Class Balance -> Straight ([-5°, +5°]): %d (%.1f%%)",
        straight,
        100 * straight / len(angles),
    )
    logger.info(
        "Class Balance -> Right (> +5°): %d (%.1f%%)", right_turns, 100 * right_turns / len(angles)
    )
    logger.info("=======================================================")

    return output_csv_path


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    v_dir = base / "recording_dataset"
    raw_out = base / "main" / "data" / "extracted_frames"
    proc_out = base / "main" / "data" / "processed" / "images"
    out_csv = base / "main" / "data" / "processed" / "labels_continuous.csv"
    qc_out = base / "main" / "data" / "qc_samples"

    run_auto_label_pipeline(
        video_dir=v_dir,
        raw_frames_dir=raw_out,
        processed_images_dir=proc_out,
        output_csv_path=out_csv,
        qc_output_dir=qc_out,
        target_fps=10.0,
    )
