"""Video sampling module for NXP Cup Autonomous Driving.

Extracts frames from raw 30 FPS videos at a fixed sampling rate (10 FPS),
saving high-resolution frames for 3D geometric scene analysis and
resized 200x66 frames for neural network training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)


def sample_video_frames(
    video_path: str | Path,
    raw_output_dir: str | Path,
    resized_output_dir: str | Path,
    target_fps: float = 10.0,
    target_size: tuple[int, int] = (200, 66),  # (width, height)
) -> list[dict[str, Any]]:
    """Samples frames from a single video at target_fps.

    Args:
        video_path: Path to video file.
        raw_output_dir: Directory to save high-resolution frames.
        resized_output_dir: Directory to save resized 200x66 frames.
        target_fps: Target frame rate (default 10.0 FPS).
        target_size: (width, height) for network input.

    Returns:
        list of metadata dictionaries for each extracted frame.
    """
    video_path = Path(video_path)
    raw_output_dir = Path(raw_output_dir)
    resized_output_dir = Path(resized_output_dir)

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    resized_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps != native_fps:  # Check for NaN or non-positive
        native_fps = 30.0

    step = max(1, int(round(native_fps / target_fps)))
    total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_id = video_path.stem
    frame_records: list[dict[str, Any]] = []

    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            timestamp_s = frame_idx / native_fps
            frame_name_raw = f"{video_id}_frame_{extracted_count:05d}.jpg"
            frame_name_resized = f"{video_id}_frame_{extracted_count:05d}_200x66.jpg"

            raw_path = raw_output_dir / frame_name_raw
            resized_path = resized_output_dir / frame_name_resized

            # Save raw HD frame
            cv2.imwrite(str(raw_path), frame)

            # Resize and save for model training
            resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(resized_path), resized_frame)

            frame_records.append(
                {
                    "video_id": video_id,
                    "frame_idx": extracted_count,
                    "native_frame_idx": frame_idx,
                    "timestamp_s": round(timestamp_s, 3),
                    "raw_filename": frame_name_raw,
                    "resized_filename": frame_name_resized,
                    "raw_path": str(raw_path),
                    "resized_path": str(resized_path),
                }
            )
            extracted_count += 1

        frame_idx += 1

    cap.release()
    logger.info(
        "Sampled %s: %d frames extracted from %d native frames (step=%d, native_fps=%.1f)",
        video_id,
        extracted_count,
        total_native_frames,
        step,
        native_fps,
    )
    return frame_records


def sample_all_videos(
    video_dir: str | Path,
    raw_output_dir: str | Path,
    resized_output_dir: str | Path,
    target_fps: float = 10.0,
    target_size: tuple[int, int] = (200, 66),
) -> list[dict[str, Any]]:
    """Samples frames from all videos in video_dir."""
    video_dir = Path(video_dir)
    video_files = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm")))

    all_records: list[dict[str, Any]] = []
    logger.info("Found %d videos in %s", len(video_files), video_dir)

    for vid_file in video_files:
        records = sample_video_frames(
            video_path=vid_file,
            raw_output_dir=raw_output_dir,
            resized_output_dir=resized_output_dir,
            target_fps=target_fps,
            target_size=target_size,
        )
        all_records.extend(records)

    logger.info("Total frames sampled across all videos: %d", len(all_records))
    return all_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_dir = Path(__file__).resolve().parents[2]
    raw_vids = base_dir / "recording_dataset"
    raw_frames = base_dir / "main" / "data" / "extracted_frames"
    proc_images = base_dir / "main" / "data" / "processed" / "images"

    sample_all_videos(raw_vids, raw_frames, proc_images, target_fps=10.0)
