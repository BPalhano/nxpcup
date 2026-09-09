"""Temporal smoothing module for continuous steering angle sequences.

Applies outlier interpolation and Savitzky-Golay polynomial filtering
to eliminate high-frequency camera vibration while preserving real vehicle
curve dynamics across video laps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def smooth_lap_angles(
    frame_records: list[dict[str, Any]],
    confidence_threshold: float = 0.35,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    max_steering_deg: float = 30.0,
) -> list[dict[str, Any]]:
    """Applies interpolation and Savitzky-Golay smoothing to a video lap sequence.

    Args:
        frame_records: List of frame dicts with 'angle_deg', 'confidence', etc.
        confidence_threshold: Cutoff below which an angle is considered unreliable.
        savgol_window: Odd integer window size for Savitzky-Golay filter.
        savgol_polyorder: Polynomial order for Savitzky-Golay filter.
        max_steering_deg: Maximum steering angle limit (30.0°).

    Returns:
        list of updated frame dicts with smoothed 'angle_deg' and 'angle_norm'.
    """
    if len(frame_records) < 4:
        return frame_records

    angles = np.array([float(r["angle_deg"]) for r in frame_records], dtype=np.float64)
    confidences = np.array([float(r["confidence"]) for r in frame_records], dtype=np.float64)
    n = len(angles)

    # Identify unreliable frames
    low_conf_mask = confidences < confidence_threshold

    # Interpolate low-confidence frames using high-confidence neighbors
    if np.any(low_conf_mask):
        valid_indices = np.where(~low_conf_mask)[0]
        invalid_indices = np.where(low_conf_mask)[0]

        if len(valid_indices) >= 2:
            angles[invalid_indices] = np.interp(
                invalid_indices,
                valid_indices,
                angles[valid_indices],
            )
        elif len(valid_indices) == 1:
            angles[invalid_indices] = angles[valid_indices[0]]
        else:
            # Fallback if no frames were high confidence
            angles[:] = 0.0

    # Ensure window length is odd and does not exceed sequence length
    window = min(savgol_window, n if n % 2 != 0 else n - 1)
    if window >= savgol_polyorder + 2:
        smoothed_angles = savgol_filter(angles, window_length=window, polyorder=savgol_polyorder)
    else:
        smoothed_angles = angles

    # Clamp and update records
    smoothed_angles = np.clip(smoothed_angles, -max_steering_deg, max_steering_deg)

    for idx, rec in enumerate(frame_records):
        mode = str(rec.get("mode", ""))
        if "CROSSROAD" in mode or "PEDESTRIAN" in mode:
            # Under NXP Cup rules, car must proceed strictly straight (0.0°)
            final_deg = 0.0
        elif mode == "STRAIGHT" or abs(float(rec.get("angle_deg", 0.0))) == 0.0:
            # Straight sections must stay strictly straight (0.0°)
            final_deg = 0.0
        else:
            smoothed_deg = float(smoothed_angles[idx])
            final_deg = 0.0 if abs(smoothed_deg) < 3.5 else smoothed_deg

        rec["raw_angle_deg"] = rec["angle_deg"]
        rec["angle_deg"] = round(final_deg, 2)
        rec["angle_norm"] = round(final_deg / max_steering_deg, 4)

    return frame_records
