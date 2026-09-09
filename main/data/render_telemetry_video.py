"""Script to generate a telemetry animation video (at least 15 seconds).

Renders the camera video with:
1. Centered corridor guidance and optimal racing line trajectory projected onto the road.
2. High-visibility numeric steering angle HUD in the lower right corner, showing
   continuous angle, track mode (STRAIGHT, CROSSROAD, PEDESTRIAN, LEFT/RIGHT APEX),
   gauge, and confidence.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from main.data.geometry_estimator import GeometryEstimationResult, TrackGeometryEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def draw_hud_lower_right(
    frame: np.ndarray,
    angle_deg: float,
    angle_norm: float,
    confidence: float,
    frame_idx: int,
    total_frames: int,
    fps: float = 10.0,
    mode: str = "STRAIGHT",
) -> np.ndarray:
    """Draws a professional telemetry HUD in the lower right corner of the frame."""
    h, w = frame.shape[:2]

    # Card dimensions in lower right corner
    card_w, card_h = 360, 165
    margin_x, margin_y = 25, 25
    x1 = w - card_w - margin_x
    y1 = h - card_h - margin_y
    x2 = w - margin_x
    y2 = h - margin_y

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 30), -1)

    # Blend overlay with original frame (alpha = 0.80)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Color & direction badge based on mode and angle
    if "CROSSROAD" in mode or "PEDESTRIAN" in mode:
        accent_color = (80, 220, 100)  # Emerald Green
        direction_text = "CROSSROAD (0.0°)" if "CROSSROAD" in mode else "PEDESTRIAN (0.0°)"
    elif abs(angle_deg) < 3.5 or mode == "STRAIGHT":
        accent_color = (80, 220, 100)  # Emerald Green
        direction_text = "STRAIGHT (0.0°)"
    elif angle_deg < 0:
        accent_color = (40, 140, 255)  # Amber / Orange for Left Apex
        direction_text = "LEFT APEX"
    else:
        accent_color = (255, 180, 50)  # Cyan / Bright Blue for Right Apex
        direction_text = "RIGHT APEX"

    # Border outline
    cv2.rectangle(frame, (x1, y1), (x2, y2), accent_color, 2)

    # Header bar
    cv2.rectangle(frame, (x1, y1), (x2, y1 + 28), (35, 42, 54), -1)
    cv2.putText(
        frame,
        "NXP CUP 3D TELEMETRY",
        (x1 + 14, y1 + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (220, 230, 245),
        1,
        cv2.LINE_AA,
    )

    # Current time
    time_s = frame_idx / fps
    cv2.putText(
        frame,
        f"{time_s:04.1f}s",
        (x2 - 60, y1 + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        (180, 195, 210),
        1,
        cv2.LINE_AA,
    )

    # Big numeric readout in center of card
    sign = "+" if angle_deg > 0 else ""
    angle_str = f"{sign}{angle_deg:.2f}°"
    cv2.putText(
        frame,
        angle_str,
        (x1 + 18, y1 + 80),
        cv2.FONT_HERSHEY_DUPLEX,
        1.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Direction / Mode badge
    cv2.putText(
        frame,
        direction_text,
        (x1 + 185, y1 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        accent_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Norm: {angle_norm:+.3f}",
        (x1 + 185, y1 + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 185, 200),
        1,
        cv2.LINE_AA,
    )

    # Gauge bar background
    bar_y = y1 + 115
    bar_w = 310
    bar_x = x1 + 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 58, 70), -1)

    # Center tick mark (0°)
    center_tick_x = bar_x + bar_w // 2
    cv2.line(frame, (center_tick_x, bar_y - 3), (center_tick_x, bar_y + 15), (200, 200, 200), 2)

    # Active bar representing steer displacement
    clamped_norm = float(np.clip(angle_norm, -1.0, 1.0))
    bar_fill_w = int(clamped_norm * (bar_w // 2))
    if bar_fill_w > 0:
        cv2.rectangle(frame, (center_tick_x, bar_y), (center_tick_x + bar_fill_w, bar_y + 12), accent_color, -1)
    elif bar_fill_w < 0:
        cv2.rectangle(frame, (center_tick_x + bar_fill_w, bar_y), (center_tick_x, bar_y + 12), accent_color, -1)

    # Sub-footer info: gauge labels and confidence
    cv2.putText(
        frame,
        "-30°",
        (bar_x, bar_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (140, 150, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "0°",
        (center_tick_x - 6, bar_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (140, 150, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "+30°",
        (bar_x + bar_w - 24, bar_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (140, 150, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {int(confidence * 100)}%",
        (bar_x + 95, bar_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (80, 220, 100) if confidence > 0.6 else (0, 180, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_racing_trajectory_on_road(
    frame: np.ndarray,
    geom_res: GeometryEstimationResult,
    estimator: TrackGeometryEstimator,
    angle_deg: float,
) -> np.ndarray:
    """Projects centered corridor guidance and optimal racing line trajectory directly onto the road."""
    h, w = frame.shape[:2]
    inv_H = estimator._inv_homography_matrix
    if inv_H is None:
        return frame

    y_base = float(estimator.bev_h * 0.90)
    y_look = float(estimator.bev_h * 0.35)

    # Trajectory color
    if abs(angle_deg) < 3.5 or geom_res.mode in ("STRAIGHT", "CROSSROAD_OR_PEDESTRIAN"):
        accent_color = (80, 220, 100)  # Emerald Green
    elif angle_deg < 0:
        accent_color = (40, 140, 255)  # Amber / Orange for Left Apex
    else:
        accent_color = (255, 180, 50)  # Cyan / Blue for Right Apex

    # Sample points along the trajectory in BEV space
    n_pts = 16
    ys = np.linspace(y_base, y_look, n_pts)
    xs = np.linspace(geom_res.corridor_center_base, geom_res.racing_target_x, n_pts)

    bev_pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    cam_pts = cv2.perspectiveTransform(bev_pts, inv_H)
    cam_int = np.int32(cam_pts).reshape(-1, 2)

    # Filter points within camera frame bounds
    valid_mask = (
        (cam_int[:, 0] >= 0)
        & (cam_int[:, 0] < w)
        & (cam_int[:, 1] >= 0)
        & (cam_int[:, 1] < h)
    )
    cam_valid = cam_int[valid_mask]

    if len(cam_valid) >= 2:
        # Dark shadow outline for contrast against light road
        cv2.polylines(frame, [cam_valid], isClosed=False, color=(20, 24, 30), thickness=6, lineType=cv2.LINE_AA)
        # Main vibrant trajectory line
        cv2.polylines(frame, [cam_valid], isClosed=False, color=accent_color, thickness=3, lineType=cv2.LINE_AA)

        # Lookahead apex target circle
        target_pt = (int(cam_valid[-1, 0]), int(cam_valid[-1, 1]))
        cv2.circle(frame, target_pt, 8, (20, 24, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, target_pt, 6, accent_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, target_pt, 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return frame


def render_telemetry_video(
    video_path: str | Path,
    labels_csv_path: str | Path,
    output_mp4_path: str | Path,
    output_gif_path: str | Path | None = None,
    duration_s: float = 18.0,
    target_fps: float = 10.0,
) -> tuple[Path, Path | None]:
    """Renders the video with HUD and converts to browser-friendly formats."""
    video_path = Path(video_path)
    output_mp4_path = Path(output_mp4_path)
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # Load continuous labels
    df = pd.read_csv(labels_csv_path)
    vid_df = df[df["video_id"] == video_path.stem].sort_values("frame_idx")
    if len(vid_df) == 0:
        raise ValueError(f"No records found in CSV for video {video_path.stem}")

    num_frames_to_render = int(duration_s * target_fps)
    vid_df = vid_df.iloc[:num_frames_to_render]
    logger.info("Rendering %d frames (%.1f seconds at %.1f FPS)...", len(vid_df), len(vid_df) / target_fps, target_fps)

    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps != native_fps:
        native_fps = 30.0
    step = max(1, int(round(native_fps / target_fps)))

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_raw_mp4 = output_mp4_path.with_name("temp_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(temp_raw_mp4), fourcc, target_fps, (frame_w, frame_h))

    estimator = TrackGeometryEstimator()

    records_list = vid_df.to_dict("records")
    current_rec_idx = 0
    native_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or current_rec_idx >= len(records_list):
            break

        if native_frame_idx % step == 0:
            rec = records_list[current_rec_idx]
            angle_deg = float(rec["angle_deg"])
            angle_norm = float(rec["angle_norm"])
            confidence = float(rec["confidence"])
            mode = str(rec.get("mode", "STRAIGHT"))

            # 1. Compute geometry for road projection
            geom_res = estimator.estimate_steering(frame, return_visualization=False)

            # 2. Draw centered corridor & racing line apex trajectory on the road
            draw_racing_trajectory_on_road(frame, geom_res, estimator, angle_deg)

            # 3. Draw lower-right HUD telemetry card
            annotated_frame = draw_hud_lower_right(
                frame=frame,
                angle_deg=angle_deg,
                angle_norm=angle_norm,
                confidence=confidence,
                frame_idx=current_rec_idx,
                total_frames=len(records_list),
                fps=target_fps,
                mode=mode,
            )

            writer.write(annotated_frame)
            current_rec_idx += 1

        native_frame_idx += 1

    cap.release()
    writer.release()
    logger.info("Raw video rendered to %s", temp_raw_mp4)

    # Transcode to standard H.264 MP4 using ffmpeg for universal browser playback
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_raw_mp4),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-movflags",
        "+faststart",
        str(output_mp4_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    temp_raw_mp4.unlink(missing_ok=True)
    logger.info("Web-compatible MP4 saved to %s", output_mp4_path)

    # Also render an animated GIF (scaled to 640x360, 10 fps) for instant markdown embedding
    gif_path = None
    if output_gif_path is not None:
        gif_path = Path(output_gif_path)
        gif_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(output_mp4_path),
            "-vf",
            "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path),
        ]
        subprocess.run(gif_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Animated GIF saved to %s", gif_path)

    return output_mp4_path, gif_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render video telemetry animation HUD")
    base = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--video",
        type=str,
        default=str(base / "recording_dataset" / "WIN_20260202_15_57_23_Pro.mp4"),
        help="Path to input video file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(base / "main" / "data" / "processed" / "labels_continuous.csv"),
        help="Path to labels_continuous.csv",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=25.0,
        help="Duration in seconds to render",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target rendering FPS",
    )
    parser.add_argument(
        "--out-mp4",
        type=str,
        default=str(base / "main" / "data" / "telemetry_animation_15_57_23.mp4"),
        help="Output MP4 path",
    )
    parser.add_argument(
        "--out-gif",
        type=str,
        default=str(base / "main" / "data" / "telemetry_animation_15_57_23.gif"),
        help="Output GIF path",
    )

    args = parser.parse_args()

    render_telemetry_video(
        video_path=Path(args.video),
        labels_csv_path=Path(args.csv),
        output_mp4_path=Path(args.out_mp4),
        output_gif_path=Path(args.out_gif),
        duration_s=args.duration,
        target_fps=args.fps,
    )
