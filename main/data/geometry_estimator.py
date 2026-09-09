"""3D Scene-Aware Track Geometry, Corridor Centering & Racing Line Estimator.

Implements autonomous trajectory auto-labeling for the NXP Cup track:
1. Inverse Perspective Mapping (IPM / Bird's Eye View) with strict trapezoidal masking.
2. Corridor Centering: Tracks the white corridor between the two black boundary lines
   (never clinging to a single boundary line).
3. NXP Cup Competition Track Rules:
   - Crossroads (90° intersections): Car must proceed strictly straight (0.0°).
   - Pedestrian / Start-Finish multi-stripe hash marks: Car must proceed strictly straight (0.0°).
   - Straight track (seguir em frente): Filtered with deadband to strictly 0.0°.
   - Curves (left, right, chicanes): Calculates the optimal racing line cutting towards the
     inner apex (tangenciamento mais rápido) to maximize cornering velocity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometryEstimationResult:
    """Result of geometric steering angle estimation for a single frame."""

    angle_deg: float
    angle_norm: float  # Normalized in [-1.0, 1.0]
    confidence: float  # Score in [0.0, 1.0]
    curvature: float
    lateral_offset_px: float
    left_line_detected: bool
    right_line_detected: bool
    mode: str = "STRAIGHT"  # "STRAIGHT", "LEFT_CURVE", "RIGHT_CURVE", "CROSSROAD_OR_PEDESTRIAN"
    corridor_center_base: float = 0.0
    corridor_center_lookahead: float = 0.0
    racing_target_x: float = 0.0
    bev_image: np.ndarray | None = None
    annotated_image: np.ndarray | None = None


class TrackGeometryEstimator:
    """Estimates continuous vehicle steering angle with corridor centering and racing line."""

    def __init__(
        self,
        bev_width: int = 400,
        bev_height: int = 400,
        lookahead_ratio: float = 0.65,
        max_steering_deg: float = 30.0,
        wheelbase_px: float = 120.0,
        deadband_deg: float = 3.5,
        apex_cut_ratio: float = 0.28,
    ) -> None:
        """Initialize TrackGeometryEstimator.

        Args:
            bev_width: Width of the warped Bird's Eye View image.
            bev_height: Height of the warped Bird's Eye View image.
            lookahead_ratio: Fraction of BEV height used for lookahead distance.
            max_steering_deg: Physical maximum steering angle (30.0°).
            wheelbase_px: Effective wheelbase metric in BEV pixel space.
            deadband_deg: Angle threshold below which track is classified as straight (0.0°).
            apex_cut_ratio: Fraction of half-track width shifted toward inside apex.
        """
        self.bev_w = bev_width
        self.bev_h = bev_height
        self.lookahead_y = int(self.bev_h * (1.0 - lookahead_ratio))
        self.max_steering_deg = max_steering_deg
        self.wheelbase = wheelbase_px
        self.deadband_deg = deadband_deg
        self.apex_cut_ratio = apex_cut_ratio

        self._homography_matrix: np.ndarray | None = None
        self._inv_homography_matrix: np.ndarray | None = None
        self._mask_bev: np.ndarray | None = None

    def _get_homography_and_mask(self, frame_w: int, frame_h: int) -> tuple[np.ndarray, np.ndarray]:
        """Computes and caches IPM homography matrix and valid BEV mask."""
        if self._homography_matrix is not None and self._mask_bev is not None:
            return self._homography_matrix, self._mask_bev

        # Trapezoidal source points on the road in the camera view (bottom half)
        src_points = np.array(
            [
                [frame_w * 0.20, frame_h * 0.48],  # Top-left
                [frame_w * 0.80, frame_h * 0.48],  # Top-right
                [frame_w * 0.98, frame_h * 0.95],  # Bottom-right
                [frame_w * 0.02, frame_h * 0.95],  # Bottom-left
            ],
            dtype=np.float32,
        )

        # Rectangular target points in Bird's Eye View
        dst_points = np.array(
            [
                [self.bev_w * 0.15, 0],
                [self.bev_w * 0.85, 0],
                [self.bev_w * 0.85, self.bev_h],
                [self.bev_w * 0.15, self.bev_h],
            ],
            dtype=np.float32,
        )

        self._homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self._inv_homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # Build binary mask of the valid projected road area to discard outer gym floor
        mask_cam = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(mask_cam, [src_points.astype(np.int32)], (255,))
        self._mask_bev = cv2.warpPerspective(mask_cam, self._homography_matrix, (self.bev_w, self.bev_h))

        return self._homography_matrix, self._mask_bev

    def compute_bev(self, frame: np.ndarray) -> np.ndarray:
        """Warps camera frame into top-down Bird's Eye View (IPM)."""
        h, w = frame.shape[:2]
        H, _ = self._get_homography_and_mask(w, h)
        return cv2.warpPerspective(frame, H, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)

    def detect_transverse_pattern(self, bev_gray: np.ndarray, mask_bev: np.ndarray) -> bool:
        """Detects crossroads (90° intersections) or pedestrian-style transversal stripes.

        In NXP Cup rules, the car must proceed strictly straight (0.0°) through these sections.
        """
        road_mask = mask_bev > 200
        dark_mask = ((bev_gray < 100) & road_mask).astype(np.uint8) * 255

        # Horizontal morphological line kernel
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
        h_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, h_kernel)

        # Inspect central corridor: columns 170 to 230 in BEV
        center_stripe = h_lines[:, 170:230]
        row_cov = np.sum(center_stripe > 0, axis=1)
        # Rows with transverse line covering >= 45 px across central corridor
        transverse_rows = np.count_nonzero(row_cov >= 45)
        return bool(transverse_rows >= 10)

    def extract_corridor_slices(
        self,
        bev_gray: np.ndarray,
        mask_bev: np.ndarray,
        num_slices: int = 10,
    ) -> list[tuple[float, float, float, float, float]]:
        """Segments the drivable white track corridor between the two dark boundary lines.

        Returns:
            List of tuples: (center_x, y_center, corridor_width, left_edge, right_edge)
        """
        road_mask = mask_bev > 200
        white_mask = (bev_gray > 135) & road_mask

        slice_h = self.bev_h // num_slices
        slices = []

        for i in range(num_slices):
            y1 = self.bev_h - (i + 1) * slice_h
            y2 = self.bev_h - i * slice_h
            y_center = (y1 + y2) / 2.0

            slice_white = white_mask[y1:y2, :]
            row_counts = np.sum(slice_white, axis=0)
            white_cols = np.where(row_counts > slice_h * 0.35)[0]

            if len(white_cols) > 30:
                xl = float(white_cols[0])
                xr = float(white_cols[-1])
                cw = xr - xl
                if cw > 60:
                    xc = (xl + xr) / 2.0
                    slices.append((xc, y_center, cw, xl, xr))

        return slices

    def estimate_steering(
        self,
        frame: np.ndarray,
        return_visualization: bool = False,
    ) -> GeometryEstimationResult:
        """Estimates continuous steering angle from input frame with racing line apex clipping."""
        h, w = frame.shape[:2]
        H, mask_bev = self._get_homography_and_mask(w, h)
        bev = cv2.warpPerspective(frame, H, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        mid_x = self.bev_w // 2

        # 1. Guard clause: blank or uniform dark frames
        if bev_gray.max() < 40 or float(np.std(bev_gray)) < 8.0:
            return GeometryEstimationResult(
                angle_deg=0.0,
                angle_norm=0.0,
                confidence=0.0,
                curvature=0.0,
                lateral_offset_px=0.0,
                left_line_detected=False,
                right_line_detected=False,
                mode="BLANK_FRAME",
                bev_image=bev if return_visualization else None,
                annotated_image=frame.copy() if return_visualization else None,
            )

        # 2. Check for crossroads or pedestrian transversal stripes
        is_transverse = self.detect_transverse_pattern(bev_gray, mask_bev)
        if is_transverse:
            annotated_img = None
            if return_visualization:
                annotated_img = frame.copy()
                cv2.putText(
                    annotated_img,
                    "NXP CUP: CROSSROAD / PEDESTRIAN -> STRICTLY STRAIGHT (0.0 deg)",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 220, 100),
                    2,
                )

            return GeometryEstimationResult(
                angle_deg=0.0,
                angle_norm=0.0,
                confidence=1.0,
                curvature=0.0,
                lateral_offset_px=0.0,
                left_line_detected=True,
                right_line_detected=True,
                mode="CROSSROAD_OR_PEDESTRIAN",
                corridor_center_base=float(mid_x),
                corridor_center_lookahead=float(mid_x),
                racing_target_x=float(mid_x),
                bev_image=bev if return_visualization else None,
                annotated_image=annotated_img,
            )

        # 3. Segment white corridor slices
        slices = self.extract_corridor_slices(bev_gray, mask_bev)

        if len(slices) < 4:
            return GeometryEstimationResult(
                angle_deg=0.0,
                angle_norm=0.0,
                confidence=0.15,
                curvature=0.0,
                lateral_offset_px=0.0,
                left_line_detected=False,
                right_line_detected=False,
                mode="FALLBACK_UNCERTAIN",
                corridor_center_base=float(mid_x),
                corridor_center_lookahead=float(mid_x),
                racing_target_x=float(mid_x),
                bev_image=bev if return_visualization else None,
                annotated_image=frame.copy() if return_visualization else None,
            )

        pts_x = np.array([s[0] for s in slices])
        pts_y = np.array([s[1] for s in slices])
        widths = np.array([s[2] for s in slices])

        # Fit 2nd-degree polynomial: x(y) = a*y^2 + b*y + c
        poly_coeffs = np.polyfit(pts_y, pts_x, deg=2)

        y_base = float(self.bev_h * 0.90)
        y_look = float(self.bev_h * 0.35)

        base_center_x = float(np.polyval(poly_coeffs, y_base))
        look_center_x = float(np.polyval(poly_coeffs, y_look))
        avg_w = float(np.mean(widths))

        # Lateral displacement & centerline trajectory angle
        lateral_offset = base_center_x - mid_x
        delta_x = look_center_x - mid_x
        delta_y = y_base - y_look
        raw_angle_deg = float(np.degrees(np.arctan2(delta_x, delta_y)))

        # 4. Deadband for straight section: maintain strictly 0.0°
        if abs(raw_angle_deg) < self.deadband_deg:
            annotated_img = None
            if return_visualization:
                annotated_img = frame.copy()
                cv2.putText(
                    annotated_img,
                    f"Straight Corridor (0.0 deg) | Offset: {lateral_offset:+.1f}px",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (80, 220, 100),
                    2,
                )

            return GeometryEstimationResult(
                angle_deg=0.0,
                angle_norm=0.0,
                confidence=0.96,
                curvature=0.0,
                lateral_offset_px=round(lateral_offset, 2),
                left_line_detected=True,
                right_line_detected=True,
                mode="STRAIGHT",
                corridor_center_base=round(base_center_x, 1),
                corridor_center_lookahead=round(look_center_x, 1),
                racing_target_x=round(look_center_x, 1),
                bev_image=bev if return_visualization else None,
                annotated_image=annotated_img,
            )

        # 5. Racing Line in Curves: Cut towards the inner apex
        apex_clip = self.apex_cut_ratio * (avg_w / 2.0)
        if raw_angle_deg < 0:
            # Curve to the left -> cut towards inside left edge
            target_x = look_center_x - apex_clip
            mode = "LEFT_CURVE"
        else:
            # Curve to the right -> cut towards inside right edge
            target_x = look_center_x + apex_clip
            mode = "RIGHT_CURVE"

        # Pure pursuit curvature & steering angle to optimal racing target
        racing_dx = target_x - mid_x
        lookahead_dist_sq = delta_y**2 + racing_dx**2
        curvature = float(2.0 * racing_dx / max(1.0, lookahead_dist_sq))

        steer_rad = np.arctan2(racing_dx, delta_y) * 1.25
        raw_racing_angle_deg = float(np.degrees(steer_rad))
        clamped_deg = float(np.clip(raw_racing_angle_deg, -self.max_steering_deg, self.max_steering_deg))
        normalized_angle = clamped_deg / self.max_steering_deg

        confidence = float(min(1.0, len(slices) / 10.0 * 1.1))

        annotated_img = None
        if return_visualization:
            annotated_img = frame.copy()
            accent = (40, 140, 255) if clamped_deg < 0 else (255, 180, 50)
            cv2.putText(
                annotated_img,
                f"{mode}: {clamped_deg:+.1f} deg (Racing Apex Target)",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                accent,
                2,
            )

        return GeometryEstimationResult(
            angle_deg=round(clamped_deg, 2),
            angle_norm=round(normalized_angle, 4),
            confidence=round(confidence, 3),
            curvature=round(curvature, 6),
            lateral_offset_px=round(lateral_offset, 2),
            left_line_detected=True,
            right_line_detected=True,
            mode=mode,
            corridor_center_base=round(base_center_x, 1),
            corridor_center_lookahead=round(look_center_x, 1),
            racing_target_x=round(target_x, 1),
            bev_image=bev if return_visualization else None,
            annotated_image=annotated_img,
        )
