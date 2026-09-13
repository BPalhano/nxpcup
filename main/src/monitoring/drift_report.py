"""Data Drift & Model Resilience Monitoring with Evidently AI.

Sprint 4 Deliverable for NXP Cup Autonomous Driving:
1. Splits or samples test frames into Baseline (Group A: clean lighting) and
   Drift Condition (Group B: dimmed lighting + realistic cast shadows).
2. Extracts domain-specific computer vision statistical features:
   - Mean brightness, contrast, RGB channel distributions
   - Edge density, Laplacian sharpness, HSV saturation, dark pixel fraction
3. Executes multi-model inference (MicroViT vs PalhanoCNN vs DAVE-2) across both groups.
4. Generates an interactive Evidently AI HTML Report and a structured JSON summary
   proving MicroViT's resilience under adverse illumination.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.data.dataset import load_dataset_records, split_dataset_records  # noqa: E402
from main.src.inference.strategy import create_inference_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.drift_report")


def apply_synthetic_drift(
    image_rgb: np.ndarray,
    seed: int,
    brightness_factor: float = 0.60,
    shadow_darkness: float = 0.45,
) -> np.ndarray:
    """Applies realistic NXP Cup environmental drift to an RGB road frame.

    Simulates:
    1. Global illumination drop (dim indoor track, cloud cover).
    2. Sharp diagonal cast shadows from track barriers and track-side personnel.

    Args:
        image_rgb: Input RGB image (H, W, 3) in [0, 255].
        seed: Random seed for deterministic reproducibility.
        brightness_factor: Multiplier for global illumination reduction.
        shadow_darkness: Residual brightness fraction in the shadow polygon.

    Returns:
        Drifted RGB image (H, W, 3) as uint8.
    """
    img = image_rgb.copy()
    h, w, _ = img.shape
    rng = np.random.RandomState(seed)

    # 1. Global illumination reduction
    img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    # 2. Add realistic cast shadow polygon across track corridor
    mask = np.ones((h, w), dtype=np.float32)
    x1 = int(rng.randint(0, max(1, w // 2)))
    x2 = int(rng.randint(x1 + 25, min(w, x1 + 110)))
    poly = np.array(
        [
            [x1, 0],
            [x2, 0],
            [min(w, x2 + 35), h],
            [max(0, x1 - 35), h],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [poly], (float(shadow_darkness),))

    # 3. Soft blur on shadow boundary to mimic optical penumbra
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    drifted = np.clip(img.astype(np.float32) * mask_blurred[:, :, np.newaxis], 0, 255).astype(
        np.uint8
    )
    return drifted


def extract_image_features(image_rgb: np.ndarray) -> dict[str, float]:
    """Extracts numerical computer vision features for data drift evaluation."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    mean_r = float(np.mean(image_rgb[:, :, 0]))
    mean_g = float(np.mean(image_rgb[:, :, 1]))
    mean_b = float(np.mean(image_rgb[:, :, 2]))

    mean_saturation = float(np.mean(hsv[:, :, 1]))

    # Laplacian variance for edge sharpness
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Canny edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Fraction of dark pixels (Y < 50)
    dark_pixel_fraction = float(np.count_nonzero(gray < 50) / gray.size)

    return {
        "mean_brightness": round(mean_brightness, 2),
        "std_contrast": round(std_contrast, 2),
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "mean_saturation": round(mean_saturation, 2),
        "laplacian_var": round(laplacian_var, 2),
        "edge_density": round(edge_density, 4),
        "dark_pixel_fraction": round(dark_pixel_fraction, 4),
    }


def plot_feature_distributions(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plots comparative histograms showing statistical feature shift under drift."""
    features_to_plot = [
        ("mean_brightness", "Mean Brightness (0-255)"),
        ("std_contrast", "Contrast (Std Dev)"),
        ("dark_pixel_fraction", "Dark Pixel Ratio (<50)"),
        ("edge_density", "Canny Edge Density"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(features_to_plot):
        ax = axes[idx]
        ax.hist(
            ref_df[col],
            bins=25,
            alpha=0.65,
            label="Baseline (Group A)",
            color="#1f77b4",
            density=True,
            edgecolor="black",
        )
        ax.hist(
            curr_df[col],
            bins=25,
            alpha=0.65,
            label="Drift Condition (Group B)",
            color="#d62728",
            density=True,
            edgecolor="black",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.suptitle(
        "NXP Cup Track Feature Distribution Shift: Baseline vs. Drift (Low Light + Shadows)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_resilience_comparison(
    resilience_data: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Plots comparative bar chart of model MAE under Baseline vs. Drift Condition."""
    models = list(resilience_data.keys())
    baseline_maes = [resilience_data[m]["mae_baseline"] for m in models]
    drift_maes = [resilience_data[m]["mae_drift"] for m in models]
    deltas = [resilience_data[m]["mae_degradation"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x - width / 2,
        baseline_maes,
        width,
        label="Baseline (Clean)",
        color="#2ca02c",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        drift_maes,
        width,
        label="Drift (Shadows & Dim)",
        color="#d62728",
        edgecolor="black",
    )

    ax.set_ylabel("Mean Absolute Error (Degrees °)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Resilience Under Environmental Drift (Shadows & Low-Light)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    formatted_labels = [
        m.replace("_", " ").title().replace("Dave2", "DAVE-2 (NVIDIA)").replace("Cnn", "CNN")
        for m in models
    ]
    ax.set_xticklabels(formatted_labels, fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add text labels on bars
    for i in range(len(models)):
        b1_val = baseline_maes[i]
        b2_val = drift_maes[i]
        delta_val = deltas[i]
        ax.text(
            x[i] - width / 2,
            b1_val + 0.15,
            f"{b1_val:.2f}°",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            x[i] + width / 2,
            b2_val + 0.15,
            f"{b2_val:.2f}°\n(+{delta_val:.2f}°)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_drift_report(
    csv_path: Path,
    images_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path, Path, Path]:
    """Orchestrates data drift generation, multi-model evaluation, and Evidently reporting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = MAIN_DIR / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test dataset records from: %s", csv_path)
    records = load_dataset_records(csv_path)
    _, _, test_records = split_dataset_records(records, seed=seed)

    if max_samples is not None and max_samples < len(test_records):
        test_records = test_records[:max_samples]

    logger.info(
        "Evaluating %d test frames for Data Drift and Model Resilience...", len(test_records)
    )

    # 1. Load Inference Engines for Clean and Drift Streams
    engines_clean = {}
    engines_drift = {}
    model_names = ["dave2", "palhano_cnn", "micro_vit", "micro_vit_smoothed"]

    for m in model_names:
        if m == "micro_vit_smoothed":
            base_onnx = MAIN_DIR / "training_results" / "micro_vit" / "micro_vit.onnx"
            base_ckpt = MAIN_DIR / "training_results" / "micro_vit" / "best_model.pt"
            if base_onnx.exists():
                logger.info("Using smoothed ONNX engine for %s: %s", m, base_onnx)
                engines_clean[m] = create_inference_engine(
                    backend="onnx", model_name_or_path=str(base_onnx), apply_smoothing=True
                )
                engines_drift[m] = create_inference_engine(
                    backend="onnx", model_name_or_path=str(base_onnx), apply_smoothing=True
                )
            elif base_ckpt.exists():
                logger.info("Using smoothed PyTorch engine for %s: %s", m, base_ckpt)
                engines_clean[m] = create_inference_engine(
                    backend="pytorch",
                    model_name_or_path="micro_vit",
                    checkpoint_path=str(base_ckpt),
                    apply_smoothing=True,
                )
                engines_drift[m] = create_inference_engine(
                    backend="pytorch",
                    model_name_or_path="micro_vit",
                    checkpoint_path=str(base_ckpt),
                    apply_smoothing=True,
                )
            else:
                engines_clean[m] = create_inference_engine(
                    backend="pytorch", model_name_or_path="micro_vit", apply_smoothing=True
                )
                engines_drift[m] = create_inference_engine(
                    backend="pytorch", model_name_or_path="micro_vit", apply_smoothing=True
                )
        else:
            onnx_candidate = MAIN_DIR / "training_results" / m / f"{m}.onnx"
            ckpt_candidate = MAIN_DIR / "training_results" / m / "best_model.pt"

            if onnx_candidate.exists():
                logger.info("Using ONNX engine for %s: %s", m, onnx_candidate)
                engines_clean[m] = create_inference_engine(
                    backend="onnx", model_name_or_path=str(onnx_candidate)
                )
                engines_drift[m] = create_inference_engine(
                    backend="onnx", model_name_or_path=str(onnx_candidate)
                )
            elif ckpt_candidate.exists():
                logger.info("Using PyTorch engine for %s: %s", m, ckpt_candidate)
                engines_clean[m] = create_inference_engine(
                    backend="pytorch", model_name_or_path=m, checkpoint_path=str(ckpt_candidate)
                )
                engines_drift[m] = create_inference_engine(
                    backend="pytorch", model_name_or_path=m, checkpoint_path=str(ckpt_candidate)
                )
            else:
                logger.warning("No checkpoint found for %s, creating default model.", m)
                engines_clean[m] = create_inference_engine(backend="pytorch", model_name_or_path=m)
                engines_drift[m] = create_inference_engine(backend="pytorch", model_name_or_path=m)

    # Ensure records have video_id and frame_idx and sort chronologically
    pattern = re.compile(r"^(.*)_frame_(\d+)_.*$")
    for rec in test_records:
        if "video_id" not in rec or not rec["video_id"]:
            match = pattern.match(rec["filename"])
            if match:
                rec["video_id"] = match.group(1)
                rec["frame_idx"] = int(match.group(2))
            else:
                rec["video_id"] = ""
                rec["frame_idx"] = 0

    test_records.sort(key=lambda r: (str(r.get("video_id", "")), int(r.get("frame_idx", 0))))

    ref_rows: list[dict[str, Any]] = []
    curr_rows: list[dict[str, Any]] = []

    current_video_id: str | None = None
    t0 = time.time()
    for idx, rec in enumerate(test_records):
        img_path = images_dir / rec["filename"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        vid = str(rec.get("video_id", ""))
        if vid != current_video_id:
            current_video_id = vid
            for m in model_names:
                engines_clean[m].reset()
                engines_drift[m].reset()

        clean_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        drift_img = apply_synthetic_drift(clean_img, seed=seed + idx)

        gt_angle = float(rec["angle_deg"])

        # Extract CV features
        ref_feats = extract_image_features(clean_img)
        curr_feats = extract_image_features(drift_img)

        ref_row = {
            "filename": rec["filename"],
            "target_steer_deg": gt_angle,
            "lateral_offset_px": float(rec.get("lateral_offset_px", 0.0)),
            **ref_feats,
        }
        curr_row = {
            "filename": rec["filename"],
            "target_steer_deg": gt_angle,
            "lateral_offset_px": float(rec.get("lateral_offset_px", 0.0)),
            **curr_feats,
        }

        # Run multi-model predictions across both streams
        for m in model_names:
            p_clean = engines_clean[m].infer(clean_img)["angle_deg"]
            p_drift = engines_drift[m].infer(drift_img)["angle_deg"]

            err_clean = abs(p_clean - gt_angle)
            err_drift = abs(p_drift - gt_angle)

            ref_row[f"pred_{m}"] = round(p_clean, 2)
            ref_row[f"abs_err_{m}"] = round(err_clean, 2)

            curr_row[f"pred_{m}"] = round(p_drift, 2)
            curr_row[f"abs_err_{m}"] = round(err_drift, 2)

        ref_rows.append(ref_row)
        curr_rows.append(curr_row)

    ref_df = pd.DataFrame(ref_rows)
    curr_df = pd.DataFrame(curr_rows)

    elapsed = time.time() - t0
    logger.info("Processed %d samples in %.2fs (%.1f FPS)", len(ref_df), elapsed, (len(ref_df) * 2) / elapsed)

    # 2. Compute Resilience Metrics
    resilience_summary: dict[str, dict[str, float]] = {}
    for m in model_names:
        mae_base = float(ref_df[f"abs_err_{m}"].mean())
        mae_drift = float(curr_df[f"abs_err_{m}"].mean())
        delta_mae = mae_drift - mae_base
        degrade_pct = (delta_mae / max(mae_base, 1e-6)) * 100.0

        resilience_summary[m] = {
            "mae_baseline": round(mae_base, 2),
            "mae_drift": round(mae_drift, 2),
            "mae_degradation": round(delta_mae, 2),
            "degradation_percent": round(degrade_pct, 1),
        }
        logger.info(
            "Model Resilience [%s]: Base MAE = %.2f° | Drift MAE = %.2f° | Degradation = +%.2f° (+%.1f%%)",
            m,
            mae_base,
            mae_drift,
            delta_mae,
            degrade_pct,
        )

    # 3. Generate Evidently AI Report
    logger.info("Generating Evidently AI Data Drift Report...")
    # Exclude metadata columns from statistical test
    cols_to_test = [
        c
        for c in ref_df.columns
        if c not in ["filename", "target_steer_deg", "lateral_offset_px"]
    ]
    ref_sub = ref_df[cols_to_test]
    curr_sub = curr_df[cols_to_test]

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_sub, current_data=curr_sub)

    html_primary = reports_dir / "data_drift_report.html"
    snapshot.save_html(str(html_primary))
    logger.info("Saved primary Evidently report to: %s", html_primary)

    # Mirror to training_results
    html_mirrored = output_dir / "data_drift_report.html"
    shutil.copy2(html_primary, html_mirrored)

    # 4. Generate Visual Plots
    dist_plot_path = output_dir / "drift_feature_distribution.png"
    plot_feature_distributions(ref_df, curr_df, dist_plot_path)

    resilience_plot_path = output_dir / "model_resilience_comparison.png"
    plot_resilience_comparison(resilience_summary, resilience_plot_path)

    # 5. Save Structured JSON Summary
    summary_json_path = output_dir / "data_drift_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_evaluated": len(ref_df),
        "drift_conditions": {
            "global_dimming_factor": 0.60,
            "cast_shadow_polygon": True,
            "shadow_darkness": 0.45,
        },
        "model_resilience": resilience_summary,
        "features_evaluated": cols_to_test,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("✅ Sprint 4 Data Drift analysis successfully complete!")
    logger.info("  HTML Report  : %s", html_mirrored)
    logger.info("  Summary JSON : %s", summary_json_path)
    logger.info("  Dist Plot    : %s", dist_plot_path)
    logger.info("  Resilience   : %s", resilience_plot_path)

    return html_mirrored, summary_json_path, dist_plot_path, resilience_plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidently AI Data Drift Analysis for NXP Cup")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(MAIN_DIR / "data" / "processed" / "labels_continuous.csv"),
        help="Path to CSV labels file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(MAIN_DIR / "data" / "processed" / "images"),
        help="Path to images directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MAIN_DIR / "training_results"),
        help="Output directory for reports and charts",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test samples to process",
    )
    args = parser.parse_args()

    generate_drift_report(
        csv_path=Path(args.csv_path),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
