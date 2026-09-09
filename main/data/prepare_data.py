"""Data preparation script for NXP Cup dataset.

Cleans labels, associates image files in `resized_img_road/` with steering angles,
and generates `main/data/processed/labels_clean.csv`.
"""

import csv
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def map_class_to_angle(class_label: str) -> tuple[float, float]:
    """Map string class label to continuous steering angle (degrees and normalized).

    Args:
        class_label: Categorical string ("left_curve", "right_curve", "straight", "crossroad")

    Returns:
        tuple: (angle_deg, angle_norm) where angle_norm is in [-1.0, 1.0].
    """
    clean_label = class_label.strip().lower()
    if clean_label == "left_curve":
        return -30.0, -1.0
    elif clean_label == "right_curve":
        return 30.0, 1.0
    else:
        # Straight, crossroad, or other neutral road conditions
        return 0.0, 0.0


def prepare_dataset(
    source_labels_csv: Path,
    source_images_dir: Path,
    target_dir: Path,
) -> Path:
    """Prepares structured labels and ensures images are accessible.

    Args:
        source_labels_csv: Path to legacy labels.csv.
        source_images_dir: Path to resized_img_road/.
        target_dir: Destination path (e.g. main/data/processed).

    Returns:
        Path: Path to clean output CSV.
    """
    target_images_dir = target_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    output_csv = target_dir / "labels_clean.csv"

    # Map of existing files in source_images_dir
    existing_images = set(os.listdir(source_images_dir))
    logger.info("Found %d images in source directory: %s", len(existing_images), source_images_dir)

    # Link source images to target directory if not already linked
    for img_name in existing_images:
        target_img_path = target_images_dir / img_name
        if not target_img_path.exists():
            source_img_path = source_images_dir / img_name
            os.symlink(source_img_path.resolve(), target_img_path)

    matched_records = []
    missing_records = []

    with open(source_labels_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            raw_name, raw_class = row[0].strip(), row[1].strip()
            angle_deg, angle_norm = map_class_to_angle(raw_class)

            # Candidate filenames
            clean_stem = raw_name.replace("_traitee", "")
            candidate1 = f"{clean_stem}_200x66.jpg"
            candidate2 = f"{raw_name}_200x66.jpg"
            candidate3 = f"{raw_name}.jpg"
            candidate4 = f"{clean_stem}.jpg"

            resolved_filename = None
            for c in [candidate1, candidate2, candidate3, candidate4]:
                if c in existing_images:
                    resolved_filename = c
                    break

            if resolved_filename:
                matched_records.append(
                    {
                        "filename": resolved_filename,
                        "angle_deg": angle_deg,
                        "angle_norm": angle_norm,
                        "class_name": raw_class,
                    }
                )
            else:
                missing_records.append((row_idx, raw_name))

    logger.info(
        "Successfully matched %d / %d records",
        len(matched_records),
        len(matched_records) + len(missing_records),
    )
    if missing_records:
        logger.warning("%d records could not be matched with an image file", len(missing_records))

    # Write clean CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "angle_deg", "angle_norm", "class_name"])
        writer.writeheader()
        writer.writerows(matched_records)

    logger.info("Saved clean labels to %s", output_csv)
    return output_csv


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    src_labels = base_dir / "main" / "deep_learning" / "labels.csv"
    src_images = base_dir / "main" / "deep_learning" / "resized_img_road"
    dest_dir = base_dir / "main" / "data" / "processed"

    prepare_dataset(src_labels, src_images, dest_dir)
