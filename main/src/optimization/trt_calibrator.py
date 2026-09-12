"""TensorRT INT8 Entropy Calibrator for NXP Cup Edge Quantization.

Reads representative track frames from main/data/processed/images/ and generates
an INT8 calibration cache using the IInt8EntropyCalibrator2 algorithm.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    BaseCalibrator = object
    TRT_AVAILABLE = False
    cuda: Any = None
    trt: Any = None
else:
    try:
        import pycuda.driver as cuda
        import tensorrt as trt

        TRT_AVAILABLE = True
        BaseCalibrator = trt.IInt8EntropyCalibrator2
    except ImportError:
        TRT_AVAILABLE = False
        BaseCalibrator = object
        cuda = None
        trt = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.trt_calibrator")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class NXPTrackCalibrator(BaseCalibrator):
    """Calibrator for TensorRT INT8 Post-Training Quantization (PTQ).

    Iterates over representative real-world road frames to calculate
    optimal dynamic quantization thresholds.
    """

    def __init__(
        self,
        images_dir: str | Path,
        cache_file: str | Path,
        batch_size: int = 16,
        max_batches: int = 32,  # 32 * 16 = 512 calibration images
        input_shape: tuple[int, int, int] = (3, 66, 200),
    ) -> None:
        if TRT_AVAILABLE:
            super().__init__()
        self.images_dir = Path(images_dir)
        self.cache_file = Path(cache_file)
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.input_shape = input_shape
        self.current_batch = 0

        self.image_paths = sorted(list(self.images_dir.glob("*.jpg")))[: batch_size * max_batches]
        logger.info(
            "Initialized INT8 Calibrator with %d frames (batch_size: %d, batches: %d)",
            len(self.image_paths),
            batch_size,
            max_batches,
        )

        self.device_input = None
        if TRT_AVAILABLE and cuda is not None:
            self.device_input = cuda.mem_alloc(
                batch_size * input_shape[0] * input_shape[1] * input_shape[2] * 4
            )

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str] | None = None) -> list[Any] | None:
        if self.current_batch >= self.max_batches or not self.image_paths:
            return None

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.image_paths))

        if start_idx >= end_idx:
            return None

        batch_data = []
        for i in range(start_idx, end_idx):
            img_path = self.image_paths[i]
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((self.input_shape[1], self.input_shape[2], 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            # Transpose HWC -> CHW
            img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
            batch_data.append(img)

        # Pad if last batch is smaller
        while len(batch_data) < self.batch_size:
            batch_data.append(batch_data[0])

        batch_np = np.ascontiguousarray(np.stack(batch_data), dtype=np.float32)
        self.current_batch += 1

        if TRT_AVAILABLE and cuda is not None and self.device_input is not None:
            cuda.memcpy_htod(self.device_input, batch_np)
            return [int(self.device_input)]

        return [batch_np]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_file.exists():
            logger.info("Reading calibration cache from: %s", self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing calibration cache to: %s", self.cache_file)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def generate_simulated_calibration_cache(output_file: str | Path) -> Path:
    """Generates a reference calibration cache file for environments without CUDA/TensorRT."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = b"TRT-8600-EntropyCalibration2\n"
    data = b"# Simulated INT8 Calibration Scale Table for NXP Cup\n"
    with open(output_path, "wb") as f:
        f.write(header + data)
    logger.info("Saved calibration cache reference at: %s", output_path)
    return output_path
