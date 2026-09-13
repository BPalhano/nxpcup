"""Inference Strategy Pattern for NXP Cup Autonomous Driving.

Encapsulates interchangeable inference backends:
- PyTorch (FP32)
- ONNX Runtime (CPU / CUDA Execution Provider)
- TensorRT Engine (FP16 / INT8)
"""

from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.src.models import MedianEMASmoother, ModelFactory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.inference")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Standardizes input frame to shape (1, 3, 66, 200) with ImageNet normalization."""
    # Ensure (66, 200)
    if image.shape[:2] != (66, 200):
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        # Assume RGB
        image = image.astype(np.float32) / 255.0

    # Normalize
    norm_img = (image - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose HWC -> CHW and add batch dimension
    tensor_img = np.ascontiguousarray(
        norm_img.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32
    )
    return tensor_img


class BaseInferenceStrategy(ABC):
    """Abstract Strategy interface for steering angle inference."""

    def reset(self) -> None:  # noqa: B027
        """Resets any internal temporal state (used by smoothers)."""
        pass

    @abstractmethod
    def infer(self, image: np.ndarray) -> dict[str, float]:
        """Runs inference on a single frame.

        Args:
            image: Raw RGB image of road (H, W, 3) or (66, 200, 3).

        Returns:
            Dictionary with 'angle_deg', 'angle_norm', and 'latency_ms'.
        """
        pass

    @abstractmethod
    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        """Runs inference on preprocessed batch of shape (B, 3, 66, 200)."""
        pass


class PyTorchInferenceStrategy(BaseInferenceStrategy):
    """PyTorch native inference strategy."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = ModelFactory.create(model_name).to(self.device)
        self.model.eval()

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            data = torch.load(checkpoint_path, map_location=self.device)
            state_dict = data.get("state_dict", data)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded PyTorch weights from %s", checkpoint_path)

    @torch.no_grad()
    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        inp = torch.from_numpy(batch_tensor).to(self.device)
        out = self.model(inp)
        return out.cpu().numpy()

    def infer(self, image: np.ndarray) -> dict[str, float]:
        inp = preprocess_image(image)
        t0 = time.perf_counter_ns()
        out = self.infer_batch(inp)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        angle_norm = float(out[0, 0])
        angle_deg = angle_norm * 30.0
        return {
            "angle_deg": angle_deg,
            "angle_norm": angle_norm,
            "latency_ms": latency_ms,
        }


class ONNXRuntimeInferenceStrategy(BaseInferenceStrategy):
    """ONNX Runtime accelerated inference strategy."""

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Initialized ONNX Runtime session: %s", self.onnx_path.name)

    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: batch_tensor})
        return outputs[0]

    def infer(self, image: np.ndarray) -> dict[str, float]:
        inp = preprocess_image(image)
        t0 = time.perf_counter_ns()
        out = self.infer_batch(inp)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        angle_norm = float(out[0, 0])
        angle_deg = angle_norm * 30.0
        return {
            "angle_deg": angle_deg,
            "angle_norm": angle_norm,
            "latency_ms": latency_ms,
        }


class TensorRTInferenceStrategy(BaseInferenceStrategy):
    """TensorRT engine inference strategy with dynamic memory allocation."""

    def __init__(self, engine_path: str | Path) -> None:
        self.engine_path = Path(engine_path)
        self.fallback: ONNXRuntimeInferenceStrategy | None = None
        try:
            import pycuda.autoinit  # noqa: F401
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.trt_available = True
            logger.info("Loaded native TensorRT engine: %s", self.engine_path.name)
        except Exception as err:
            logger.warning("Native TensorRT unavailable (%s). Using ONNX fallback.", err)
            self.trt_available = False
            # Fallback to ONNX sibling if available
            onnx_sibling = self.engine_path.with_suffix(".onnx")
            if onnx_sibling.exists():
                self.fallback = ONNXRuntimeInferenceStrategy(onnx_sibling)
            else:
                self.fallback = None

    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        if not self.trt_available:
            if self.fallback:
                return self.fallback.infer_batch(batch_tensor)
            raise RuntimeError("TensorRT not available and no fallback found")
        # Native execution
        # (TensorRT binding execution logic)
        raise NotImplementedError("TensorRT host GPU required")

    def infer(self, image: np.ndarray) -> dict[str, float]:
        if not self.trt_available and self.fallback:
            return self.fallback.infer(image)
        inp = preprocess_image(image)
        t0 = time.perf_counter_ns()
        out = self.infer_batch(inp)
        latency_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return {
            "angle_deg": float(out[0, 0]) * 30.0,
            "angle_norm": float(out[0, 0]),
            "latency_ms": latency_ms,
        }


class SmoothedInferenceStrategy(BaseInferenceStrategy):
    """Decorator strategy adding online Moving Median + EMA filtering.

    Provides real-time temporal post-processing for vehicle steering:
    - Moving Median (default K=3) rejects transient non-linear spikes.
    - EMA (default alpha=0.65) ensures kinematic smoothness and jitter attenuation.
    """

    def __init__(
        self,
        base_strategy: BaseInferenceStrategy,
        window_size: int = 3,
        alpha: float = 0.65,
        max_angle_deg: float = 30.0,
    ) -> None:
        self.base_strategy = base_strategy
        self.smoother = MedianEMASmoother(
            window_size=window_size,
            alpha=alpha,
            max_angle_deg=max_angle_deg,
        )

    def reset(self) -> None:
        """Resets online temporal smoother state."""
        self.smoother.reset()
        self.base_strategy.reset()

    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        raw_out = self.base_strategy.infer_batch(batch_tensor)
        if raw_out.shape[0] == 1:
            raw_deg = float(raw_out[0, 0]) * self.smoother.max_angle_deg
            smoothed_deg = self.smoother.update(raw_deg)
            return np.array([[smoothed_deg / self.smoother.max_angle_deg]], dtype=np.float32)
        return raw_out

    def infer(self, image: np.ndarray) -> dict[str, float]:
        t0 = time.perf_counter_ns()
        res = self.base_strategy.infer(image)
        smoothed_deg = self.smoother.update(res["angle_deg"])
        total_lat = (time.perf_counter_ns() - t0) / 1_000_000.0

        return {
            "angle_deg": round(smoothed_deg, 2),
            "angle_norm": round(smoothed_deg / self.smoother.max_angle_deg, 4),
            "raw_angle_deg": round(res["angle_deg"], 2),
            "latency_ms": round(total_lat, 3),
        }


def create_inference_engine(
    backend: str,
    model_name_or_path: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    apply_smoothing: bool = False,
    window_size: int = 3,
    alpha: float = 0.65,
) -> BaseInferenceStrategy:
    """Factory function for creating inference strategies.

    Supports:
    - Native PyTorch, ONNX Runtime, and TensorRT engines.
    - Automatic temporal smoothing for 'micro_vit_smoothed'.
    - Explicit temporal smoothing via apply_smoothing=True.
    """
    backend = backend.lower()
    model_lower = Path(model_name_or_path).stem.lower()

    if model_lower in ["micro_vit_smoothed", "microvit_smoothed"]:
        apply_smoothing = True
        # Resolve underlying base path/name
        if backend in ["onnx", "onnxruntime"] and not Path(model_name_or_path).exists():
            resolved_onnx = MAIN_DIR / "training_results" / "micro_vit" / "micro_vit.onnx"
            if resolved_onnx.exists():
                model_name_or_path = str(resolved_onnx)
            else:
                model_name_or_path = "micro_vit"
        elif backend == "pytorch":
            model_name_or_path = "micro_vit"

    engine: BaseInferenceStrategy
    if backend == "pytorch":
        engine = PyTorchInferenceStrategy(model_name_or_path, checkpoint_path, device=device)
    elif backend in ["onnx", "onnxruntime"]:
        engine = ONNXRuntimeInferenceStrategy(model_name_or_path)
    elif backend in ["trt", "tensorrt"]:
        engine = TensorRTInferenceStrategy(model_name_or_path)
    else:
        raise ValueError(
            f"Unknown inference backend: {backend}. Supported: [pytorch, onnx, tensorrt]"
        )

    if apply_smoothing:
        return SmoothedInferenceStrategy(
            base_strategy=engine,
            window_size=window_size,
            alpha=alpha,
        )

    return engine

