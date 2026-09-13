"""FastAPI Steering Inference Service for NXP Cup Autonomous Driving.

Provides high-throughput, low-latency REST endpoints for real-time steering prediction:
- GET /health: Service health and system telemetry.
- GET /models: Architectural specs and metrics of available models.
- POST /steer: Real-time steering prediction from multipart image file.
- POST /steer/base64: Real-time steering prediction from Base64 encoded image.
- POST /reset: Resets online temporal smoother buffers between laps.
"""

from __future__ import annotations

import base64
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.src.inference.strategy import (  # noqa: E402
    BaseInferenceStrategy,
    create_inference_engine,
)

logger = logging.getLogger("nxpcup.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

START_TIME = time.time()

AVAILABLE_MODELS = [
    "micro_vit_smoothed",
    "micro_vit",
    "palhano_cnn",
    "dave2",
]
DEFAULT_MODEL = "micro_vit_smoothed"
DEFAULT_BACKEND = "onnx"

# Engine cache: (model_name, backend) -> BaseInferenceStrategy
ENGINE_CACHE: dict[tuple[str, str], BaseInferenceStrategy] = {}


def get_model_path(model_name: str, backend: str) -> str:
    """Resolves local weights or ONNX graph path for a given model."""
    results_dir = MAIN_DIR / "training_results"
    model_dir = results_dir / model_name.replace("_smoothed", "")

    if backend in ["onnx", "onnxruntime"]:
        onnx_file = model_dir / f"{model_name.replace('_smoothed', '')}.onnx"
        if onnx_file.exists():
            return str(onnx_file)
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_file}")
    elif backend == "pytorch":
        pt_file = model_dir / "best_model.pt"
        if pt_file.exists():
            return str(pt_file)
        # Fallback to initialized factory
        return model_name
    return model_name


def get_engine(model_name: str, backend: str) -> BaseInferenceStrategy:
    """Retrieves or creates a cached inference engine."""
    model_name = model_name.lower()
    backend = backend.lower()

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Invalid model '{model_name}'. Available: {AVAILABLE_MODELS}"
        )
    if backend not in ["onnx", "pytorch"]:
        raise ValueError(f"Invalid backend '{backend}'. Supported: ['onnx', 'pytorch']")

    cache_key = (model_name, backend)
    if cache_key in ENGINE_CACHE:
        return ENGINE_CACHE[cache_key]

    try:
        model_path = get_model_path(model_name, backend)
        checkpoint_path = model_path if backend == "pytorch" and model_path.endswith(".pt") else None
        target_model = model_name if backend == "pytorch" else model_path

        engine = create_inference_engine(
            backend=backend,
            model_name_or_path=target_model,
            checkpoint_path=checkpoint_path,
        )
        ENGINE_CACHE[cache_key] = engine
        logger.info("Initialized and cached inference engine: %s (%s)", model_name, backend)
        return engine
    except Exception as err:
        logger.warning("Failed to initialize backend '%s' for '%s': %s. Trying PyTorch fallback.", backend, model_name, err)
        if backend != "pytorch":
            return get_engine(model_name, "pytorch")
        raise RuntimeError(f"Cannot initialize model {model_name}: {err}") from err


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for warm-up and graceful shutdown."""
    logger.info("Starting NXP Cup Vision Steering API...")
    try:
        engine = get_engine(DEFAULT_MODEL, DEFAULT_BACKEND)
        # Warm-up inference
        dummy = np.zeros((66, 200, 3), dtype=np.uint8)
        engine.infer(dummy)
        logger.info("Engine warmed up successfully: %s (%s)", DEFAULT_MODEL, DEFAULT_BACKEND)
    except Exception as err:
        logger.error("Warm-up failed: %s", err)
    yield
    logger.info("Shutting down NXP Cup Vision Steering API. Clearing engine cache.")
    ENGINE_CACHE.clear()


app = FastAPI(
    title="NXP Cup Vision Steering API",
    description=(
        "Production-grade Edge AI REST API for Real-Time Autonomous Scale Vehicle Steering. "
        "Built with FastAPI, ONNX Runtime, and PyTorch."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class SteerResponse(BaseModel):
    status: str = Field("success", description="Response status")
    model: str = Field(..., description="Active steering architecture name")
    backend: str = Field(..., description="Execution provider used (onnx / pytorch)")
    steer_deg: float = Field(..., description="Recommended steering angle in degrees [-30.0, +30.0]")
    steer_norm: float = Field(..., description="Normalized steering command [-1.0, +1.0]")
    raw_steer_deg: float | None = Field(None, description="Unfiltered steering angle if smoother is active")
    inference_time_ms: float = Field(..., description="Latency of execution in milliseconds")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of inference")


class Base64SteerRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded RGB/BGR image string (JPEG or PNG)")
    model_name: str | None = Field(None, description="Optional override of steering architecture")
    backend: str | None = Field(None, description="Optional override of execution backend")


class HealthResponse(BaseModel):
    status: str
    active_model: str
    active_backend: str
    available_models: list[str]
    supported_backends: list[str]
    version: str
    uptime_seconds: float


class ModelSpec(BaseModel):
    name: str
    paradigm: str
    parameters: int
    size_fp32: str
    p50_latency_ms: float
    throughput_fps: float
    nominal_mae: float
    drift_mae: float
    resilience_tag: str


class ModelsResponse(BaseModel):
    default_model: str
    default_backend: str
    models: list[ModelSpec]


# --- Helper Functions ---
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw byte array into RGB uint8 numpy image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from provided bytes.")
    # Convert BGR -> RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --- Endpoints ---
@app.get("/", summary="Root index")
async def root() -> dict[str, Any]:
    return {
        "title": "NXP Cup Vision Steering API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "models_url": "/models",
        "steer_url": "/steer",
    }


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        active_model=DEFAULT_MODEL,
        active_backend=DEFAULT_BACKEND,
        available_models=AVAILABLE_MODELS,
        supported_backends=["onnx", "pytorch"],
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


@app.get("/models", response_model=ModelsResponse, summary="List available neural steering models")
async def list_models() -> ModelsResponse:
    specs = [
        ModelSpec(
            name="micro_vit_smoothed",
            paradigm="Hybrid Vision Transformer + Kinematic Median-EMA Filter",
            parameters=359841,
            size_fp32="1.39 MB",
            p50_latency_ms=2.18,
            throughput_fps=437.0,
            nominal_mae=4.91,
            drift_mae=11.85,
            resilience_tag="Best Stability (+8.2% drift, -75% jitter)",
        ),
        ModelSpec(
            name="micro_vit",
            paradigm="Hybrid Vision Transformer (Stem + MHSA)",
            parameters=359841,
            size_fp32="1.39 MB",
            p50_latency_ms=2.70,
            throughput_fps=373.8,
            nominal_mae=5.66,
            drift_mae=11.56,
            resilience_tag="Resilient (+21.1% drift)",
        ),
        ModelSpec(
            name="palhano_cnn",
            paradigm="Pruned Deep ConvNet (ENSICAEN baseline)",
            parameters=886037,
            size_fp32="3.38 MB",
            p50_latency_ms=0.71,
            throughput_fps=1324.9,
            nominal_mae=7.21,
            drift_mae=11.51,
            resilience_tag="Sensitive (+59.5% drift)",
        ),
        ModelSpec(
            name="dave2",
            paradigm="Deep ConvNet (NVIDIA baseline)",
            parameters=1595511,
            size_fp32="6.09 MB",
            p50_latency_ms=1.25,
            throughput_fps=813.8,
            nominal_mae=6.63,
            drift_mae=11.73,
            resilience_tag="Catastrophic Fail (+76.9% drift)",
        ),
    ]
    return ModelsResponse(
        default_model=DEFAULT_MODEL,
        default_backend=DEFAULT_BACKEND,
        models=specs,
    )


@app.post("/reset", summary="Reset temporal filter buffers")
async def reset_smoother(
    model_name: Annotated[str, Query(description="Model whose smoother state to reset")] = DEFAULT_MODEL,
) -> dict[str, str]:
    """Resets online median and EMA buffers between laps or after disengagements."""
    for (m, _), eng in ENGINE_CACHE.items():
        if m == model_name:
            eng.reset()
    return {"status": "success", "message": f"Temporal state reset for model '{model_name}'"}


@app.post("/steer", response_model=SteerResponse, summary="Infer steering angle from uploaded image file")
async def predict_steer_multipart(
    file: Annotated[UploadFile, File(description="Road camera image (JPEG or PNG)")],
    model_name: Annotated[str, Query(description="Target steering architecture")] = DEFAULT_MODEL,
    backend: Annotated[str, Query(description="Execution backend (onnx or pytorch)")] = DEFAULT_BACKEND,
) -> SteerResponse:
    """Real-time steering prediction from multipart image file."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        img_rgb = decode_image_bytes(content)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image content: {err}",
        ) from err

    try:
        engine = get_engine(model_name, backend)
        result = engine.infer(img_rgb)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as infer_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {infer_err}",
        ) from infer_err

    return SteerResponse(
        status="success",
        model=model_name,
        backend=backend,
        steer_deg=result["angle_deg"],
        steer_norm=result["angle_norm"],
        raw_steer_deg=result.get("raw_angle_deg"),
        inference_time_ms=result["latency_ms"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/steer/base64", response_model=SteerResponse, summary="Infer steering angle from Base64 image payload")
async def predict_steer_base64(request: Base64SteerRequest) -> SteerResponse:
    """Real-time steering prediction from Base64 encoded image string."""
    model_name = request.model_name or DEFAULT_MODEL
    backend = request.backend or DEFAULT_BACKEND

    try:
        raw_str = request.image_base64
        # Strip header if present (e.g. data:image/jpeg;base64,...)
        if "," in raw_str:
            raw_str = raw_str.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_str)
        img_rgb = decode_image_bytes(img_bytes)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode base64 image: {err}",
        ) from err

    try:
        engine = get_engine(model_name, backend)
        result = engine.infer(img_rgb)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as infer_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {infer_err}",
        ) from infer_err

    return SteerResponse(
        status="success",
        model=model_name,
        backend=backend,
        steer_deg=result["angle_deg"],
        steer_norm=result["angle_norm"],
        raw_steer_deg=result.get("raw_angle_deg"),
        inference_time_ms=result["latency_ms"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main.src.api.app:app", host="0.0.0.0", port=8000, reload=True)
