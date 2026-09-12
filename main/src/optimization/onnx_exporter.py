"""PyTorch to ONNX Exporter with Dynamic Batching and ONNX-Simplifier.

Features:
- Exports PyTorch models (DAVE-2, PalhanoCNN, MicroViT) to ONNX format.
- Configures dynamic batch axes for flexible inference runtime batching.
- Applies onnx-simplifier (constant folding, operator fusion).
- Validates numerical parity between PyTorch FP32 and ONNX Runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_DIR = PROJECT_ROOT / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main.src.models import ModelFactory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nxpcup.onnx_exporter")


def export_model_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 66, 200),
    opset_version: int = 17,
    simplify: bool = True,
    verify_parity: bool = True,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Exports a PyTorch model to an optimized ONNX file.

    Args:
        model: PyTorch nn.Module.
        output_path: Destination path for .onnx file.
        input_shape: Input tensor shape (default: (1, 3, 66, 200)).
        opset_version: Target ONNX opset version (default: 17).
        simplify: Whether to run onnxsim optimization.
        verify_parity: Whether to verify outputs against ONNX Runtime.
        tolerance: Maximum acceptable absolute difference.

    Returns:
        Dictionary containing export metadata and parity metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    logger.info("Exporting model to ONNX: %s (opset: %d)...", output_path, opset_version)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["steering_angle"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "steering_angle": {0: "batch_size"},
        },
        dynamo=False,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported raw ONNX model (%.2f MB)", file_size_mb)

    # Apply onnx-simplifier
    is_simplified = False
    if simplify:
        try:
            import onnxsim

            logger.info("Running onnxsim graph simplification...")
            model_proto = onnx.load(str(output_path))
            simplified_model, check = onnxsim.simplify(model_proto)
            if check:
                onnx.save(simplified_model, str(output_path))
                is_simplified = True
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info("Graph simplified successfully (%.2f MB)", file_size_mb)
            else:
                logger.warning("onnxsim check returned False; keeping unsimplified graph")
        except Exception as err:
            logger.warning("onnxsim optimization skipped or failed: %s", err)

    # Parity verification with ONNX Runtime
    max_diff = 0.0
    is_parity_valid = False
    if verify_parity:
        logger.info("Verifying numerical parity against ONNX Runtime...")
        test_batch = torch.randn(4, 3, 66, 200, dtype=torch.float32)

        with torch.no_grad():
            torch_out = model(test_batch).cpu().numpy()

        ort_session = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {"input": test_batch.numpy()}
        ort_out = ort_session.run(None, ort_inputs)[0]

        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        mean_diff = float(np.mean(np.abs(torch_out - ort_out)))
        logger.info("Parity Check -> Max Diff: %.2e | Mean Diff: %.2e", max_diff, mean_diff)

        if max_diff <= tolerance:
            logger.info("✅ Numerical parity verified successfully (max_diff <= %.1e)", tolerance)
            is_parity_valid = True
        else:
            logger.warning("⚠️ Parity difference %.2e exceeds tolerance %.1e", max_diff, tolerance)

    return {
        "output_path": str(output_path),
        "size_mb": file_size_mb,
        "simplified": is_simplified,
        "opset_version": opset_version,
        "parity_verified": is_parity_valid,
        "max_difference": max_diff,
    }


def load_model_for_export(
    model_name: str,
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """Loads a model instance, optionally populating weights from checkpoint."""
    model = ModelFactory.create(model_name)
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            logger.info("Loading weights from checkpoint: %s", ckpt_path)
            data = torch.load(ckpt_path, map_location="cpu")
            state_dict = data.get("state_dict", data)
            model.load_state_dict(state_dict)
        else:
            logger.warning("Checkpoint path %s not found. Using initialized weights.", ckpt_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NXP Cup models to optimized ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="micro_vit",
        choices=["dave2", "palhano_cnn", "micro_vit", "all"],
        help="Architecture to export",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MAIN_DIR / "outputs" / "onnx"),
        help="Directory to save exported .onnx files",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--no-simplify", action="store_true", help="Disable onnxsim")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_export = (
        ["micro_vit", "palhano_cnn", "dave2"] if args.model == "all" else [args.model]
    )

    for m_name in models_to_export:
        logger.info("==================================================")
        logger.info("Exporting: %s", m_name)
        logger.info("==================================================")

        # Look for checkpoint in training_results if not specified
        ckpt = args.checkpoint
        if ckpt is None:
            candidate = MAIN_DIR / "training_results" / m_name / "best_model.pt"
            if candidate.exists():
                ckpt = candidate

        model = load_model_for_export(m_name, ckpt)
        target_path = out_dir / f"{m_name}.onnx"

        res = export_model_to_onnx(
            model=model,
            output_path=target_path,
            opset_version=args.opset,
            simplify=not args.no_simplify,
        )
        model_subfolder = out_dir / m_name
        if model_subfolder.exists():
            import shutil

            shutil.copy2(target_path, model_subfolder / f"{m_name}.onnx")
        logger.info("Export summary: %s", res)


if __name__ == "__main__":
    main()
