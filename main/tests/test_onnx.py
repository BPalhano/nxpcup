"""Unit tests for ONNX export and numerical parity with PyTorch."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from main.src.models import ModelFactory
from main.src.optimization.onnx_exporter import export_model_to_onnx


@pytest.mark.parametrize("model_name", ["dave2", "palhano_cnn", "micro_vit"])
def test_onnx_export_and_validity(model_name: str) -> None:
    """Verifies that each model exports to valid ONNX and satisfies onnx.checker."""
    model = ModelFactory.create(model_name)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / f"{model_name}.onnx"
        meta = export_model_to_onnx(
            model=model,
            output_path=onnx_file,
            simplify=True,
            verify_parity=True,
        )

        assert onnx_file.exists()
        assert meta["parity_verified"] is True
        assert meta["max_difference"] < 1e-4

        # Validate with official ONNX checker
        proto = onnx.load(str(onnx_file))
        onnx.checker.check_model(proto)


@pytest.mark.parametrize("model_name", ["dave2", "palhano_cnn", "micro_vit"])
def test_onnx_dynamic_batching(model_name: str) -> None:
    """Verifies that the exported ONNX model runs with dynamic batch sizes (1, 4, 16)."""
    model = ModelFactory.create(model_name)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / f"{model_name}_dyn.onnx"
        export_model_to_onnx(
            model=model,
            output_path=onnx_file,
            simplify=True,
            verify_parity=False,
        )

        session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

        for b_size in [1, 4, 16]:
            inp = np.random.randn(b_size, 3, 66, 200).astype(np.float32)
            out = session.run(None, {"input": inp})[0]
            assert out.shape == (b_size, 1)
            assert not np.isnan(out).any()
