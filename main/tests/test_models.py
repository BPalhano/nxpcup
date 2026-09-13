"""Unit tests for NXP Cup neural network models and factory."""

import io

import numpy as np
import pytest
import torch

from main.src.models import (
    DAVE2,
    MedianEMASmoother,
    MicroViT,
    MicroViTSmoothed,
    ModelFactory,
    PalhanoCNN,
    count_parameters,
)


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Standard input batch (B, C, H, W) = (4, 3, 66, 200)."""
    return torch.randn(4, 3, 66, 200)


def test_dave2_architecture(dummy_input: torch.Tensor) -> None:
    """Verifies DAVE-2 forward pass and parameter count (~1.59M)."""
    model = DAVE2(dropout=0.2)
    output = model(dummy_input)

    assert output.shape == (4, 1)
    assert output.dtype == torch.float32

    counts = count_parameters(model)
    assert 1_500_000 <= counts["total"] <= 1_650_000
    assert counts["trainable"] == counts["total"]


def test_palhano_cnn_exact_parameter_count(dummy_input: torch.Tensor) -> None:
    """Verifies Palhano CNN forward pass and exact 886,037 parameter count from paper."""
    model = PalhanoCNN(dropout=0.2)
    output = model(dummy_input)

    assert output.shape == (4, 1)
    assert output.dtype == torch.float32

    counts = count_parameters(model)
    assert counts["total"] == 886_037
    assert counts["trainable"] == 886_037


def test_micro_vit_architecture(dummy_input: torch.Tensor) -> None:
    """Verifies MicroViT forward pass and parameter budget (350k - 450k)."""
    model = MicroViT(depth=2, embed_dim=128, num_heads=4)
    output = model(dummy_input)

    assert output.shape == (4, 1)
    assert output.dtype == torch.float32

    counts = count_parameters(model)
    assert 340_000 <= counts["total"] <= 450_000
    assert counts["trainable"] == counts["total"]


@pytest.mark.parametrize("model_name", ["dave2", "palhano_cnn", "micro_vit"])
def test_models_backward_gradient_flow(model_name: str, dummy_input: torch.Tensor) -> None:
    """Verifies that all parameters receive gradients during backpropagation."""
    model = ModelFactory.create(model_name)
    dummy_input = dummy_input.clone().requires_grad_(True)

    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    assert dummy_input.grad is not None

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradient"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_model_factory_dispatch() -> None:
    """Verifies that ModelFactory correctly maps model keys and raises for unknown."""
    dave = ModelFactory.create("dave2")
    assert isinstance(dave, DAVE2)

    palhano = ModelFactory.create("palhano_cnn")
    assert isinstance(palhano, PalhanoCNN)

    vit = ModelFactory.create("micro_vit")
    assert isinstance(vit, MicroViT)

    vit_smoothed = ModelFactory.create("micro_vit_smoothed")
    assert isinstance(vit_smoothed, MicroViTSmoothed)

    with pytest.raises(ValueError, match="Unknown model name"):
        ModelFactory.create("unsupported_model_xyz")


def test_median_ema_smoother_properties() -> None:
    """Verifies outlier rejection and exponential smoothing properties."""
    smoother = MedianEMASmoother(window_size=3, alpha=0.65, max_angle_deg=30.0)

    # 1. Single transient spike rejection: sequence [0, 0, 25.0, 0, 0]
    # At t=2: window is [0, 0, 25.0] -> median is 0.0 -> EMA remains 0.0!
    out0 = smoother.update(0.0)
    out1 = smoother.update(0.0)
    out2 = smoother.update(25.0)  # Transient shadow noise spike!
    out3 = smoother.update(0.0)

    assert out0 == 0.0
    assert out1 == 0.0
    assert out2 == 0.0, f"Expected 0.0 spike rejection, got {out2}"
    assert out3 == 0.0

    # 2. Smooth sequence batch processing
    smoother.reset()
    seq = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
    smoothed = smoother.smooth_sequence(seq)
    assert len(smoothed) == len(seq)
    assert np.all(smoothed <= 20.0)
    assert np.all(smoothed >= 5.0)


def test_micro_vit_smoothed_architecture(dummy_input: torch.Tensor) -> None:
    """Verifies MicroViTSmoothed parameter count, eval filtering and training mode."""
    model = MicroViTSmoothed()

    # Train mode should output raw tensor (4, 1)
    model.train()
    out_train = model(dummy_input)
    assert out_train.shape == (4, 1)

    # Eval mode single frame should apply online filter
    model.eval()
    single_inp = dummy_input[:1]
    out_eval = model(single_inp)
    assert out_eval.shape == (1, 1)
    assert -1.0 <= float(out_eval[0, 0].item()) <= 1.0

    # Counts should match MicroViT
    counts = count_parameters(model)
    assert 340_000 <= counts["total"] <= 450_000


@pytest.mark.parametrize("model_name", ["dave2", "palhano_cnn", "micro_vit"])
def test_onnx_export_compatibility(model_name: str) -> None:
    """Verifies that each model can be exported to ONNX without errors."""
    model = ModelFactory.create(model_name)
    model.eval()

    dummy_input = torch.randn(1, 3, 66, 200)
    buffer = io.BytesIO()

    torch.onnx.export(
        model,
        (dummy_input,),
        buffer,  # type: ignore[arg-type]
        input_names=["input_tensor"],
        output_names=["steering_angle"],
        dynamic_axes={"input_tensor": {0: "batch_size"}, "steering_angle": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )

    assert len(buffer.getvalue()) > 10_000

