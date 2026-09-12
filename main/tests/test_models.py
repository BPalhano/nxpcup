"""Unit tests for NXP Cup neural network models and factory."""

import io

import pytest
import torch

from main.src.models import (
    DAVE2,
    MicroViT,
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

    with pytest.raises(ValueError, match="Unknown model name"):
        ModelFactory.create("unsupported_model_xyz")


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
