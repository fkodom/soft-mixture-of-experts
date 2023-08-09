from typing import Optional

import pytest
import torch

from soft_mixture_of_experts.vit import (
    _build_soft_moe_vit,
    _build_vit,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("mlp_ratio", [1.0, 2.0])
@pytest.mark.parametrize("num_encoder_layers", [1, 2])
@pytest.mark.parametrize("nhead", [1, 2])
@pytest.mark.parametrize("d_model", [8])
@pytest.mark.parametrize("patch_size", [4])
@pytest.mark.parametrize("image_size", [32])
@pytest.mark.parametrize("num_classes", [10, None])
@pytest.mark.parametrize("batch_size", [2])
def test_vit(
    batch_size: int,
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    mlp_ratio: float,
    num_channels: int,
):
    vit = _build_vit(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        mlp_ratio=mlp_ratio,
        num_channels=num_channels,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(
        (batch_size, num_channels, image_size, image_size),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = vit.forward(x)
    features = vit.forward(x, return_features=True)
    assert y.size(0) == batch_size
    assert y.ndim == 2
    if num_classes is None:
        assert y.size(1) == d_model
    else:
        assert y.size(1) == num_classes
    assert features.ndim == 3
    assert features.size(0) == batch_size
    assert features.size(1) == (image_size // patch_size) ** 2
    assert features.size(2) == d_model

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False


@pytest.mark.parametrize("num_experts", [1, 2])
@pytest.mark.parametrize("slots_per_expert", [1, 2])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize("mlp_ratio", [1.0, 2.0])
@pytest.mark.parametrize("num_encoder_layers", [1, 2])
@pytest.mark.parametrize("nhead", [1, 2])
@pytest.mark.parametrize("d_model", [8])
@pytest.mark.parametrize("patch_size", [4])
@pytest.mark.parametrize("image_size", [32])
@pytest.mark.parametrize("num_classes", [10, None])
@pytest.mark.parametrize("batch_size", [2])
def test_soft_moe_vit(
    batch_size: int,
    num_classes: Optional[int],
    image_size: int,
    patch_size: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    mlp_ratio: float,
    num_channels: int,
    slots_per_expert: int,
    num_experts: int,
):
    vit = _build_soft_moe_vit(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        mlp_ratio=mlp_ratio,
        num_channels=num_channels,
        slots_per_expert=slots_per_expert,
        num_experts=num_experts,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(
        (batch_size, num_channels, image_size, image_size),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = vit.forward(x)
    features = vit.forward(x, return_features=True)
    assert y.size(0) == batch_size
    assert y.ndim == 2
    if num_classes is None:
        assert y.size(1) == d_model
    else:
        assert y.size(1) == num_classes
    assert features.ndim == 3
    assert features.size(0) == batch_size
    assert features.size(1) == (image_size // patch_size) ** 2
    assert features.size(2) == d_model

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False
