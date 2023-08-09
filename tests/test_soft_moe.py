import pytest
import torch

from soft_mixture_of_experts.soft_moe import SoftMoE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("in_features", [8])
@pytest.mark.parametrize("out_features", [4])
@pytest.mark.parametrize("num_experts", [1, 4])
@pytest.mark.parametrize("slots_per_expert", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16])
def test_soft_moe(
    in_features: int,
    out_features: int,
    num_experts: int,
    slots_per_expert: int,
    bias: bool,
    batch_size: int,
    seq_len: int,
):
    experts = SoftMoE(
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        bias=bias,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(
        (batch_size, seq_len, in_features),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = experts.forward(x)
    assert y.size(0) == x.size(0)
    assert y.size(1) == x.size(1)
    assert y.size(2) == out_features

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False


def test_soft_moe_exceptions():
    experts = SoftMoE(
        in_features=8,
        out_features=4,
        num_experts=4,
        slots_per_expert=2,
        device=DEVICE,
        dtype=DTYPE,
    )

    # Test wrong input dimension
    x = torch.randn((1, 4, 16), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)

    # Test invalid number of dimensions
    x = torch.randn((1, 4, 2, 8), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)
