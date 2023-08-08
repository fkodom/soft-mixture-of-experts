import pytest
import torch

from soft_mixture_of_experts.soft_moe import SoftMoE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("embed_dim", [8])
@pytest.mark.parametrize("num_experts", [1, 4])
@pytest.mark.parametrize("slots_per_expert", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16])
def test_soft_moe(
    embed_dim: int,
    num_experts: int,
    slots_per_expert: int,
    bias: bool,
    batch_size: int,
    seq_len: int,
):
    experts = SoftMoE(
        embed_dim=embed_dim,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        bias=bias,
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.randn(
        (batch_size, seq_len, embed_dim),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = experts.forward(x)
    assert y.shape == x.shape
    assert y.requires_grad

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False


def test_soft_moe_exceptions():
    experts = SoftMoE(
        embed_dim=8,
        num_experts=4,
        slots_per_expert=2,
        device=DEVICE,
        dtype=DTYPE,
    )

    # Test wrong embedding dimension
    x = torch.randn((1, 4, 16), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)

    # Test invalid number of dimensions
    x = torch.randn((1, 4, 2, 8), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)
