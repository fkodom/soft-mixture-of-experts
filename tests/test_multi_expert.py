import pytest
import torch

from soft_mixture_of_experts.multi_expert import MultiExpertLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("in_features", [8])
@pytest.mark.parametrize("out_features", [4])
@pytest.mark.parametrize("num_experts", [1, 4])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16])
def test_multi_expert_layer(
    in_features: int,
    out_features: int,
    num_experts: int,
    bias: bool,
    batch_size: int,
    seq_len: int,
):
    experts = MultiExpertLayer(
        in_features=in_features,
        out_features=out_features,
        num_experts=num_experts,
        bias=bias,
        device=DEVICE,
        dtype=DTYPE,
    )

    test_shapes = [
        (batch_size, num_experts, in_features),
        (batch_size, num_experts, seq_len, in_features),
    ]
    for shape in test_shapes:
        x = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Check that forward pass works
        y = experts.forward(x)
        assert y.size(0) == x.size(0)
        assert y.size(1) == x.size(1)
        assert y.size(-1) == out_features
        if y.ndim == 4:
            assert y.size(2) == x.size(2)
        assert y.requires_grad

        # Check that gradients are propagated
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.requires_grad is False


def test_multi_expert_layer_exceptions():
    experts = MultiExpertLayer(
        in_features=8, out_features=4, num_experts=4, device=DEVICE, dtype=DTYPE
    )

    # Test wrong input dimension
    x = torch.randn((1, 4, 16), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)

    # Test wrong number of experts
    x = torch.randn((1, 8, 8), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)

    # Test invalid number of dimensions
    x = torch.randn((1, 4, 2, 2, 8), device=DEVICE, dtype=DTYPE)
    with pytest.raises(ValueError):
        experts.forward(x)
