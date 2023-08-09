import pytest
import torch

from soft_mixture_of_experts.transformer import (
    SoftMoEDecoder,
    SoftMoEDecoderLayer,
    SoftMoEEncoder,
    SoftMoEEncoderLayer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("num_experts", [1, 2])
@pytest.mark.parametrize("slots_per_expert", [1, 2])
@pytest.mark.parametrize("d_model", [8])
@pytest.mark.parametrize("nhead", [1, 2])
@pytest.mark.parametrize("dim_feedforward", [16])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("batch_size", [2])
def test_soft_moe_encoder(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    dim_feedforward: int,
    nhead: int,
    d_model: int,
    slots_per_expert: int,
    num_experts: int,
    norm_first: bool,
):
    layer = SoftMoEEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        norm_first=norm_first,
        device=DEVICE,
        dtype=DTYPE,
    )
    encoder = SoftMoEEncoder(layer, num_layers=num_layers)
    x = torch.randn(
        (batch_size, seq_len, d_model),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = encoder.forward(x)
    assert y.shape == x.shape

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("num_experts", [1, 4])
@pytest.mark.parametrize("slots_per_expert", [1, 2])
@pytest.mark.parametrize("d_model", [8])
@pytest.mark.parametrize("nhead", [1, 4])
@pytest.mark.parametrize("dim_feedforward", [16])
@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_soft_moe_decoder(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    dim_feedforward: int,
    nhead: int,
    d_model: int,
    slots_per_expert: int,
    num_experts: int,
    norm_first: bool,
):
    layer = SoftMoEDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        norm_first=norm_first,
        device=DEVICE,
        dtype=DTYPE,
    )
    decoder = SoftMoEDecoder(layer, num_layers=num_layers)
    x = torch.randn(
        (batch_size, seq_len, d_model),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )
    mem = torch.randn(
        (batch_size, seq_len, d_model),
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    # Check that forward pass works
    y = decoder.forward(x, mem)
    assert y.shape == x.shape

    # Check that gradients are propagated
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.requires_grad is False
    assert mem.grad is not None
    assert mem.grad.shape == mem.shape
    assert mem.grad.requires_grad is False
