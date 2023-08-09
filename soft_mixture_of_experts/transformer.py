from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn

from soft_mixture_of_experts.soft_moe import SoftMoE


class SoftMoEEncoderLayer(nn.Module):
    """PyTorch module for Soft-MoE Transformer Encoder Layer, as described in:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    NOTE: Nearly identical to a standard Transformer encoder layer, except that we
    replace the second feedforward layer with 'SoftMoE'.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        num_experts: int = 128,
        slots_per_expert: int = 1,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.norm_first = norm_first
        self.activation = activation

        self.dropout = nn.Dropout(dropout)

        # self-attention block
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

        # feedforward / soft-moe block
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.linear = nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.moe = SoftMoE(
            in_features=dim_feedforward,
            out_features=d_model,
            num_experts=num_experts,
            slots_per_expert=slots_per_expert,
            device=device,
            dtype=dtype,
        )

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass for the self-attention block.  Identical to the self-attention
        block in a normal Transformer (i.e. in 'nn.TransformerEncoderLayer')
        """
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout(x)

    # feedforward / soft-moe block
    def _ff_block(self, x: Tensor) -> Tensor:
        """Forward pass for the FeedForward block, which now includes a SoftMoE layer.
        Mostly copy-pasta from 'nn.TransformerEncoderLayer'.  The only difference
        is swapping 'self.linear2' for 'self.moe'.
        """
        x = self.moe(self.dropout(self.activation(self.linear(x))))
        return self.dropout(x)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x


if __name__ == "__main__":
    # TODO: Convert to unit tests

    batch_size = 2
    seq_len = 32
    d_model = 512
    nhead = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    layer = SoftMoEEncoderLayer(
        d_model=d_model, nhead=nhead, device=device, dtype=dtype
    )
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    y = layer(x)
    print(y.shape)
