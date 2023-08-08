from __future__ import annotations

import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn

from soft_mixture_of_experts.multi_expert import MultiExpertLayer


class SoftMoE(nn.Module):
    """A PyTorch module for Soft-MoE, as described in the paper:
        "From Sparse to Soft Mixtures of Experts"
        https://arxiv.org/pdf/2308.00951.pdf

    einstein notation:
    - b: batch size
    - m: input sequence length
    - d: embedding dimension
    - n: num experts
    - p: num slots per expert
    - (n * p): total number of slots

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        slots_per_expert (int): number of slots per expert (p)
        bias (bool): whether to include a bias term. Default: True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        slots_per_expert: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.bias = bias

        self.phi = nn.Parameter(
            torch.empty(
                (embed_dim, num_experts, slots_per_expert),
                device=device,
                dtype=dtype,
            )
        )
        self.experts = MultiExpertLayer(
            embed_dim=embed_dim,
            num_experts=num_experts,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Copy weight initialization from 'nn.Linear.reset_parameters'
        # TODO: Check for initialization strategy from the paper
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the Soft-MoE layer, as described in:
            https://arxiv.org/pdf/2308.00951.pdf
        See: equations (1-3), algorithm 1, and figure 2

        einstein notation:
        - b: batch size
        - m: input sequence length
        - d: embedding dimension
        - n: num experts
        - p: num slots per expert
        - (n * p): total number of slots

        Args:
            x (Tensor): input tensor of shape (b, m, d)

        Returns:
            Tensor: output tensor of shape (b, m, d)
        """
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected x.size(-1)={x.size(-1)} to match embed_dim={self.embed_dim}, "
                f"but got {x.size(-1)}."
            )
        elif x.ndim != 3:
            raise ValueError(f"Expected input to have 3 dimensions, but got {x.ndim}.")

        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = logits.softmax(dim=0)  # denoted 'D' in the paper
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=self.num_experts,
        )

        # NOTE: To save memory, I don't rename the intermediate tensors Y, Ys, Xs.
        # Instead, I just overwrite the 'x' variable.  The names from the paper are
        # included in a comment for each line below.
        x = einsum(x, dispatch_weights, "b m d, b m n p -> b n p d")  # Xs
        x = self.experts(x)  # Ys
        x = einsum(x, combine_weights, "b n p d, b m n p -> b m d")  # Y

        return x

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, "
            f"slots_per_expert={self.slots_per_expert}, bias={self.bias}"
        )
