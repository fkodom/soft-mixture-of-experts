from __future__ import annotations

import math
from typing import Optional, Union

import torch
from einops import einsum, rearrange
from torch import Tensor, nn


class MultiExpertLayer(nn.Module):
    """A more efficient alternative to creating 'n' separate expert layers (likely
    from 'nn.Linear' modules).  Instead, we create a single set of batched weights
    and biases, and apply all 'experts' in parallel.

    Args:
        embed_dim (int): embedding dimension (d)
        num_experts (int): number of experts (n)
        bias (bool): whether to include a bias term. Default: True
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.empty((num_experts, embed_dim, embed_dim), device=device, dtype=dtype)
        )
        bias_param: Optional[nn.Parameter] = None
        if bias:
            bias_param = nn.Parameter(
                torch.empty((num_experts, embed_dim), device=device, dtype=dtype)
            )
        # Include type annotation for mypy :D
        self.bias: Optional[nn.Parameter]
        self.register_parameter("bias", bias_param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # NOTE: Mostly copy-pasta from 'nn.Linear.reset_parameters'
        #
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected input with last dimension {self.embed_dim}, but "
                f"found {x.size(-1)}"
            )

        # NOTE: 'd1' and 'd2' are both equal to 'embed_dim'. But for 'einsum' to
        # work correctly, we have to give them different names.
        x = einsum(x, self.weight, "b n ... d1, n d1 d2 -> b n ... d2")

        if self.bias is not None:
            # NOTE: When used with 'SoftMoE' the inputs to 'MultiExpertLayer' will
            # always be 4-dimensional.  But it's easy enough to generalize for 3D
            # inputs as well, so I decided to include that here.
            if x.ndim == 3:
                bias = rearrange(self.bias, "n d -> () n d")
            elif x.ndim == 4:
                bias = rearrange(self.bias, "n d -> () n () d")
            else:
                raise ValueError(
                    f"Expected input to have 3 or 4 dimensions, but got {x.ndim}"
                )
            x = x + bias

        return x

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_experts={self.num_experts}, "
            f"bias={self.bias is not None}"
        )


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

        # TODO: Check for initialization strategy from the paper
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

        logits = einsum(x, self.phi, "b m d, d n p -> b m n p")
        dispatch_weights = logits.softmax(dim=0)  # denoted 'D' in the paper
        # NOTE: The 'torch.softmax' function does not support multiple values for the
        # 'dim' argument (unlike jax), so we are forced to flatten the last two dimensions.
        # Then, we rearrange the Tensor into its original shape.
        combine_weights = rearrange(
            logits.flatten(start_dim=2).softmax(dim=-1),
            "b m (n p) -> b m n p",
            n=num_experts,
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


if __name__ == "__main__":
    # testing/debugging
    # TODO: convert to unit tests

    batch_size = 2
    seq_len = 16
    embed_dim = 2048
    num_experts = 32
    slots_per_expert = 2

    device = "cuda"
    dtype = torch.float16

    x = torch.randn((batch_size, seq_len, embed_dim), device=device, dtype=dtype)
    moe = SoftMoE(
        embed_dim=embed_dim,
        num_experts=num_experts,
        slots_per_expert=slots_per_expert,
        device=device,
        dtype=dtype,
    )
    linear = nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)
    y = moe.forward(x)
    print(x.shape, y.shape)
