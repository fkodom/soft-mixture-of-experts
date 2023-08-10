# soft-mixture-of-experts

PyTorch implementation of Soft MoE by Google Brain in [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951.pdf)

<img src="doc/soft-moe-layer.jpeg" alt="soft-moe-layer" width="600"/>

> Thanks to [lucidrains](github.com/lucidrains) for his excellent `x-transformers` library! 🎉
> 
> The ViT implementations here are *heavily* based on his [ViTransformerWrapper](https://github.com/lucidrains/x-transformers/blob/7395ebd9bfaea85ef8358e1d46ca176351058017/x_transformers/x_transformers.py#L1215).

### TODO

- [x] Implement Soft MoE layer ([Usage](#softmoe), [Code](./soft_mixture_of_experts/soft_moe.py))
- [x] Example end-to-end Transformer models
    - [x] vision transformer ([Usage](#vision-transformers), [Code](./soft_mixture_of_experts/vit.py))
    - [ ] ~~language model~~ (skip for now)
    - [x] add to README
- [x] Set up unit tests
    - [x] SoftMoE
    - [x] Transformer layers
    - [x] ViT models
- [ ] Reproduce parameter counts from Table 3
- [ ] Reproduce inference benchmarks from Tables 1, 2
- [ ] Release on PyPI
    - [ ] Prerelease
    - [ ] Stable


## Install

PyPI:
> work in progress

From source:
```bash
pip install "soft-mixture-of-experts @ git+ssh://git@github.com/fkodom/soft-mixture-of-experts.git"
```

For contributors:
```bash
# Clone/fork this repo. Example:
gh repo clone fkodom/soft-mixture-of-experts
cd soft-mixture-of-experts
# Install all dev dependencies (tests etc.) in editable mode
pip install -e .[test]
# Setup pre-commit hooks
pre-commit install
```


## Usage

### Vision Transformers

Using the `ViT` and `SoftMoEViT` classes directly:

```python
from soft_mixture_of_experts.vit import ViT, SoftMoEViT

vit = ViT(num_classes=1000, device="cuda")
moe_vit = SoftMoEViT(num_classes=1000, num_experts=32, device="cuda")

# image shape: (batch_size, channels, height, width)
image = torch.randn(1, 3, 224, 224, device="cuda")

# classification prediction
# output shape: (batch_size, num_classes)
y_vit = vit(image)
y_moe = moe_vit(image)

# feature embeddings
# output shape: (batch_size, num_patches, d_model)
features_vit = vit(image, return_features=True)
features_moe = moe_vit(image, return_features=True)
```

or using pre-configured models:
```python
from soft_mixture_of_experts.vit import soft_moe_vit_small

# Available models:
# - soft_moe_vit_small
# - soft_moe_vit_base
# - soft_moe_vit_large
# - vit_small
# - vit_base
# - vit_large
# - vit_huge

# Roughly 930M parameters 👀
moe_vit = soft_moe_vit_small(num_classes=1000, device="cuda")

# Everything else works the same as above...
```


### Transformer Layers

```python
from soft_mixture_of_experts.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

encoder = TransformerEncoder(
    TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6,
)
decoder = TransformerDecoder(
    TransformerDecoderLayer(d_model=512, nhead=8),
    num_layers=6,
)

# input shape: (batch_size, seq_len, d_model)
x = torch.randn(2, 128, 512, device="cuda")

mem = encoder(x)
print(mem.shape)
# torch.Size([2, 128, 512])

y = decoder(x, mem)
print(y.shape)
# torch.Size([2, 128, 512])
```


### Soft MoE

```python
import torch

from soft_mixture_of_experts.soft_moe import SoftMoE

# SoftMoE with 32 experts, 2 slots per expert (64 total):
moe = SoftMoE(
    embed_dim=512,
    num_experts=32,
    slots_per_expert=2,
    bias=False,  # optional, default: True
    device="cuda",  # optional, default: None
)

# input shape: (batch_size, seq_len, embed_dim)
x = torch.randn(2, 128, 512, device="cuda")

y = moe(x)
print(y.shape)
# torch.Size([2, 128, 512])
```


## Test

Tests run automatically through GitHub Actions on each `git push`.

You can also run tests manually with `pytest`:
```bash
pytest
```


## Citations

```bibtex
@misc{puigcerver2023sparse,
      title={From Sparse to Soft Mixtures of Experts}, 
      author={Joan Puigcerver and Carlos Riquelme and Basil Mustafa and Neil Houlsby},
      year={2023},
      eprint={2308.00951},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
