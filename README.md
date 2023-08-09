# soft-mixture-of-experts

> work in progress

PyTorch implementation of Soft MoE by Google Brain in [From Sparse to Soft Mixtures of Experts](https://arxiv.org/abs/2308.00951.pdf)

<img src="doc/soft-moe-layer.jpeg" alt="soft-moe-layer" width="600"/>

### TODO

- [x] Implement Soft MoE layer
- [x] Set up unit tests
- [ ] Example end-to-end Transformer models
    - [ ] language model
    - [ ] vision transformer
- [ ] Reproduce benchmarks from paper
    - [ ] ViT inference time (Tables 1, 2)
    - [ ] ViT training step time??? (Figure 7)
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
# Clone/fork this repo
gh repo clone fkodom/soft-mixture-of-experts
cd soft-mixture-of-experts
# Install all dev dependencies (tests etc.) in editable mode
pip install -e .[test]
# Setup pre-commit hooks
pre-commit install
```


## Usage

### `SoftMoE`

As a standalone module:

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