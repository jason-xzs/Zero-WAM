# OpenFlamingo Perceiver building blocks

Source: https://github.com/mlfoundations/open_flamingo/blob/655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py

Commit: `655f693fbfa04cd6e9a987d960654624d48917cf`.

Upstream helpers.py SHA256: `6ba531354ef0e2ad2970f8a2b228c4d97645892eef8c10dc6cbc2c9761adf58e`.

The `FeedForward` and `PerceiverAttention` definitions are extracted from the
upstream source. We remove the `einops_exts.rearrange_many` dependency by using
one `einops.rearrange` call per tensor, with the identical pattern. The attention
math, layer normalization, latent self-K/V, residual feed-forward building
block, and initialization remain as upstream. The MIT license is included.
Upstream itself credits https://github.com/lucidrains/flamingo-pytorch.

The state-conditioned query construction, fixed two-dimensional state positions,
static/state/global-local controls, and Zero-WAM integration in `rgb_readout.py`
are new work in this repository. This is not a reproduction of Flamingo or
SemanticVLA, and does not use their pretrained model weights.
