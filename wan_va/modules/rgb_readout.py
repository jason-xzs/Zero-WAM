"""Matched static and observation-conditioned readouts of human RGB memory.

The attention/FFN building blocks come from OpenFlamingo (see the bundled
NOTICE). The controlled query conditioning and the split global/local readout
are repository-specific experiments inspired by task-conditioned aggregation.
This module does not read future robot frames, actions, or teacher targets.
The caller is responsible for passing only currently available state tokens.
"""

import math

import torch
from torch import nn

from .third_party.open_flamingo import FeedForward, PerceiverAttention


class DemoReadout(nn.Module):
    """Compress ordered human tokens with a fixed output/parameter budget.

    At the same configured token budget, all modes instantiate the same
    parameter shapes. ``static`` masks
    the state contribution for every query; ``state`` uses it for every query;
    ``global_local`` masks it for the first ``global_tokens`` queries. Since the
    upstream attention includes latent self-K/V, the latter mode processes the
    two query groups separately so state cannot leak into the global group.

    ``state_features`` are pre-transformer observation patch embeddings. Spatial
    coordinates belong to those patches, not to the human memory. Human time
    information remains in its supplied ordered tokens. Nothing is cached here:
    an updated observation must produce a new call to this module.
    """

    MODES = {"static", "state", "global_local"}

    def __init__(
        self,
        *,
        hidden_dim: int,
        state_dim: int,
        num_heads: int,
        num_tokens: int = 32,
        global_tokens: int = 8,
        mode: str = "static",
        num_layers: int = 1,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim), ("state_dim", state_dim),
            ("num_heads", num_heads), ("num_tokens", num_tokens),
            ("num_layers", num_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if mode not in self.MODES:
            raise ValueError(f"Unsupported readout mode: {mode!r}")
        if isinstance(global_tokens, bool) or not isinstance(global_tokens, int):
            raise ValueError("global_tokens must be an integer")
        if not 0 <= global_tokens <= num_tokens:
            raise ValueError("global_tokens must be in 0..num_tokens")
        if mode == "global_local" and not 0 < global_tokens < num_tokens:
            raise ValueError("global_local needs both global and local tokens")
        if not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_tokens = num_tokens
        self.global_tokens = global_tokens
        self.queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.state_norm = nn.LayerNorm(state_dim, eps=norm_eps)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.state_attention = PerceiverAttention(
            dim=hidden_dim, dim_head=hidden_dim // num_heads, heads=num_heads
        )
        self.state_ff = FeedForward(hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(
                    dim=hidden_dim, dim_head=hidden_dim // num_heads,
                    heads=num_heads,
                ),
                FeedForward(hidden_dim),
            ])
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)

    def _validate_features(self, features, name, dim):
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError(f"{name} must have shape [B,N,{dim}]")
        if features.shape[-1] != dim or any(size < 1 for size in features.shape):
            raise ValueError(f"{name} has invalid shape {tuple(features.shape)}")
        if not features.is_floating_point():
            raise TypeError(f"{name} must contain floating-point features")
        if features.device != self.queries.device:
            raise ValueError(f"{name} must be on the readout device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} contains non-finite features")

    def _state_positions(self, features, positions):
        """Fixed sinusoidal row/column encoding; no resolution-specific weights."""
        batch, patches, _ = features.shape
        if positions is None:
            # This fallback has an explicit raster ordering; production callers
            # supply the actual row/column grid, including camera layout.
            positions = torch.stack([
                torch.zeros(patches, device=features.device),
                torch.arange(patches, device=features.device),
            ], dim=-1)
        if not isinstance(positions, torch.Tensor):
            raise TypeError("state_positions must be a tensor")
        if positions.ndim == 2:
            positions = positions.unsqueeze(0)
        if (
            positions.ndim != 3 or positions.shape[1:] != (patches, 2)
            or positions.shape[0] not in (1, batch)
        ):
            raise ValueError("state_positions must have shape [P,2] or [B,P,2]")
        if positions.device != features.device:
            raise ValueError("state_positions must be on the readout device")
        if positions.dtype == torch.bool or positions.is_complex():
            raise TypeError("state_positions must contain real coordinates")
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("state_positions contains non-finite coordinates")
        frequencies = torch.exp(
            torch.arange(0, self.hidden_dim, 4, device=features.device,
                         dtype=torch.float32)
            * (-math.log(10000.0) / self.hidden_dim)
        )
        angles = positions.float().unsqueeze(-1) * frequencies
        row, column = angles.unbind(dim=-2)
        encoding = torch.stack(
            [row.sin(), row.cos(), column.sin(), column.cos()], dim=-1
        ).flatten(-2)[..., :self.hidden_dim]
        return encoding.to(dtype=features.dtype)

    def _read_human(self, human, queries):
        for attention, ff in self.layers:
            queries = queries + attention(human, queries)
            queries = queries + ff(queries)
        return self.output_norm(queries)

    def forward(self, human_tokens, state_features, state_positions=None):
        self._validate_features(human_tokens, "human_tokens", self.hidden_dim)
        self._validate_features(state_features, "state_features", self.state_dim)
        if human_tokens.shape[0] != state_features.shape[0]:
            raise ValueError("human_tokens and state_features batch mismatch")
        dtype = self.queries.dtype
        human = human_tokens.to(dtype=dtype).unsqueeze(1)
        state = self.state_projection(
            self.state_norm(state_features.to(dtype=self.state_norm.weight.dtype))
        )
        state = state + self._state_positions(state, state_positions)
        state = state.unsqueeze(1)
        queries = self.queries[None, None].expand(human.shape[0], 1, -1, -1)
        state_delta = self.state_attention(state, queries)
        state_delta = state_delta + self.state_ff(state_delta)

        if self.mode == "static":
            # Keep the same computation/parameter graph for the matched control.
            # This also gives FSDP a well-defined zero gradient on these tensors.
            result = self._read_human(human, queries + state_delta * 0.0)
        elif self.mode == "state":
            result = self._read_human(human, queries + state_delta)
        else:
            cut = self.global_tokens
            global_queries = queries[:, :, :cut] + state_delta[:, :, :cut] * 0.0
            local_queries = queries[:, :, cut:] + state_delta[:, :, cut:]
            result = torch.cat([
                self._read_human(human, global_queries),
                self._read_human(human, local_queries),
            ], dim=-2)
        return result.squeeze(1)


__all__ = ["DemoReadout"]
