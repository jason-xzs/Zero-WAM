"""RGB transition conditions using WALA's original, depth-disabled tokenizer.

Cached visual features are frozen inputs. All auxiliary predictions and losses
are computed inside this module's forward, while policy tokens depend only on
the demonstrated human video. Call ``forward`` and ``fuse`` from the owning Wan
model's forward; under FSDP the conditioner belongs to that root's parameter
group and must not be separately sharded around only its ``forward`` method.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .third_party.wala import ContinuousTransitionBottleneckTokenizer


class RGBTransitionConditioner(nn.Module):
    """Encode ordered human clips and train RGB-only transition objectives.

    ``human_features`` is [B,S,T,P,D], with the first frame of each clip acting
    as its starting state. ``robot_features`` is [B,T_r,P_r,D], with frame zero
    the current robot observation. Robot futures are used only in auxiliary
    training, never in the returned conditioning tokens or ``fuse``.

    Defaults: D=768, H=256, K=4, two encoder/decoder layers and four heads.
    ``feature_normalization='layer_norm'`` uses non-affine, per-patch feature
    normalization in FP32 (eps=1e-6), independently for every frame. No temporal
    or batch statistics are used. ``'none'`` preserves the original WALA input
    scale. Both encoder inputs and detached delta targets use the same rule.

    Losses are WALA's mean L1 and cosine feature-delta losses. ``l1_weight`` and
    ``cosine_weight`` combine each pair; ``human_self_weight``,
    ``robot_self_weight`` and ``transfer_weight`` combine the three objectives
    into ``losses['total']``. All six unweighted components are also returned.
    """

    def __init__(self, config: dict, video_dim: int = 3072):
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("RGB transition config must be a dictionary")
        self.config = dict(config)
        if config.get("use_depth", False):
            raise ValueError("RGBTransitionConditioner requires use_depth=False")

        self.feature_dim = self._positive_int(config, "feature_dim", 768)
        self.hidden_dim = self._positive_int(config, "hidden_dim", 256)
        self.num_transition_tokens = self._positive_int(
            config, "num_transition_tokens", 4
        )
        self.num_heads = self._positive_int(config, "num_heads", 4)
        self.max_clips = self._positive_int(config, "max_clips", 32)
        self.max_future_steps = self._positive_int(config, "max_future_steps", 64)
        self.max_patches = self._positive_int(config, "max_patches", 1024)
        if (
            isinstance(video_dim, bool)
            or not isinstance(video_dim, int)
            or video_dim < 1
        ):
            raise ValueError("video_dim must be a positive integer")
        self.video_dim = video_dim
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.feature_normalization = config.get("feature_normalization", "layer_norm")
        if self.feature_normalization not in {"layer_norm", "none"}:
            raise ValueError("feature_normalization must be 'layer_norm' or 'none'")
        self.norm_eps = float(config.get("norm_eps", 1e-6))
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        self.l1_weight = self._weight(config, "l1_weight", 1.0)
        self.cosine_weight = self._weight(config, "cosine_weight", 0.1)
        self.loss_weights = {
            name: self._weight(config, f"{name}_weight", 1.0)
            for name in ("human_self", "robot_self", "transfer")
        }

        self.tokenizer = ContinuousTransitionBottleneckTokenizer(
            dino_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            encoder_layers=self._positive_int(config, "encoder_layers", 2),
            decoder_layers=self._positive_int(config, "decoder_layers", 2),
            num_heads=self.num_heads,
            max_future_steps=self.max_future_steps,
            max_patches=self.max_patches,
            dropout=0.0,
            smooth_l1_weight=self.l1_weight,
            cosine_weight=self.cosine_weight,
            composition_weight=0.0,
            reencode_weight=0.0,
            content_invariance_weight=0.0,
            use_depth=False,
        )
        self.tokenizer.set_gradient_checkpointing(
            bool(config.get("gradient_checkpointing", False))
        )
        self.clip_order_embedding = nn.Parameter(
            torch.randn(self.max_clips, 1, self.hidden_dim) * 0.02
        )

        # Attention runs at H, not video_dim; only the input/output projections
        # span the Wan hidden dimension. There is no new full-width transformer.
        self.hidden_norm = nn.LayerNorm(video_dim, eps=self.norm_eps)
        self.condition_norm = nn.LayerNorm(self.hidden_dim, eps=self.norm_eps)
        self.query_proj = nn.Linear(video_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, video_dim)
        # FSDP2 cannot shard scalar parameters; a one-element vector broadcasts
        # identically and still supports item() for training metrics.
        self.gate = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _positive_int(config, key, default):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        return value

    @staticmethod
    def _weight(config, key, default):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and non-negative")
        return value

    def _validate_features(self, features, name, ndim, min_frames):
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if features.ndim != ndim:
            raise ValueError(
                f"{name} must have {ndim} dimensions, got {features.shape}"
            )
        if not features.is_floating_point():
            raise TypeError(f"{name} must contain floating point features")
        if any(size < 1 for size in features.shape):
            raise ValueError(f"{name} dimensions must be nonempty")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"{name} feature dim must be {self.feature_dim}")
        if not min_frames <= features.shape[-3] <= self.max_future_steps + 1:
            raise ValueError(
                f"{name} must have {min_frames}..{self.max_future_steps + 1} frames"
            )
        if features.shape[-2] > self.max_patches:
            raise ValueError(f"{name} patch count exceeds max_patches")
        if features.device != self.tokenizer.current_input_proj.weight.device:
            raise ValueError(f"{name} must be on the conditioner's device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} contains non-finite features")

    def _prepare_features(self, features):
        # Feature-cache tensors are not trainable, including robot targets.
        features = features.detach()
        if self.feature_normalization == "layer_norm":
            features = F.layer_norm(
                features.float(), (self.feature_dim,), eps=self.norm_eps
            )
        return features.to(dtype=self.tokenizer.current_input_proj.weight.dtype)

    def _rgb_loss(self, current, future, tokens):
        prediction = self.tokenizer.decode_delta(
            current, tokens, num_future_steps=future.shape[1]
        )
        target = (future - current[:, None]).detach()
        l1 = F.l1_loss(prediction.float(), target.float())
        cosine = self.tokenizer._token_cosine_loss(prediction, target)
        return self.l1_weight * l1 + self.cosine_weight * cosine, l1, cosine

    def forward(self, human_features, robot_features=None, compute_aux=False):
        """Return human-only tokens and auxiliary losses (empty when disabled)."""
        self._validate_features(human_features, "human_features", 5, min_frames=2)
        batch, clips, frames, patches, dim = human_features.shape
        if clips > self.max_clips:
            raise ValueError("human_features clip count exceeds max_clips")
        if robot_features is not None:
            self._validate_features(
                robot_features, "robot_features", 4, min_frames=2 if compute_aux else 1
            )
            if robot_features.shape[0] != batch:
                raise ValueError("human_features and robot_features batch mismatch")

        human = self._prepare_features(human_features).reshape(
            batch * clips, frames, patches, dim
        )
        raw_tokens = self.tokenizer.encode(human[:, 0], human[:, 1:])
        ordered_tokens = raw_tokens.reshape(
            batch, clips, self.num_transition_tokens, self.hidden_dim
        ) + self.clip_order_embedding[None, :clips].to(dtype=raw_tokens.dtype)
        tokens = ordered_tokens.reshape(
            batch, clips * self.num_transition_tokens, self.hidden_dim
        )
        if not compute_aux:
            return tokens, {}

        human_loss = self._rgb_loss(human[:, 0], human[:, 1:], raw_tokens)
        zero = human_loss[0] * 0.0
        objectives = {
            "human_self": human_loss,
            "robot_self": (zero, zero, zero),
            "transfer": (zero, zero, zero),
        }
        if robot_features is not None:
            robot = self._prepare_features(robot_features)
            robot_tokens = self.tokenizer.encode(robot[:, 0], robot[:, 1:])
            objectives["robot_self"] = self._rgb_loss(
                robot[:, 0], robot[:, 1:], robot_tokens
            )
            # No robot-future token enters this path: the future is only its
            # detached prediction target, and human tokens preserve clip order.
            objectives["transfer"] = self._rgb_loss(
                robot[:, 0], robot[:, 1:], tokens
            )

        losses = {}
        for name, (combined, l1, cosine) in objectives.items():
            losses[name] = combined
            losses[f"{name}_l1"] = l1
            losses[f"{name}_cosine"] = cosine
        losses["total"] = sum(
            self.loss_weights[name] * objectives[name][0]
            for name in self.loss_weights
        )
        return tokens, losses

    def fuse(self, human_hidden, tokens):
        """Gated residual cross-attention; gate zero preserves Wan exactly."""
        for value, name, last_dim in (
            (human_hidden, "human_hidden", self.video_dim),
            (tokens, "tokens", self.hidden_dim),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(f"{name} must have shape [B,N,{last_dim}]")
            if value.shape[-1] != last_dim or any(size < 1 for size in value.shape):
                raise ValueError(f"{name} has invalid shape {value.shape}")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")
        if human_hidden.shape[0] != tokens.shape[0]:
            raise ValueError("human_hidden and tokens batch mismatch")
        if human_hidden.device != tokens.device:
            raise ValueError("human_hidden and tokens device mismatch")
        if human_hidden.device != self.query_proj.weight.device:
            raise ValueError("human_hidden must be on the conditioner's device")

        query = self.query_proj(
            self.hidden_norm(human_hidden.to(dtype=self.hidden_norm.weight.dtype))
        )
        conditions = self.condition_norm(
            tokens.to(dtype=self.condition_norm.weight.dtype)
        )
        key, value = self.key_proj(conditions), self.value_proj(conditions)
        batch, count, _ = query.shape
        head_dim = self.hidden_dim // self.num_heads
        query = query.reshape(batch, count, self.num_heads, head_dim).transpose(1, 2)
        key = key.reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.reshape(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, count, self.hidden_dim)
        residual = self.output_proj(attended).to(dtype=human_hidden.dtype)
        return human_hidden + self.gate.tanh().to(dtype=human_hidden.dtype) * residual
