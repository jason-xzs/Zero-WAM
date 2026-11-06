# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Checkpoint-compatible Transformer used by the Robotwin ICL policy.

The original Next-Forcing model is intentionally kept in ``model.py``.  This
module contains the additional action expert and the cache metadata required by
the released ICL checkpoints.  ICL tokens are persistent cache entries: target
video queries may attend to them, while action queries cannot attend to them
directly.
"""

import math
import logging
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from .model import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from .rgb_transition import RGBTransitionConditioner


ICL_CACHE_TYPE = 2
PREDICTION_CACHE_TYPE = 1
OBSERVATION_CACHE_TYPE = 0

_compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
_compiled_create_block_mask = torch.compile(create_block_mask)
logger = logging.getLogger(__name__)


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)
    )
    return torch.view_as_real(x_complex * freqs).flatten(3).to(x.dtype)


class ICLAttentionBackend:
    """Holds the per-forward masks shared by all Transformer blocks."""

    self_mask = None
    cross_mask = None

    @classmethod
    def build_self_mask(
        cls,
        query_type_ids: torch.Tensor,
        key_type_ids: torch.Tensor,
        query_seq_ids: torch.Tensor,
        key_seq_ids: torch.Tensor,
        query_frame_ids: torch.Tensor,
        key_frame_ids: torch.Tensor,
        window_size: int,
        device: torch.device,
        compile_mask: bool = True,
    ):
        def window_mask(b, h, q_idx, kv_idx):
            return (window_size == -1) | (
                (query_frame_ids[q_idx] - key_frame_ids[kv_idx]).abs()
                <= window_size
            )

        def valid_sequence_mask(b, h, q_idx, kv_idx):
            return (key_type_ids[kv_idx] != -1) & (
                query_seq_ids[q_idx] == key_seq_ids[kv_idx]
            )

        def non_icl_key_mask(b, h, q_idx, kv_idx):
            return key_type_ids[kv_idx] != ICL_CACHE_TYPE

        def video_to_icl_mask(b, h, q_idx, kv_idx):
            return (
                (query_type_ids[q_idx] == 0)
                & (key_type_ids[kv_idx] == ICL_CACHE_TYPE)
                & (query_seq_ids[q_idx] == key_seq_ids[kv_idx])
            )

        mask_mod = or_masks(
            and_masks(valid_sequence_mask, window_mask, non_icl_key_mask),
            video_to_icl_mask,
        )

        mask_builder = _compiled_create_block_mask if compile_mask else create_block_mask
        cls.self_mask = mask_builder(
            mask_mod,
            1,
            1,
            len(query_type_ids),
            len(key_type_ids),
            device=device,
            _compile=compile_mask,
        )
        return cls.self_mask

    @classmethod
    def build_training_self_mask(
        cls,
        seq_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        noise_ids: torch.Tensor,
        type_ids: torch.Tensor,
        icl_ids: torch.Tensor,
        window_size: int,
        device: torch.device,
        compile_mask: bool = True,
    ):
        """Build the temporal-forcing mask for one Robotwin ICL sample."""

        def valid_pair(b, h, q_idx, kv_idx):
            return (seq_ids[q_idx] >= 0) & (seq_ids[q_idx] == seq_ids[kv_idx])

        def in_window(b, h, q_idx, kv_idx):
            return (window_size == -1) | (
                (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size
            )

        def base_tf_pair(b, h, q_idx, kv_idx):
            clean_to_clean = (
                (noise_ids[q_idx] == 1)
                & (noise_ids[kv_idx] == 1)
                & (frame_ids[kv_idx] <= frame_ids[q_idx])
            )
            noisy_to_clean = (
                (noise_ids[q_idx] == 0)
                & (noise_ids[kv_idx] == 1)
                & (frame_ids[kv_idx] < frame_ids[q_idx])
            )
            noisy_to_noisy = (
                (noise_ids[q_idx] == 0)
                & (noise_ids[kv_idx] == 0)
                & (frame_ids[kv_idx] == frame_ids[q_idx])
            )
            return (
                (icl_ids[q_idx] == 0)
                & (icl_ids[kv_idx] == 0)
                & (clean_to_clean | noisy_to_clean | noisy_to_noisy)
            )

        def target_video_to_icl(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] >= 0)
                & (seq_ids[q_idx] == seq_ids[kv_idx])
                & (icl_ids[q_idx] == 0)
                & (icl_ids[kv_idx] == 1)
                & (type_ids[q_idx] == 0)
            )

        def icl_to_icl(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] >= 0)
                & (seq_ids[q_idx] == seq_ids[kv_idx])
                & (icl_ids[q_idx] == 1)
                & (icl_ids[kv_idx] == 1)
            )

        def mask_mod(b, h, q_idx, kv_idx):
            base = valid_pair(b, h, q_idx, kv_idx) & in_window(
                b, h, q_idx, kv_idx
            ) & base_tf_pair(b, h, q_idx, kv_idx)
            return (
                base
                | target_video_to_icl(b, h, q_idx, kv_idx)
                | icl_to_icl(b, h, q_idx, kv_idx)
            )

        mask_builder = _compiled_create_block_mask if compile_mask else create_block_mask
        cls.self_mask = mask_builder(
            mask_mod,
            1,
            1,
            len(seq_ids),
            len(seq_ids),
            device=device,
            _compile=compile_mask,
        )
        return cls.self_mask

    @classmethod
    def build_cross_mask(
        cls,
        query_seq_ids: torch.Tensor,
        encoder_seq_ids: torch.Tensor,
        device: torch.device,
        compile_mask: bool = True,
    ):
        def sequence_mask(b, h, q_idx, kv_idx):
            return query_seq_ids[q_idx] == encoder_seq_ids[kv_idx]

        mask_builder = _compiled_create_block_mask if compile_mask else create_block_mask
        cls.cross_mask = mask_builder(
            sequence_mask,
            1,
            1,
            len(query_seq_ids),
            len(encoder_seq_ids),
            device=device,
            _compile=compile_mask,
        )
        return cls.cross_mask

    @classmethod
    def apply(
        cls, query, key, value, *, cross_attention: bool, block_mask=None
    ):
        if block_mask is None:
            block_mask = cls.cross_mask if cross_attention else cls.self_mask
        if block_mask is None:
            raise RuntimeError("Attention mask was not initialized")
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        half_dtypes = (torch.float16, torch.bfloat16)
        if query.dtype not in half_dtypes:
            query = query.to(torch.bfloat16)
        if key.dtype not in half_dtypes:
            key = key.to(torch.bfloat16)
        if value.dtype not in half_dtypes:
            value = value.to(torch.bfloat16)
        query = query.to(value.dtype)
        key = key.to(value.dtype)

        return _compiled_flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        ).transpose(1, 2)


class WanICLAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        dropout: float = 0.0,
        cross_attention_dim_head: Optional[int] = None,
        action_inner_dim: int = -1,
        attn_moe: bool = True,
    ):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )
        action_dim = action_inner_dim if action_inner_dim > 0 else dim

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)]
        )
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)

        if not attn_moe:
            raise ValueError("The released ICL checkpoints require attn_moe=True")
        self.action_to_q = nn.Linear(action_dim, self.inner_dim, bias=True)
        self.action_to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, action_dim, bias=True), nn.Dropout(dropout)]
        )
        self.action_norm_q = deepcopy(self.norm_q)
        if cross_attention_dim_head is None:
            self.action_to_k = nn.Linear(action_dim, self.kv_inner_dim, bias=True)
            self.action_to_v = nn.Linear(action_dim, self.kv_inner_dim, bias=True)
            self.action_norm_k = deepcopy(self.norm_k)
        else:
            # Keep these aliases: they are present in the released state dict.
            self.action_to_k = self.to_k
            self.action_to_v = self.to_v
            self.action_norm_k = self.norm_k

        self.attn_caches = {}
        self.self_block_mask = None
        self.cross_block_mask = None

    def set_block_masks(self, self_mask=None, cross_mask=None) -> None:
        self.self_block_mask = self_mask
        self.cross_block_mask = cross_mask

    def clear_cache(self, cache_name: str) -> None:
        self.attn_caches.pop(cache_name, None)

    def prune_cache(self, cache_name: str, keep: torch.Tensor) -> None:
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return
        cache["k"] = cache["k"][:, keep]
        cache["v"] = cache["v"][:, keep]

    def _project(
        self,
        hs_latent: Optional[torch.Tensor],
        hs_action: Optional[torch.Tensor],
        hs_pad: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        latent_length = 0 if hs_latent is None else hs_latent.shape[1]
        action_length = 0 if hs_action is None else hs_action.shape[1]

        if encoder_hidden_states is not None:
            query_parts = []
            if hs_latent is not None:
                query_parts.append(
                    self.norm_q(self.to_q(hs_latent)).unflatten(2, (self.heads, -1))
                )
            if hs_action is not None:
                query_parts.append(
                    self.action_norm_q(self.action_to_q(hs_action)).unflatten(
                        2, (self.heads, -1)
                    )
                )
            if not query_parts:
                raise ValueError("Cross attention requires video or action queries")
            reference = query_parts[0]
            query_parts.append(
                reference.new_zeros(
                    reference.shape[0], hs_pad.shape[1], self.heads, reference.shape[-1]
                )
            )
            query = torch.cat(query_parts, dim=1)
            key = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(
                2, (self.heads, -1)
            )
            value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
            return query, key, value, latent_length, action_length

        projected = []
        if hs_latent is not None:
            projected.append(
                (
                    self.norm_q(self.to_q(hs_latent)).unflatten(2, (self.heads, -1)),
                    self.norm_k(self.to_k(hs_latent)).unflatten(2, (self.heads, -1)),
                    self.to_v(hs_latent).unflatten(2, (self.heads, -1)),
                )
            )
        if hs_action is not None:
            projected.append(
                (
                    self.action_norm_q(self.action_to_q(hs_action)).unflatten(
                        2, (self.heads, -1)
                    ),
                    self.action_norm_k(self.action_to_k(hs_action)).unflatten(
                        2, (self.heads, -1)
                    ),
                    self.action_to_v(hs_action).unflatten(2, (self.heads, -1)),
                )
            )
        if not projected:
            raise ValueError("Self attention requires video or action tokens")
        reference = projected[0][0]
        padding = reference.new_zeros(
            reference.shape[0], hs_pad.shape[1], self.heads, reference.shape[-1]
        )
        query = torch.cat([part[0] for part in projected] + [padding], dim=1)
        key = torch.cat([part[1] for part in projected] + [padding], dim=1)
        value = torch.cat([part[2] for part in projected] + [padding], dim=1)
        return query, key, value, latent_length, action_length

    def forward(
        self,
        hs_latent: Optional[torch.Tensor],
        hs_action: Optional[torch.Tensor],
        hs_pad: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        rotary_emb: Optional[torch.Tensor],
        update_cache: int,
        cache_name: str,
    ):
        query, key, value, latent_length, action_length = self._project(
            hs_latent, hs_action, hs_pad, encoder_hidden_states
        )
        is_cross_attention = encoder_hidden_states is not None
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)

        if not is_cross_attention:
            cache = self.attn_caches.get(cache_name)
            if cache is not None:
                key = torch.cat([cache["k"], key], dim=1)
                value = torch.cat([cache["v"], value], dim=1)

        hidden_states = ICLAttentionBackend.apply(
            query,
            key,
            value,
            cross_attention=is_cross_attention,
            block_mask=(
                self.cross_block_mask if is_cross_attention else self.self_block_mask
            ),
        ).flatten(2, 3)

        if not is_cross_attention and update_cache:
            self.attn_caches[cache_name] = {"k": key, "v": value}

        out_latent = None
        out_action = None
        if latent_length:
            out_latent = self.to_out[1](
                self.to_out[0](hidden_states[:, :latent_length])
            )
        if action_length:
            action_start = latent_length
            out_action = self.action_to_out[1](
                self.action_to_out[0](
                    hidden_states[:, action_start : action_start + action_length]
                )
            )
        return out_latent, out_action


class WanICLTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        action_ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
        action_inner_dim: int,
        attn_moe: bool = True,
        fully_share: bool = False,
    ):
        super().__init__()
        if fully_share:
            raise ValueError("The released ICL checkpoints require fully_share=False")
        action_dim = action_inner_dim if action_inner_dim > 0 else dim

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_norm1 = FP32LayerNorm(
            action_dim, eps, elementwise_affine=False
        )
        self.attn1 = WanICLAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            action_inner_dim=action_dim,
            attn_moe=attn_moe,
        )
        self.attn2 = WanICLAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            cross_attention_dim_head=dim // num_heads,
            action_inner_dim=action_dim,
            attn_moe=attn_moe,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.action_norm2 = (
            FP32LayerNorm(action_dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.ffn = FeedForward(
            dim, inner_dim=ffn_dim, activation_fn="gelu-approximate"
        )
        self.action_ffn = FeedForward(
            action_dim,
            inner_dim=action_ffn_dim,
            activation_fn="gelu-approximate",
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_norm3 = FP32LayerNorm(
            action_dim, eps, elementwise_affine=False
        )
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.scale_shift_table_action = nn.Parameter(
            torch.randn(1, 6, action_dim) / action_dim**0.5
        )

    @staticmethod
    def _modulation(table, temb):
        values = table[None] + temb.float()
        return values.unbind(dim=2)

    def forward(
        self,
        hs_latent: Optional[torch.Tensor],
        hs_action: Optional[torch.Tensor],
        hs_pad: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_latent: Optional[torch.Tensor],
        temb_action: Optional[torch.Tensor],
        rotary_emb: torch.Tensor,
        update_cache: int = 0,
        cache_name: str = "pos",
    ):
        latent_mod = (
            None
            if hs_latent is None
            else self._modulation(self.scale_shift_table, temb_latent)
        )
        action_mod = (
            None
            if hs_action is None
            else self._modulation(self.scale_shift_table_action, temb_action)
        )

        norm_latent = None
        if hs_latent is not None:
            shift, scale = latent_mod[0], latent_mod[1]
            norm_latent = (
                self.norm1(hs_latent.float()) * (1 + scale) + shift
            ).to(hs_latent.dtype)
        norm_action = None
        if hs_action is not None:
            shift, scale = action_mod[0], action_mod[1]
            norm_action = (
                self.action_norm1(hs_action.float()) * (1 + scale) + shift
            ).to(hs_action.dtype)

        attn_latent, attn_action = self.attn1(
            norm_latent,
            norm_action,
            hs_pad,
            None,
            rotary_emb,
            update_cache,
            cache_name,
        )
        if hs_latent is not None:
            hs_latent = (
                hs_latent.float() + attn_latent.float() * latent_mod[2]
            ).to(hs_latent.dtype)
        if hs_action is not None:
            hs_action = (
                hs_action.float() + attn_action.float() * action_mod[2]
            ).to(hs_action.dtype)

        cross_latent, cross_action = self.attn2(
            None
            if hs_latent is None
            else self.norm2(hs_latent.float()).to(hs_latent.dtype),
            None
            if hs_action is None
            else self.action_norm2(hs_action.float()).to(hs_action.dtype),
            hs_pad,
            encoder_hidden_states,
            None,
            0,
            cache_name,
        )
        if hs_latent is not None:
            hs_latent = hs_latent + cross_latent
            ff_input = (
                self.norm3(hs_latent.float()) * (1 + latent_mod[4])
                + latent_mod[3]
            ).to(hs_latent.dtype)
            hs_latent = (
                hs_latent.float() + self.ffn(ff_input).float() * latent_mod[5]
            ).to(hs_latent.dtype)
        if hs_action is not None:
            hs_action = hs_action + cross_action
            ff_input = (
                self.action_norm3(hs_action.float()) * (1 + action_mod[4])
                + action_mod[3]
            ).to(hs_action.dtype)
            hs_action = (
                hs_action.float()
                + self.action_ffn(ff_input).float() * action_mod[5]
            ).to(hs_action.dtype)
        return hs_latent, hs_action


class WanICLTransformer3DModel(ModelMixin, ConfigMixin):
    """Inference model for the released Robotwin ICL checkpoints."""

    _supports_gradient_checkpointing = False
    _no_split_modules = ["WanICLTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        action_inner_dim=3072,
        action_ffn_dim=14336,
        action_embedder_bias=True,
        attn_moe=True,
        fully_share=False,
        attn_window=64,
        enable_mcp=True,
        num_mcp_modules=4,
        mcp_blocks_per_group=1,
        mcp_hidden_collect_layers=(3, 11, 19, 29),
        enable_rgb_transition=False,
        rgb_transition_config=None,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.action_inner_dim = action_inner_dim
        self.attn_window = attn_window
        self.enable_rgb_transition = bool(enable_rgb_transition)
        self.rgb_transition = (
            RGBTransitionConditioner(rgb_transition_config or {}, video_dim=self.inner_dim)
            if self.enable_rgb_transition else None
        )

        self.rope = WanRotaryPosEmbed(
            attention_head_dim, self.patch_size, rope_max_seq_len
        )
        patch_channels = in_channels * math.prod(self.patch_size)
        self.patch_embedding_mlp = nn.Linear(patch_channels, self.inner_dim)
        # Kept for exact compatibility with the training checkpoint.
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.action_embedder = nn.Linear(
            action_dim, action_inner_dim, bias=action_embedder_bias
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            self.inner_dim,
            freq_dim,
            self.inner_dim * 6,
            text_dim,
            pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        block_kwargs = dict(
            dim=self.inner_dim,
            ffn_dim=ffn_dim,
            action_ffn_dim=action_ffn_dim,
            num_heads=num_attention_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            action_inner_dim=action_inner_dim,
            attn_moe=attn_moe,
            fully_share=fully_share,
        )
        self.blocks = nn.ModuleList(
            [WanICLTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )

        self.enable_mcp = enable_mcp
        self.mcp_hidden_collect_layers = list(mcp_hidden_collect_layers)
        if enable_mcp:
            self.mcp_mlp_hidden = nn.Sequential(
                nn.Linear(
                    self.inner_dim * len(self.mcp_hidden_collect_layers),
                    self.inner_dim,
                    bias=True,
                ),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim, bias=True),
            )
            self.mcp_projections = nn.ModuleList(
                [
                    nn.Linear(self.inner_dim * 2, self.inner_dim, bias=True)
                    for _ in range(num_mcp_modules)
                ]
            )
            self.mcp_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            WanICLTransformerBlock(**block_kwargs)
                            for _ in range(mcp_blocks_per_group)
                        ]
                    )
                    for _ in range(num_mcp_modules)
                ]
            )

        self.norm_out = FP32LayerNorm(
            self.inner_dim, eps, elementwise_affine=False
        )
        self.action_norm_out = FP32LayerNorm(
            action_inner_dim, eps, elementwise_affine=False
        )
        self.proj_out = nn.Linear(
            self.inner_dim, out_channels * math.prod(self.patch_size)
        )
        self.action_proj_out = nn.Linear(action_inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5
        )
        self.scale_shift_table_action = nn.Parameter(
            torch.randn(1, 2, action_inner_dim) / action_inner_dim**0.5
        )
        self.clear_cache()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load an unchanged backbone, explicitly initializing only a new RGB branch.

        Diffusers constructs models with initialization disabled (and often on
        meta). Missing new parameters therefore cannot be used as initialized
        weights, even when its load report calls them "newly initialized".
        """
        return_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            *args, output_loading_info=True, **kwargs
        )
        if model.enable_rgb_transition:
            missing = set(loading_info.get("missing_keys", []))
            new_keys = {
                f"rgb_transition.{key}" for key in model.rgb_transition.state_dict()
            }
            invalid_missing = missing - new_keys
            if (
                invalid_missing
                or loading_info.get("unexpected_keys")
                or loading_info.get("mismatched_keys")
                or loading_info.get("error_msgs")
            ):
                raise RuntimeError(
                    "RGB-transition checkpoint does not match the backbone: "
                    f"{loading_info}"
                )
            if missing:
                if missing != new_keys:
                    raise RuntimeError(
                        "Refusing a partially missing RGB-transition checkpoint; "
                        f"missing keys: {sorted(missing)}"
                    )
                if kwargs.get("device_map") is not None:
                    raise ValueError(
                        "Initialize the new RGB branch on CPU before device_map dispatch"
                    )
                reference = model.patch_embedding_mlp.weight
                model.rgb_transition = RGBTransitionConditioner(
                    dict(model.config.rgb_transition_config or {}),
                    video_dim=model.inner_dim,
                ).to(device=reference.device, dtype=reference.dtype)
                model.rgb_transition.train(model.training)
                logger.info(
                    "Initialized %d new RGB-transition tensors; backbone loaded unchanged. "
                    "Missing checkpoint keys: %s", len(missing), sorted(missing)
                )
            model.rgb_transition_loading_info = loading_info
        return (model, loading_info) if return_loading_info else model

    def _rgb_zero_losses(self):
        # Different ranks can independently drop ICL. Keep every new parameter
        # in the graph so these ranks still run matching FSDP reductions. Match
        # FP32 auxiliary losses on other ranks, including metric all-reduces.
        zero = sum(
            parameter.reshape(-1)[0].float() * 0
            for parameter in self.rgb_transition.parameters()
            if parameter.requires_grad and parameter.numel()
        )
        if not torch.is_tensor(zero):
            zero = self.patch_embedding_mlp.weight.new_zeros((), dtype=torch.float32)
        return {"human_self": zero, "robot_self": zero, "transfer": zero, "total": zero}

    @property
    def uses_rgb_readout(self):
        return self.enable_rgb_transition and self.rgb_transition.readout_mode != "human"

    def _encode_rgb_transition(self, human_hidden, input_dict, *, compute_aux):
        rgb_inputs = input_dict.get("rgb_transition")
        if not isinstance(rgb_inputs, dict) or "human_features" not in rgb_inputs:
            raise ValueError(
                "RGB-transition conditioning is enabled but human_features are missing"
            )
        human_features = rgb_inputs["human_features"]
        if not torch.is_tensor(human_features) or human_features.ndim != 5:
            raise ValueError("human_features must have shape [B, S, T, P, D]")
        human_features = human_features.to(
            device=human_hidden.device, dtype=human_hidden.dtype
        )
        robot_features = rgb_inputs.get("robot_features") if compute_aux else None
        if robot_features is not None:
            robot_features = robot_features.to(
                device=human_hidden.device, dtype=human_hidden.dtype
            )
        tokens, losses = self.rgb_transition(
            human_features, robot_features=robot_features, compute_aux=compute_aux
        )
        if compute_aux:
            # Also connect any unused auxiliary parameters when a batch has no
            # robot teacher, or an ordinary-feature control disables a loss.
            zero = self._rgb_zero_losses()["total"]
            losses = dict(losses)
            losses["total"] = losses.get("total", zero) + zero
        return tokens, losses

    def _fuse_rgb_transition(self, human_hidden, input_dict, *, compute_aux):
        tokens, losses = self._encode_rgb_transition(
            human_hidden, input_dict, compute_aux=compute_aux
        )
        return self.rgb_transition.fuse(human_hidden, tokens), losses

    def _training_rgb_readout(self, noisy, clean, memory, input_dict):
        """Read each chunk from its preceding observed frame, never its future.

        The first chunk has the recorded initial observation (frame zero), as
        in server inference. All later chunks use frame start-1. Clean/history
        tokens get the same causal conditioning as their noisy counterparts.
        """
        stream = input_dict["latent_dict"]
        grid = stream["grid_id"][0]
        frame_count = stream["latent"].shape[2]
        if self.patch_size[0] != 1:
            raise ValueError("RGB readout requires temporal patch size 1")
        patches = clean.shape[1] // frame_count
        if patches * frame_count != clean.shape[1]:
            raise ValueError("Invalid RGB readout frame/patch layout")
        observation_latents = input_dict.get("readout_observation_latents")
        if not torch.is_tensor(observation_latents):
            raise ValueError("RGB readout requires original clean readout_observation_latents")
        if tuple(observation_latents.shape) != tuple(stream["latent"].shape):
            raise ValueError("readout_observation_latents must match the robot latent layout")
        # Backbone history may receive conditional-noise augmentation. The
        # readout instead observes the original clean frame, as deployment does.
        observation_states = rearrange(
            observation_latents.to(self.patch_embedding_mlp.weight),
            "b c f (h p2) (w p3) -> b (f h w) (c p2 p3)",
            p2=self.patch_size[1], p3=self.patch_size[2],
        )
        observation_states = self.patch_embedding_mlp(observation_states).reshape(
            1, frame_count, patches, self.inner_dim)
        clean_frames = clean.reshape(1, frame_count, patches, self.inner_dim)
        noisy_frames = noisy.reshape_as(clean_frames)
        chunk = int(input_dict["chunk_size"])
        teacher_targets = input_dict.get("readout_teacher_targets")
        teacher_valid = input_dict.get("readout_teacher_valid")
        expected_chunks = (frame_count + chunk - 1) // chunk
        if teacher_targets is not None:
            expected = (expected_chunks, self.rgb_transition.readout.num_tokens,
                        self.rgb_transition.hidden_dim)
            if tuple(teacher_targets.shape) != expected or teacher_valid is None:
                raise ValueError("Future teacher targets do not match readout chunks")
            if tuple(teacher_valid.shape) != (expected_chunks,):
                raise ValueError("Future teacher validity does not match chunks")
        new_noisy, new_clean, alignments = [], [], []
        for index, start in enumerate(range(0, frame_count, chunk)):
            end = min(start + chunk, frame_count)
            observed = max(start - 1, 0)
            state = observation_states[:, observed]
            positions = grid[1:3, observed * patches:(observed + 1) * patches].T
            tokens = self.rgb_transition.read_tokens(memory, state, positions)
            new_noisy.append(self.rgb_transition.fuse(
                noisy_frames[:, start:end].flatten(1, 2), tokens))
            new_clean.append(self.rgb_transition.fuse(
                clean_frames[:, start:end].flatten(1, 2), tokens))
            if teacher_targets is not None and bool(teacher_valid[index]):
                target = teacher_targets[index:index + 1].detach().float()
                alignments.append((1 - F.cosine_similarity(
                    tokens.float(), target, dim=-1, eps=1e-6)).mean())
        alignment = (torch.stack(alignments).mean() if alignments
                     else memory.float().sum() * 0)
        return torch.cat(new_noisy, 1), torch.cat(new_clean, 1), alignment

    def _inference_rgb_readout(self, hidden, input_dict, grid, seq_ids):
        if self.rgb_readout_memory is None:
            return hidden
        positive = seq_ids == 0
        if not bool(positive.any()):
            return hidden
        state_latent = input_dict.get("readout_state_latent")
        if state_latent is None:
            raise ValueError("RGB readout requires an actual observed readout_state_latent")
        if state_latent.ndim != 5 or state_latent.shape[0] != 1 or state_latent.shape[2] != 1:
            raise ValueError("readout_state_latent must contain exactly one observed frame")
        state = rearrange(
            state_latent.to(self.patch_embedding_mlp.weight),
            "b c f (h p2) (w p3) -> b (f h w) (c p2 p3)",
            p2=self.patch_size[1], p3=self.patch_size[2],
        )
        state = self.patch_embedding_mlp(state)
        # The first spatial grid within the positive stream matches its state.
        positions = grid[1:3, positive][:, :state.shape[1]].T
        if positions.shape[0] != state.shape[1]:
            raise ValueError("Readout observation and target camera grids differ")
        tokens = self.rgb_transition.read_tokens(self.rgb_readout_memory, state, positions)
        # Keep CFG-negative hidden states bitwise untouched, including packed CFG.
        result = hidden.clone()
        result[:, positive] = self.rgb_transition.fuse(hidden[:, positive], tokens)
        return result

    def disable_mcp_modules(self) -> None:
        self.enable_mcp = False

    def clear_cache(self, cache_name: str = "pos") -> None:
        self.rgb_readout_memory = None
        self.type_ids_cache = None
        self.seq_ids_cache = None
        self.frame_ids_cache = None
        self.cache_type_ids_cache = None
        if hasattr(self, "blocks"):
            for block in self.blocks:
                block.attn1.clear_cache(cache_name)

    def _prune_cache(self, keep: torch.Tensor, cache_name: str = "pos") -> None:
        if self.type_ids_cache is None:
            return
        self.type_ids_cache = self.type_ids_cache[keep]
        self.seq_ids_cache = self.seq_ids_cache[keep]
        self.frame_ids_cache = self.frame_ids_cache[keep]
        self.cache_type_ids_cache = self.cache_type_ids_cache[keep]
        for block in self.blocks:
            block.attn1.prune_cache(cache_name, keep)

    def clear_prediction_cache(
        self, attn_window: Optional[int] = None, cache_name: str = "pos"
    ) -> None:
        """Drop generated type-1 entries while retaining ICL for the episode."""
        if self.cache_type_ids_cache is None:
            return
        keep_icl = self.cache_type_ids_cache == ICL_CACHE_TYPE
        keep_obs = self.cache_type_ids_cache == OBSERVATION_CACHE_TYPE
        window = self.attn_window if attn_window is None else attn_window
        if window > 0 and keep_obs.any():
            next_frame_id = self.frame_ids_cache[keep_obs].max() + 1
            keep_obs = keep_obs & (
                (self.frame_ids_cache - next_frame_id).abs() <= window
            )
        self._prune_cache(keep_icl | keep_obs, cache_name)

    def cache_counts(self) -> dict[int, int]:
        if self.cache_type_ids_cache is None:
            return {}
        valid_tokens = self.frame_ids_cache != -1
        return {
            cache_type: int(
                ((self.cache_type_ids_cache == cache_type) & valid_tokens).sum()
            )
            for cache_type in (
                OBSERVATION_CACHE_TYPE,
                PREDICTION_CACHE_TYPE,
                ICL_CACHE_TYPE,
            )
        }

    def _embed_stream(self, stream, mode):
        latents = stream["noisy_latents"]
        if mode == "video":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
            repeats = (latents.shape[-2] // self.patch_size[1]) * (
                latents.shape[-1] // self.patch_size[2]
            )
            condition_embedder = self.condition_embedder
        else:
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
            repeats = latents.shape[-2] * latents.shape[-1]
            condition_embedder = self.condition_embedder_action
        timesteps = stream["timesteps"]
        if timesteps.ndim == 1:
            timesteps = timesteps[None]
        token_timesteps = torch.repeat_interleave(timesteps, repeats, dim=1)
        temb, timestep_proj = condition_embedder(
            token_timesteps, dtype=hidden_states.dtype
        )
        return hidden_states, temb, timestep_proj.unflatten(2, (6, -1))

    def _append_metadata(
        self,
        query_types,
        query_seq_ids,
        query_frame_ids,
        cache_type_ids,
    ) -> None:
        stored_types = query_types.clone()
        stored_types[cache_type_ids == ICL_CACHE_TYPE] = ICL_CACHE_TYPE
        for name, value in (
            ("type_ids_cache", stored_types),
            ("seq_ids_cache", query_seq_ids),
            ("frame_ids_cache", query_frame_ids),
            ("cache_type_ids_cache", cache_type_ids),
        ):
            previous = getattr(self, name)
            setattr(self, name, value if previous is None else torch.cat([previous, value]))

    @staticmethod
    def _set_masks(blocks, self_mask, cross_mask) -> None:
        for block in blocks:
            block.attn1.set_block_masks(self_mask=self_mask)
            block.attn2.set_block_masks(cross_mask=cross_mask)

    def _training_embed(self, latents, timesteps, mode):
        model_dtype = self.patch_embedding_mlp.weight.dtype
        stream = {
            "noisy_latents": latents.to(dtype=model_dtype),
            "timesteps": timesteps,
        }
        return self._embed_stream(stream, mode)

    @staticmethod
    def _pad_metadata(value, padded_length):
        return F.pad(value, (0, padded_length), value=-1)

    def _build_training_masks(
        self,
        seq_ids,
        frame_ids,
        noise_ids,
        type_ids,
        icl_ids,
        cross_seq_ids,
        encoder_seq_ids,
        window_size,
        blocks,
    ) -> None:
        self_mask = ICLAttentionBackend.build_training_self_mask(
            seq_ids=seq_ids,
            frame_ids=frame_ids,
            noise_ids=noise_ids,
            type_ids=type_ids,
            icl_ids=icl_ids,
            window_size=window_size,
            device=seq_ids.device,
        )
        cross_mask = ICLAttentionBackend.build_cross_mask(
            query_seq_ids=cross_seq_ids,
            encoder_seq_ids=encoder_seq_ids,
            device=seq_ids.device,
        )
        self._set_masks(blocks, self_mask, cross_mask)

    def _forward_training_mcp(
        self,
        input_dict,
        collected_hidden_states,
        hs_action,
        action_timestep_proj,
        text_hidden_states,
        encoder_seq_ids,
        target_grid,
        action_grid,
        target_frame_ids,
        action_frame_ids,
        target_clean_timestep_proj,
        target_length,
        action_length,
    ):
        mcp_streams = input_dict.get("mcp_latent_dicts", [])
        if not self.enable_mcp or not mcp_streams:
            return []
        if len(collected_hidden_states) != len(self.mcp_hidden_collect_layers):
            raise RuntimeError(
                "MCP did not collect every configured backbone hidden state"
            )

        fused = self.mcp_mlp_hidden(torch.cat(collected_hidden_states, dim=-1))
        main_noisy = fused[:, :target_length]
        main_clean = fused[:, target_length : target_length * 2]
        device = main_noisy.device
        outputs = []

        for module_index, (group, stream) in enumerate(
            zip(self.mcp_blocks, mcp_streams)
        ):
            future_hs, future_temb, future_timestep_proj = self._training_embed(
                stream["noisy_latents"], stream["timesteps"], "video"
            )
            if future_hs.shape[1] != target_length:
                raise ValueError("MCP and target video token lengths must match")
            mcp_hs = self.mcp_projections[module_index](
                torch.cat([main_noisy, future_hs], dim=-1)
            )
            mcp_hs = torch.cat([mcp_hs, main_clean], dim=1)
            mcp_action = hs_action

            future_grid = stream["grid_id"][0]
            full_grid = torch.cat(
                [future_grid, target_grid, action_grid, action_grid], dim=1
            )
            total_length = target_length * 2 + action_length * 2
            padded_length = 128 - total_length % 128
            hs_pad = mcp_hs.new_zeros(
                mcp_hs.shape[0], padded_length, self.inner_dim
            )
            rotary_emb = self.rope(full_grid[None])[:, :, None]
            rotary_emb = F.pad(
                rotary_emb, (0, 0, 0, 0, 0, padded_length)
            )

            seq_ids = torch.zeros(total_length, device=device, dtype=torch.int)
            frame_ids = torch.cat(
                [
                    target_frame_ids,
                    target_frame_ids,
                    action_frame_ids,
                    action_frame_ids,
                ]
            )
            noise_ids = torch.cat(
                [
                    torch.zeros(target_length, device=device, dtype=torch.int),
                    torch.ones(target_length, device=device, dtype=torch.int),
                    torch.zeros(action_length, device=device, dtype=torch.int),
                    torch.ones(action_length, device=device, dtype=torch.int),
                ]
            )
            type_ids = torch.cat(
                [
                    torch.zeros(target_length * 2, device=device, dtype=torch.int),
                    torch.ones(action_length * 2, device=device, dtype=torch.int),
                ]
            )
            icl_ids = torch.zeros_like(type_ids)
            cross_seq_ids = torch.zeros_like(seq_ids)
            seq_ids = self._pad_metadata(seq_ids, padded_length)
            frame_ids = self._pad_metadata(frame_ids, padded_length)
            noise_ids = self._pad_metadata(noise_ids, padded_length)
            type_ids = self._pad_metadata(type_ids, padded_length)
            icl_ids = self._pad_metadata(icl_ids, padded_length)
            cross_seq_ids = self._pad_metadata(cross_seq_ids, padded_length)
            self._build_training_masks(
                seq_ids,
                frame_ids,
                noise_ids,
                type_ids,
                icl_ids,
                cross_seq_ids,
                encoder_seq_ids,
                input_dict["window_size"],
                group,
            )

            mcp_timestep_proj = torch.cat(
                [future_timestep_proj, target_clean_timestep_proj], dim=1
            )
            for block in group:
                mcp_hs, mcp_action = block(
                    mcp_hs,
                    mcp_action,
                    hs_pad,
                    text_hidden_states,
                    mcp_timestep_proj,
                    action_timestep_proj,
                    rotary_emb,
                )

            mcp_output = mcp_hs[:, :target_length]
            shift, scale = (
                self.scale_shift_table[None] + future_temb[:, :, None]
            ).unbind(dim=2)
            mcp_output = (
                self.norm_out(mcp_output.float()) * (1 + scale) + shift
            ).to(mcp_output.dtype)
            mcp_output = self.proj_out(mcp_output)
            outputs.append(
                rearrange(
                    mcp_output,
                    "1 l (n c) -> 1 (l n) c",
                    n=math.prod(self.patch_size),
                )
            )
        return outputs

    def forward_train(self, input_dict):
        latent = input_dict["latent_dict"]
        action = input_dict["action_dict"]
        if latent["noisy_latents"].shape[0] != 1:
            raise ValueError(
                "Open-source ICL training uses one sample per rank; set batch_size=1"
            )

        target_noisy_hs, target_temb, target_noisy_timestep_proj = (
            self._training_embed(
                latent["noisy_latents"], latent["timesteps"], "video"
            )
        )
        target_clean_hs, _, target_clean_timestep_proj = self._training_embed(
            latent["latent"], latent["cond_timesteps"], "video"
        )
        action_noisy_hs, action_temb, action_noisy_timestep_proj = (
            self._training_embed(
                action["noisy_latents"], action["timesteps"], "action"
            )
        )
        action_clean_hs, _, action_clean_timestep_proj = self._training_embed(
            action["latent"], action["cond_timesteps"], "action"
        )

        target_length = target_noisy_hs.shape[1]
        action_length = action_noisy_hs.shape[1]
        latent_parts = [target_noisy_hs, target_clean_hs]
        latent_timestep_parts = [
            target_noisy_timestep_proj,
            target_clean_timestep_proj,
        ]
        grid_parts = [latent["grid_id"][0], latent["grid_id"][0]]

        icl = input_dict.get("icl_latent_dict")
        rgb_losses = self._rgb_zero_losses() if self.enable_rgb_transition else {}
        icl_length = 0
        if icl is not None:
            icl_hs, _, icl_timestep_proj = self._training_embed(
                icl["latent"], icl["timesteps"], "video"
            )
            if self.enable_rgb_transition:
                if self.uses_rgb_readout:
                    memory, rgb_losses = self._encode_rgb_transition(
                        icl_hs, input_dict, compute_aux=True)
                    target_noisy_hs, target_clean_hs, alignment = self._training_rgb_readout(
                        target_noisy_hs, target_clean_hs, memory, input_dict)
                    latent_parts = [target_noisy_hs, target_clean_hs]
                    rgb_losses["teacher_alignment"] = alignment
                else:
                    icl_hs, rgb_losses = self._fuse_rgb_transition(
                        icl_hs, input_dict, compute_aux=True
                    )
            icl_length = icl_hs.shape[1]
            latent_parts.append(icl_hs)
            latent_timestep_parts.append(icl_timestep_proj)
            grid_parts.append(icl["grid_id"][0])

        hs_latent = torch.cat(latent_parts, dim=1)
        hs_action = torch.cat([action_noisy_hs, action_clean_hs], dim=1)
        latent_timestep_proj = torch.cat(latent_timestep_parts, dim=1)
        action_timestep_proj = torch.cat(
            [action_noisy_timestep_proj, action_clean_timestep_proj], dim=1
        )
        target_grid = latent["grid_id"][0]
        action_grid = action["grid_id"][0]
        full_grid = torch.cat(grid_parts + [action_grid, action_grid], dim=1)

        total_length = hs_latent.shape[1] + hs_action.shape[1]
        padded_length = 128 - total_length % 128
        hs_pad = hs_latent.new_zeros(
            hs_latent.shape[0], padded_length, self.inner_dim
        )
        rotary_emb = self.rope(full_grid[None])[:, :, None]
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))

        device = hs_latent.device
        chunk_size = int(input_dict["chunk_size"])
        max_chunk_size = int(input_dict["max_frame_chunk_size"])
        target_frame_ids = (target_grid[0] // chunk_size * 2).to(torch.int)
        action_frame_ids = (action_grid[0] // chunk_size * 2 + 1).to(torch.int)
        if icl is not None:
            icl_frame_ids = (icl["grid_id"][0][0] // max_chunk_size).to(torch.int)
        else:
            icl_frame_ids = torch.empty(0, device=device, dtype=torch.int)

        seq_ids = torch.zeros(total_length, device=device, dtype=torch.int)
        frame_ids = torch.cat(
            [
                target_frame_ids,
                target_frame_ids,
                icl_frame_ids,
                action_frame_ids,
                action_frame_ids,
            ]
        )
        noise_ids = torch.cat(
            [
                torch.zeros(target_length, device=device, dtype=torch.int),
                torch.ones(target_length, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length, device=device, dtype=torch.int),
                torch.ones(action_length, device=device, dtype=torch.int),
            ]
        )
        type_ids = torch.cat(
            [
                torch.zeros(
                    target_length * 2 + icl_length,
                    device=device,
                    dtype=torch.int,
                ),
                torch.ones(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        icl_ids = torch.cat(
            [
                torch.zeros(target_length * 2, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        cross_seq_ids = torch.cat(
            [
                torch.zeros(target_length * 2, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        seq_ids = self._pad_metadata(seq_ids, padded_length)
        frame_ids = self._pad_metadata(frame_ids, padded_length)
        noise_ids = self._pad_metadata(noise_ids, padded_length)
        type_ids = self._pad_metadata(type_ids, padded_length)
        icl_ids = self._pad_metadata(icl_ids, padded_length)
        cross_seq_ids = self._pad_metadata(cross_seq_ids, padded_length)

        text_emb = input_dict["text_emb"].to(
            device=device, dtype=self.condition_embedder.text_embedder.linear_1.weight.dtype
        )
        text_hidden_states = self.condition_embedder.text_embedder(text_emb)
        encoder_seq_ids = input_dict["encoder_seq_ids"].to(
            device=device, dtype=torch.int
        )
        self._build_training_masks(
            seq_ids,
            frame_ids,
            noise_ids,
            type_ids,
            icl_ids,
            cross_seq_ids,
            encoder_seq_ids,
            input_dict["window_size"],
            self.blocks,
        )

        collected = []
        for layer_index, block in enumerate(self.blocks):
            hs_latent, hs_action = block(
                hs_latent,
                hs_action,
                hs_pad,
                text_hidden_states,
                latent_timestep_proj,
                action_timestep_proj,
                rotary_emb,
            )
            if self.enable_mcp and layer_index in self.mcp_hidden_collect_layers:
                collected.append(hs_latent[:, : target_length * 2])

        latent_output = hs_latent[:, :target_length]
        action_output = hs_action[:, :action_length]
        shift, scale = (
            self.scale_shift_table[None] + target_temb[:, :, None]
        ).unbind(dim=2)
        latent_output = (
            self.norm_out(latent_output.float()) * (1 + scale) + shift
        ).to(latent_output.dtype)
        latent_output = self.proj_out(latent_output)
        latent_output = rearrange(
            latent_output,
            "1 l (n c) -> 1 (l n) c",
            n=math.prod(self.patch_size),
        )

        action_shift, action_scale = (
            self.scale_shift_table_action[None] + action_temb[:, :, None]
        ).unbind(dim=2)
        action_output = (
            self.action_norm_out(action_output.float())
            * (1 + action_scale)
            + action_shift
        ).to(action_output.dtype)
        action_output = self.action_proj_out(action_output)

        mcp_outputs = self._forward_training_mcp(
            input_dict=input_dict,
            collected_hidden_states=collected,
            hs_action=hs_action,
            action_timestep_proj=action_timestep_proj,
            text_hidden_states=text_hidden_states,
            encoder_seq_ids=encoder_seq_ids,
            target_grid=target_grid,
            action_grid=action_grid,
            target_frame_ids=target_frame_ids,
            action_frame_ids=action_frame_ids,
            target_clean_timestep_proj=target_clean_timestep_proj,
            target_length=target_length,
            action_length=action_length,
        )
        base_pred = (
            (latent_output, action_output, mcp_outputs)
            if self.enable_mcp else (latent_output, action_output)
        )
        if self.enable_rgb_transition:
            return {"base_pred": base_pred, "rgb_losses": rgb_losses}
        return base_pred

    def _forward_stream(
        self,
        input_dict,
        mode: str,
        update_cache: int,
        cache_name: str,
        clean_window_cache: bool,
    ):
        stream_key = "latent_res_lst" if mode == "video" else "action_res_lst"
        grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
        stream = input_dict[stream_key]
        hidden_states, temb, timestep_proj = self._embed_stream(stream, mode)
        if hidden_states.shape[0] != 1:
            raise ValueError("ICL inference expects a single sequence")

        grid_id = input_dict.get(grid_key, stream.get("grid_id"))
        if grid_id is None:
            raise KeyError(f"Missing {grid_key}")
        if grid_id.ndim == 3:
            grid_id = grid_id[0]
        rotary_emb = self.rope(grid_id[None])[:, :, None]

        token_count = hidden_states.shape[1]
        device = hidden_states.device
        query_type = torch.full(
            [token_count], 0 if mode == "video" else 1, device=device, dtype=torch.int
        )
        query_seq = input_dict.get(
            "current_seq_ids", torch.zeros(token_count, device=device, dtype=torch.int)
        ).to(device=device, dtype=torch.int)
        query_frame = input_dict.get(
            "current_frame_ids", grid_id[0].to(device=device, dtype=torch.int)
        ).to(device=device, dtype=torch.int)
        cache_types = stream.get(
            "cache_type_ids",
            torch.full(
                [token_count],
                int(input_dict.get("cache_type_id", PREDICTION_CACHE_TYPE)),
                device=device,
                dtype=torch.int,
            ),
        ).to(device=device, dtype=torch.int)
        if self.enable_rgb_transition and mode == "video":
            human_queries = cache_types == ICL_CACHE_TYPE
            if human_queries.any():
                # The public server caches a complete human demonstration in a
                # dedicated call. Reject mixed streams rather than accidentally
                # conditioning target/CFG-negative queries on human RGB.
                if not human_queries.all() or not (query_seq == 0).all():
                    raise ValueError(
                        "RGB-transition ICL caching requires a human-only, seq_id=0 stream"
                    )
                if self.uses_rgb_readout:
                    if not update_cache:
                        raise ValueError("Human readout memory must be installed with update_cache")
                    memory, _ = self._encode_rgb_transition(
                        hidden_states, input_dict, compute_aux=False)
                    self.rgb_readout_memory = memory.detach()
                else:
                    hidden_states, _ = self._fuse_rgb_transition(
                        hidden_states, input_dict, compute_aux=False
                    )
            elif self.uses_rgb_readout:
                hidden_states = self._inference_rgb_readout(
                    hidden_states, input_dict, grid_id, query_seq)
        for name, value in (
            ("current_seq_ids", query_seq),
            ("current_frame_ids", query_frame),
            ("cache_type_ids", cache_types),
        ):
            if value.numel() != token_count:
                raise ValueError(
                    f"{name} has {value.numel()} entries for {token_count} tokens"
                )

        padded_length = 128 - token_count % 128
        hs_pad = hidden_states.new_zeros(1, padded_length, self.inner_dim)
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        query_type = F.pad(query_type, (0, padded_length), value=-1)
        query_seq = F.pad(query_seq, (0, padded_length), value=-1)
        query_frame = F.pad(query_frame, (0, padded_length), value=-1)
        # VA keeps alignment padding in the KV cache for persistent ICL and
        # observation chunks.  Preserve the owning cache type so pruning drops
        # prediction padding but retains the same KV layout as deployment.  The
        # modality type remains -1, so these padding keys are still masked out.
        padding_cache_type = int(cache_types[0].item())
        cache_types = F.pad(
            cache_types, (0, padded_length), value=padding_cache_type
        )

        key_type = (
            query_type
            if self.type_ids_cache is None
            else torch.cat([self.type_ids_cache, query_type])
        )
        key_seq = (
            query_seq
            if self.seq_ids_cache is None
            else torch.cat([self.seq_ids_cache, query_seq])
        )
        key_frame = (
            query_frame
            if self.frame_ids_cache is None
            else torch.cat([self.frame_ids_cache, query_frame])
        )
        self_mask = ICLAttentionBackend.build_self_mask(
            query_type,
            key_type,
            query_seq,
            key_seq,
            query_frame,
            key_frame,
            self.attn_window,
            device,
            compile_mask=mode != "action",
        )

        text_emb = input_dict["text_emb"]
        text_hidden_states = self.condition_embedder.text_embedder(text_emb)
        encoder_seq = input_dict.get(
            "encoder_seq_ids",
            torch.zeros(text_hidden_states.shape[1], device=device, dtype=torch.int),
        ).to(device=device, dtype=torch.int)
        cross_mask = ICLAttentionBackend.build_cross_mask(
            query_seq,
            encoder_seq,
            device,
            compile_mask=mode != "action",
        )
        self._set_masks(self.blocks, self_mask, cross_mask)

        hs_latent = hidden_states if mode == "video" else None
        hs_action = hidden_states if mode == "action" else None
        for block in self.blocks:
            hs_latent, hs_action = block(
                hs_latent,
                hs_action,
                hs_pad,
                text_hidden_states,
                timestep_proj if mode == "video" else None,
                timestep_proj if mode == "action" else None,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        output_states = hs_latent if mode == "video" else hs_action
        output_table = (
            self.scale_shift_table
            if mode == "video"
            else self.scale_shift_table_action
        )
        output_norm = self.norm_out if mode == "video" else self.action_norm_out
        shift, scale = (output_table[None] + temb[:, :, None]).unbind(dim=2)
        output_states = (
            output_norm(output_states.float()) * (1 + scale) + shift
        ).to(output_states.dtype)
        if mode == "video":
            output = self.proj_out(output_states)
            output = rearrange(
                output,
                "1 l (n c) -> 1 (l n) c",
                n=math.prod(self.patch_size),
            )
        else:
            output = self.action_proj_out(output_states)

        if update_cache:
            self._append_metadata(query_type, query_seq, query_frame, cache_types)
            if clean_window_cache:
                self.clear_prediction_cache(cache_name=cache_name)
        return output

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        mode="forward_latent_only",
        clean_window_cache=False,
        return_dict=False,
        train_mode=False,
        **kwargs,
    ):
        del kwargs
        if return_dict:
            raise ValueError("WanICLTransformer3DModel returns tuples or tensors")
        if train_mode:
            return self.forward_train(input_dict)
        if mode in ("forward_latent_only", "forward_latent_only_flash"):
            return self._forward_stream(
                input_dict,
                "video",
                update_cache,
                cache_name,
                clean_window_cache,
            )
        if mode in ("forward_action_only", "forward_action_only_flash"):
            return self._forward_stream(
                input_dict,
                "action",
                update_cache,
                cache_name,
                clean_window_cache,
            )
        raise ValueError(f"Unsupported ICL inference mode: {mode}")


__all__ = [
    "ICL_CACHE_TYPE",
    "OBSERVATION_CACHE_TYPE",
    "PREDICTION_CACHE_TYPE",
    "WanICLTransformer3DModel",
]
