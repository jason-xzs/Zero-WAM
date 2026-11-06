"""Action-shaped RGB future targets, trained separately from the policy.

The query encoder adapts the actual BERT cross-attention QFormer from WoG:
https://github.com/Selen-Suyue/WoG/blob/872d1df6055c0dc5d2504d4d6a5fb71c2bae3e6c/vla/qformer.py
The source's learnable queries, decoder/cross-attention BERT configuration and
last-hidden projection are retained. We remove its unused VAE projection and
pooler, reduce the unused vocabulary, change output width to the student's
width, and add explicit temporal/spatial positions to frozen RGB features.

The flow action head and artifact checks below are this repository's adaptation,
not a reproduction of WoG's 7B VLM or its results. Crucially, future queries
never see actions, noised actions, or current robot context. Only the separate
action decoder sees those inputs during teacher training. Deployment discards
the entire teacher; the policy learns to predict its detached query targets.

WoG source license (preserved from its repository):
MIT License, Copyright (c) Microsoft Corporation.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from pathlib import Path
import hashlib
import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel


TEACHER_SCHEMA = 1
WOG_SOURCE = {
    "repository": "https://github.com/Selen-Suyue/WoG",
    "commit": "872d1df6055c0dc5d2504d4d6a5fb71c2bae3e6c",
    "file": "vla/qformer.py",
    "adaptation": "RGB-only query teacher with temporal/spatial positions and a small flow action decoder",
}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ActionFutureTeacher(nn.Module):
    """Frozen RGB -> K future queries, shaped by a masked action flow loss."""

    def __init__(self, config=None, **kwargs):
        super().__init__()
        defaults = dict(feature_dim=768, hidden_dim=256, num_tokens=32,
                        num_heads=4, encoder_layers=2, action_layers=2,
                        action_dim=30, max_future_steps=4, max_patches=256,
                        max_action_steps=64, norm_eps=1e-6)
        supplied = dict(config or {})
        supplied.update(kwargs)
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown future teacher config fields: {sorted(unknown)}")
        defaults.update(supplied)
        self.config = defaults
        for name, value in defaults.items():
            if name != "norm_eps":
                setattr(self, name, _positive(value, name))
        self.norm_eps = float(defaults["norm_eps"])
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.visual_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.future_time = nn.Parameter(torch.randn(self.max_future_steps, self.hidden_dim) * .02)
        self.spatial_position = nn.Parameter(torch.randn(self.max_patches, self.hidden_dim) * .02)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * .02)
        qconfig = BertConfig(
            hidden_size=self.hidden_dim, num_attention_heads=self.num_heads,
            intermediate_size=self.hidden_dim * 4, num_hidden_layers=self.encoder_layers,
            add_cross_attention=True, is_decoder=True, vocab_size=1,
            max_position_embeddings=max(64, self.num_tokens),
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
            use_cache=False,
        )
        self.query_encoder = BertModel(qconfig, add_pooling_layer=False)
        self.query_output = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                          nn.LayerNorm(self.hidden_dim, eps=self.norm_eps))
        self.current_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.action_projection = nn.Linear(self.action_dim, self.hidden_dim)
        self.action_position = nn.Parameter(torch.randn(self.max_action_steps, self.hidden_dim) * .02)
        self.time_projection = nn.Sequential(nn.Linear(3, self.hidden_dim), nn.SiLU(),
                                             nn.Linear(self.hidden_dim, self.hidden_dim))
        layer = nn.TransformerDecoderLayer(
            self.hidden_dim, self.num_heads, self.hidden_dim * 4,
            dropout=0., activation="gelu", batch_first=True, norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(layer, self.action_layers,
                                                    norm=nn.LayerNorm(self.hidden_dim))
        self.action_output = nn.Linear(self.hidden_dim, self.action_dim)

    def _features(self, value, ndim, label):
        if not torch.is_tensor(value) or value.ndim != ndim or not value.is_floating_point():
            raise ValueError(f"{label} must be a floating {ndim}-D tensor")
        if min(value.shape) < 1 or value.shape[-1] != self.feature_dim:
            raise ValueError(f"{label} has invalid feature dimensions")
        if value.shape[-2] > self.max_patches or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} exceeds patch capacity or is nonfinite")
        if value.device != self.visual_projection.weight.device:
            raise ValueError(f"{label} must be on the teacher's device")
        # Cached backbone features are never trained or temporally normalized.
        value = F.layer_norm(value.detach().float(), (self.feature_dim,), eps=self.norm_eps)
        return value.to(dtype=self.visual_projection.weight.dtype)

    def encode_future(self, future_features):
        """Only future RGB features enter this function; no action inputs exist."""
        future = self._features(future_features, 4, "future_features")
        batch, frames, patches, _ = future.shape
        if frames > self.max_future_steps:
            raise ValueError("Future horizon exceeds max_future_steps")
        memory = self.visual_projection(future)
        memory = memory + self.future_time[None, :frames, None] + self.spatial_position[None, None, :patches]
        queries = self.query_tokens.expand(batch, -1, -1)
        hidden = self.query_encoder(inputs_embeds=queries,
                                    encoder_hidden_states=memory.flatten(1, 2),
                                    return_dict=True).last_hidden_state
        return self.query_output(hidden)

    def action_loss(self, current_features, future_features, actions, actions_mask,
                    *, noise=None, flow_time=None):
        """Flow matching with inactive action channels excluded from input/loss.

        ``actions`` and ``actions_mask`` are [B,S,A]. Deterministic noise/time
        injection is provided for equivalence and masking tests, not inference.
        """
        current = self._features(current_features, 3, "current_features")
        future_queries = self.encode_future(future_features)
        if (not torch.is_tensor(actions) or actions.ndim != 3
                or actions.shape[-1] != self.action_dim or actions.shape[0] != current.shape[0]
                or actions.shape[0] != future_queries.shape[0]
                or not 1 <= actions.shape[1] <= self.max_action_steps):
            raise ValueError("Actions must have shape [B,S,action_dim] within horizon capacity")
        if (not torch.is_tensor(actions_mask) or actions_mask.shape != actions.shape
                or not actions.is_floating_point() or not bool(torch.isfinite(actions).all())):
            raise ValueError("Action mask shape mismatch or nonfinite actions")
        if actions.device != current.device:
            raise ValueError("Actions must be on the teacher's device")
        mask = actions_mask.to(device=actions.device, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("Action chunk contains no valid action targets")
        actions = actions.detach().float()
        noise = torch.randn_like(actions) if noise is None else noise.detach().float()
        if noise.shape != actions.shape or noise.device != actions.device or not bool(torch.isfinite(noise).all()):
            raise ValueError("Invalid flow noise")
        batch, steps, _ = actions.shape
        if flow_time is None:
            flow_time = torch.rand(batch, device=actions.device)
        if flow_time.shape != (batch,) or flow_time.device != actions.device or not bool(torch.isfinite(flow_time).all()) or bool(((flow_time < 0) | (flow_time > 1)).any()):
            raise ValueError("Flow time must have shape [B] in [0,1]")
        time = flow_time.float()[:, None, None]
        noisy = (time * actions + (1 - time) * noise).masked_fill(~mask, 0)
        target = actions - noise
        token = self.action_projection(noisy.to(self.action_projection.weight.dtype))
        tfeatures = torch.stack([flow_time.float(), torch.sin(flow_time.float() * math.pi),
                                 torch.cos(flow_time.float() * math.pi)], dim=-1)
        token = token + self.action_position[None, :steps] + self.time_projection(tfeatures.to(token.dtype))[:, None]
        current_memory = self.current_projection(current) + self.spatial_position[None, :current.shape[1]]
        memory = torch.cat([current_memory, future_queries], dim=1)
        prediction = self.action_output(self.action_decoder(token, memory)).float()
        return (prediction - target).masked_select(mask).square().mean()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_future_teacher(model, path, *, feature_spec, successful_steps, smoke_test,
                        training=None):
    """Write a complete, checksummed teacher; no optimizer state is advertised."""
    from safetensors.torch import save_file

    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite future teacher artifact: {path}")
    if isinstance(successful_steps, bool) or not isinstance(successful_steps, int) or successful_steps < 1:
        raise ValueError("A teacher needs at least one successful optimizer step")
    _validate_feature_spec(feature_spec)
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    if not all(bool(torch.isfinite(t).all()) for t in state.values()):
        raise ValueError("Cannot save nonfinite teacher parameters")
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "teacher.safetensors"
    save_file(state, str(weights) + ".tmp")
    os.replace(str(weights) + ".tmp", weights)
    metadata = dict(schema_version=TEACHER_SCHEMA, kind="rgb_action_future_teacher",
                    model_config=model.config, feature_spec=feature_spec,
                    successful_steps=successful_steps, smoke_test=bool(smoke_test),
                    training=dict(training or {}), source=WOG_SOURCE,
                    weights_sha256=_sha256(weights), checkpoint_type="weights_only")
    payload = path / "metadata.json"
    payload.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (path / "COMPLETE.json").write_text(json.dumps({
        "weights_sha256": metadata["weights_sha256"], "metadata_sha256": _sha256(payload)
    }, indent=2) + "\n")
    return metadata


def _validate_feature_spec(spec):
    required = {"rgb_encoder_id", "rgb_encoder_revision", "rgb_preprocessing"}
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError("Teacher feature provenance is incomplete")
    if not spec["rgb_encoder_id"] or not spec["rgb_encoder_revision"] or not isinstance(spec["rgb_preprocessing"], dict):
        raise ValueError("Teacher feature provenance is invalid")


def load_future_teacher(path, device="cpu", dtype=torch.float32, *, allow_smoke=False,
                        expected_feature_spec=None):
    """Load a frozen verified teacher; production rejects smoke-only artifacts."""
    from safetensors.torch import load_file

    path = Path(path)
    for filename in ("teacher.safetensors", "metadata.json", "COMPLETE.json"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Incomplete future teacher artifact: {path / filename}")
    metadata = json.loads((path / "metadata.json").read_text())
    complete = json.loads((path / "COMPLETE.json").read_text())
    if metadata.get("schema_version") != TEACHER_SCHEMA or metadata.get("kind") != "rgb_action_future_teacher":
        raise ValueError("Unsupported future teacher artifact")
    steps = metadata.get("successful_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Future teacher has no successful training steps")
    if not isinstance(metadata.get("smoke_test"), bool):
        raise ValueError("Future teacher is missing its smoke-test status")
    if metadata["smoke_test"] and not allow_smoke:
        raise ValueError("Smoke-only future teacher cannot initialize formal policy training")
    _validate_feature_spec(metadata.get("feature_spec"))
    if expected_feature_spec is not None and metadata["feature_spec"] != expected_feature_spec:
        raise ValueError("Future teacher RGB provenance differs from policy features")
    actual = _sha256(path / "teacher.safetensors")
    if actual != metadata.get("weights_sha256") or actual != complete.get("weights_sha256") or _sha256(path / "metadata.json") != complete.get("metadata_sha256"):
        raise ValueError("Future teacher checksum mismatch")
    model = ActionFutureTeacher(metadata["model_config"])
    model.load_state_dict(load_file(str(path / "teacher.safetensors"), device="cpu"), strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Future teacher contains nonfinite weights")
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model, metadata
