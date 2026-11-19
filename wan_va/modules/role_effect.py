"""Supervised role binding and effect memory for the RGB readout policy.

Masks are detached training targets, never policy inputs. Human predictions are
computed from the complete demonstration. Robot bindings are computed from the
currently observed Wan patch embeddings and are used by the deployed readout.
The geometric auxiliary target is image-plane geometry, not metric 3-D state.
"""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .rgb_transition import RGBTransitionConditioner


ROLE_LOSS_KEYS = (
    "role_human", "role_robot", "object_future", "relation_future", "donor_effect",
)


class RoleEffectConditioner(RGBTransitionConditioner):
    """Role-supervised extension with an explicitly observed-state binder.

Memory consists of K original WALA tokens, two binding tokens and two effect
tokens per human clip. It remains a tensor so the existing human KV lifecycle
can cache it. The original WALA loss remains independently configurable.

``role_targets`` contains human_masks [B,S,T,2,P], human_valid [B,S,T,2],
robot_masks [B,Tr,2,P], robot_valid [B,Tr,2] and optional robot_entity_ids
[B,Tr,2] (int64; -1 means unknown). Different identities cannot form a delta.
All are optional together;
missing targets contribute graph-connected zero losses. Data-manifest
completeness and donor semantic eligibility are validated by the dataset.
"""

    role_effect_enabled = True

    def __init__(self, config, video_dim=3072):
        config = dict(config)
        config.setdefault("readout_mode", "state")
        if config["readout_mode"] != "state":
            raise ValueError("RoleEffectConditioner requires readout_mode='state'")
        super().__init__(config, video_dim=video_dim)
        h, d = self.hidden_dim, self.feature_dim
        self.role_loss_weights = {
            key: self._weight(config, f"{key}_weight", 1.0)
            for key in ROLE_LOSS_KEYS
        }
        self.role_num_robot_views = self._positive_int(config, "role_num_robot_views", 3)
        self.role_teacher_view = int(config.get("role_teacher_view", 0))
        if not 0 <= self.role_teacher_view < self.role_num_robot_views:
            raise ValueError("role_teacher_view is outside the configured camera layout")
        self.role_patch_projection = nn.Linear(d, h)
        self.role_delta_projection = nn.Linear(d, h, bias=False)
        self.role_context_projection = nn.Linear(h, h)
        self.human_role_queries = nn.Parameter(torch.randn(2, h) * 0.02)
        self.binding_projection = nn.Sequential(nn.Linear(d + 4, h), nn.SiLU(), nn.Linear(h, h))
        self.binding_type = nn.Parameter(torch.randn(2, h) * 0.02)
        self.effect_projection = nn.Linear(2 * d + 10, h)
        self.effect_queries = nn.Parameter(torch.randn(2, h) * 0.02)
        self.effect_attention = nn.MultiheadAttention(h, self.num_heads, batch_first=True)
        self.effect_norm = nn.LayerNorm(h, eps=self.norm_eps)
        self.robot_state_norm = nn.LayerNorm(video_dim, eps=self.norm_eps)
        self.robot_state_projection = nn.Linear(video_dim, h)
        self.robot_role_queries = nn.Parameter(torch.randn(2, h) * 0.02)
        self.robot_query_projection = nn.Linear(2 * h, h)
        self.robot_binding_projection = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.bound_effect_attention = nn.MultiheadAttention(h, self.num_heads, batch_first=True)
        self.readout_binding_projection = nn.Linear(h, h)
        self.readout_output_norm = nn.LayerNorm(h, eps=self.norm_eps)
        self.object_horizon = nn.Parameter(torch.randn(self.max_future_steps, h) * 0.02)
        self.object_decoder = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
        self.relation_decoder = nn.Sequential(nn.LayerNorm(2 * h), nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 2))
        # Old smoke checkpoints did not encode the order of the future frames.
        # Preserve their exact layout/behavior when the key is absent. New
        # training explicitly enables this and saves it in the model config.
        self.role_effect_temporal_encoding = config.get("role_effect_temporal_encoding", "none")
        if self.role_effect_temporal_encoding not in {"none", "learned"}:
            raise ValueError("role_effect_temporal_encoding must be 'none' or 'learned'")
        if self.role_effect_temporal_encoding == "learned":
            self.effect_time_embedding = nn.Parameter(torch.randn(self.max_future_steps, h) * 0.02)

    def _zero(self):
        # Every rank must retain all parameters even when its labels are absent.
        return sum(p.reshape(-1)[0].float() * 0 for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _square_grid(patches, device, dtype):
        width = math.isqrt(patches)
        if width * width != patches:
            raise ValueError("Role DINO features must use a square patch grid")
        axis = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx.flatten(), yy.flatten()), -1)

    @staticmethod
    def _pool(features, masks, coordinates):
        # features [...,P,D], masks [...,R,P]; geometries [...,R,4].
        weights = masks.float() / masks.float().sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("...rp,...pd->...rd", weights, features.float())
        center = torch.einsum("...rp,pc->...rc", weights, coordinates.float())
        square = torch.einsum("...rp,pc->...rc", weights, coordinates.float().square())
        spread = (square - center.square()).clamp_min(1e-8).sqrt()
        return pooled.to(features.dtype), torch.cat((center, spread), -1).to(features.dtype)

    def _human_role_memory(self, human):
        b, s, t, p, d = human.shape
        xy = self._square_grid(p, human.device, human.dtype)
        patch = self.role_patch_projection(human)
        patch = patch + self.role_delta_projection(human - human[:, :, :1])
        # DINO already supplies spatial context; the role query also sees the
        # complete clip, enabling static references to depend on the interaction.
        context = self.role_context_projection(patch.mean((2, 3)))
        query = self.human_role_queries[None, None] + context[:, :, None]
        logits = torch.einsum("bsrh,bstph->bstrp", query, patch) / math.sqrt(self.hidden_dim)
        pooled, geometry = self._pool(human, logits.sigmoid(), xy)
        binding = self.binding_projection(torch.cat((pooled[:, :, 0], geometry[:, :, 0]), -1))
        binding = binding + self.binding_type[None, None]
        # The effect branch receives changes, not the initial appearance token.
        delta_features = (pooled[:, :, 1:] - pooled[:, :, :1]).flatten(-2)
        delta_geometry = (geometry[:, :, 1:] - geometry[:, :, :1]).flatten(-2)
        relation = geometry[..., 0, :2] - geometry[..., 1, :2]
        delta_relation = relation[:, :, 1:] - relation[:, :, :1]
        effect_input = self.effect_projection(torch.cat((delta_features, delta_geometry, delta_relation), -1))
        effect_input = effect_input.reshape(b * s, t - 1, self.hidden_dim)
        if self.role_effect_temporal_encoding == "learned":
            effect_input = effect_input + self.effect_time_embedding[None, :t - 1].to(effect_input.dtype)
        queries = self.effect_queries[None].expand(b * s, -1, -1)
        effect, _ = self.effect_attention(queries, effect_input, effect_input, need_weights=False)
        effect = self.effect_norm(queries + effect).reshape(b, s, 2, self.hidden_dim)
        return binding, effect, logits

    def _memory(self, human_features, *, compute_base_aux=False, robot_features=None):
        base, losses = super().forward(human_features, robot_features, compute_aux=compute_base_aux)
        human = self._prepare_features(human_features)
        b, s = human.shape[:2]
        binding, effect, logits = self._human_role_memory(human)
        order = self.clip_order_embedding[None, :s].to(binding.dtype)
        memory = torch.cat((base.reshape(b, s, self.num_transition_tokens, self.hidden_dim),
                            binding + order, effect + order), 2).flatten(1, 2)
        return memory, losses, logits

    def _split_memory(self, memory):
        unit = self.num_transition_tokens + 4
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_dim or memory.shape[1] % unit:
            raise ValueError("Invalid packed role/effect memory")
        packed = memory.reshape(memory.shape[0], -1, unit, self.hidden_dim)
        return packed[:, :, :self.num_transition_tokens], packed[:, :, -4:-2], packed[:, :, -2:]

    def _targets(self, targets, human_features, robot_features):
        if targets is None:
            return {}
        if not isinstance(targets, Mapping):
            raise TypeError("role_targets must be a mapping")
        result = {}
        for prefix, shape in (("human", (*human_features.shape[:3], 2, human_features.shape[-2])),
                              ("robot", None if robot_features is None else
                               (*robot_features.shape[:2], 2, robot_features.shape[-2]))):
            mask, valid = targets.get(prefix + "_masks"), targets.get(prefix + "_valid")
            if mask is None and valid is None:
                continue
            if shape is None or not torch.is_tensor(mask) or tuple(mask.shape) != shape:
                raise ValueError(f"{prefix}_masks has an invalid shape")
            if not torch.is_tensor(valid) or valid.dtype != torch.bool or tuple(valid.shape) != shape[:-1]:
                raise ValueError(f"{prefix}_valid must be a matching boolean tensor")
            mask = mask.detach().to(device=human_features.device, dtype=torch.float32)
            valid = valid.detach().to(device=human_features.device)
            if not torch.isfinite(mask).all() or bool(((mask < 0) | (mask > 1)).any()):
                raise ValueError(f"{prefix}_masks must be finite probabilities")
            if bool((valid & (mask.sum(-1) <= 0)).any()):
                raise ValueError(f"Valid {prefix} roles must have nonempty masks")
            result[prefix + "_masks"], result[prefix + "_valid"] = mask, valid
            if prefix == "robot":
                identities = targets.get("robot_entity_ids")
                if identities is None:
                    # Compatibility for explicitly stable-role module callers.
                    # The production loader requires IDs for changing roles.
                    identities = torch.arange(2, device=mask.device).expand(shape[:-1])
                if not torch.is_tensor(identities) or identities.dtype != torch.int64 or tuple(identities.shape) != shape[:-1]:
                    raise ValueError("robot_entity_ids must be matching int64 [B,Tr,2]")
                identities = identities.detach().to(mask.device)
                if bool((identities < -1).any()):
                    raise ValueError("robot_entity_ids must be nonnegative or -1 for unknown")
                result["robot_entity_ids"] = identities
        return result

    @staticmethod
    def _masked_mean(values, valid):
        valid = valid.to(values.dtype)
        return (values * valid).sum() / valid.sum().clamp_min(1)

    def _mask_loss(self, logits, target, valid):
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none").mean(-1)
        predicted = logits.float().sigmoid()
        dice = 1 - (2 * (predicted * target).sum(-1) + 1e-6) / (predicted.sum(-1) + target.sum(-1) + 1e-6)
        return self._masked_mean(bce + dice, valid)

    def forward(self, human_features, robot_features=None, compute_aux=False, *,
                role_targets=None, donor_human_features=None, donor_valid=None,
                return_aux_context=False):
        memory, losses, logits = self._memory(human_features, compute_base_aux=compute_aux,
                                             robot_features=robot_features)
        context = None
        if compute_aux:
            zero = self._zero()
            losses = dict(losses)
            losses.update({key: zero for key in ROLE_LOSS_KEYS})
            targets = self._targets(role_targets, human_features, robot_features)
            if "human_masks" in targets:
                losses["role_human"] = self._mask_loss(logits, targets["human_masks"], targets["human_valid"]) + zero
            losses["total"] = losses["total"] + self.role_loss_weights["role_human"] * losses["role_human"] + zero
            donor_memory = None
            if donor_valid is not None:
                if not torch.is_tensor(donor_valid) or donor_valid.dtype != torch.bool or tuple(donor_valid.shape) != (human_features.shape[0],):
                    raise ValueError("donor_valid must be boolean [B]")
                donor_valid = donor_valid.detach().to(human_features.device)
            if donor_human_features is not None:
                if donor_valid is None:
                    raise ValueError("Donor features require explicit donor_valid")
                donor_human_features = donor_human_features.to(human_features)
                if donor_human_features.shape[0] != human_features.shape[0]:
                    raise ValueError("Donor batch does not match recipient")
                if bool(donor_valid.any()):
                    donor_memory, _, _ = self._memory(donor_human_features)
            elif donor_valid is not None and bool(donor_valid.any()):
                raise ValueError("Valid donor requires donor_human_features")
            context = {"targets": targets,
                       "robot_features": None if robot_features is None else self._prepare_features(robot_features),
                       "donor_memory": donor_memory, "donor_valid": donor_valid}
        return (memory, losses, context) if return_aux_context else (memory, losses)

    def _bind(self, memory, state_features, state_positions):
        _, binding, effects = self._split_memory(memory)
        state = self.robot_state_projection(self.robot_state_norm(state_features.to(self.robot_state_norm.weight.dtype)))
        state = state + self.readout._state_positions(state, state_positions)
        # Only the supervised camera participates in role binding. Unlabelled
        # wrist-camera patches must not become an easy bypass around masks.
        # The original policy readout below still receives all camera views.
        selected, _ = self._teacher_view(state_positions)
        state = state[:, selected]
        queries = self.robot_role_queries[None] + state.mean(1, keepdim=True)
        scores = torch.einsum("brh,bsrh->brs", queries, binding) / math.sqrt(self.hidden_dim)
        chosen = torch.einsum("brs,bsrh->brh", scores.softmax(-1), binding)
        queries = self.robot_query_projection(torch.cat((chosen, queries), -1))
        logits = torch.einsum("brh,bph->brp", queries, state) / math.sqrt(self.hidden_dim)
        weights = logits.float().sigmoid()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("brp,bph->brh", weights.to(state.dtype), state)
        grounded = self.robot_binding_projection(torch.cat((chosen, pooled), -1))
        effect_memory = effects.flatten(1, 2)
        bound_effect, _ = self.bound_effect_attention(grounded, effect_memory, effect_memory, need_weights=False)
        tokens = self.readout(memory, state_features, state_positions)
        indices = torch.arange(tokens.shape[1], device=tokens.device) % 2
        tokens = self.readout_output_norm(tokens + self.readout_binding_projection(grounded + bound_effect)[:, indices])
        return tokens, grounded, logits

    def read_tokens(self, memory, state_features, state_positions=None):
        return self._bind(memory, state_features, state_positions)[0]

    def _teacher_view(self, positions):
        # The supported RoboTwin/Franka loader concatenates equally sized views
        # along width (lerobot_latent_dataset._cat_video_latents). Do not apply
        # this rule to the distinct robotwin_tshape layout.
        if positions is None:
            raise ValueError("Role supervision requires explicit camera grid positions")
        if positions.ndim == 3:
            if positions.shape[0] != 1:
                raise ValueError("Role camera grids must be shared across the batch")
            positions = positions[0]
        rows, columns = positions[:, 0].unique(sorted=True), positions[:, 1].unique(sorted=True)
        if len(columns) % self.role_num_robot_views:
            raise ValueError("Wan camera grid width is not divisible by role_num_robot_views")
        width = len(columns) // self.role_num_robot_views
        selected_columns = columns[self.role_teacher_view * width:(self.role_teacher_view + 1) * width]
        selected = ((positions[:, 1] >= selected_columns[0]) &
                    (positions[:, 1] <= selected_columns[-1])).nonzero().flatten()
        row_rank = torch.searchsorted(rows, positions[selected, 0].contiguous())
        column_rank = torch.searchsorted(selected_columns, positions[selected, 1].contiguous())
        raster_ids = row_rank * width + column_rank
        order = raster_ids.argsort()
        expected = torch.arange(len(rows) * width, device=positions.device)
        if not torch.equal(raster_ids[order], expected):
            raise ValueError("Role supervision requires a complete rectangular camera grid")
        return selected[order], (len(rows), width)

    def _decode_effect(self, grounded, effect_memory, offsets):
        effects = self._split_memory(effect_memory)[2].flatten(1, 2)
        effect, _ = self.bound_effect_attention(grounded, effects, effects, need_weights=False)
        hidden = grounded[:, None] + effect[:, None] + self.object_horizon[offsets][None, :, None]
        return self.object_decoder(hidden), self.relation_decoder(hidden.flatten(-2))

    def training_read_tokens(self, memory, state_features, state_positions, *,
                             aux_context, observed_index, future_indices):
        tokens, grounded, logits = self._bind(memory, state_features, state_positions)
        zero = self._zero()
        losses = {key: zero for key in ROLE_LOSS_KEYS if key != "role_human"}
        if not aux_context or aux_context["robot_features"] is None:
            return tokens, losses
        targets, robot = aux_context["targets"], aux_context["robot_features"]
        if "robot_masks" not in targets:
            return tokens, losses
        masks, valid = targets["robot_masks"], targets["robot_valid"]
        if not 0 <= observed_index < robot.shape[1]:
            raise ValueError("Observed role frame is outside the robot timeline")
        _, shape = self._teacher_view(state_positions)
        side = math.isqrt(masks.shape[-1])
        if side * side != masks.shape[-1]:
            raise ValueError("Robot role masks require a square DINO grid")
        teacher = F.interpolate(masks[:, observed_index].reshape(-1, 1, side, side),
                                size=shape, mode="area").reshape(masks.shape[0], 2, -1)
        losses["role_robot"] = self._mask_loss(logits, teacher, valid[:, observed_index])
        future = torch.as_tensor(future_indices, device=robot.device, dtype=torch.long)
        if future.numel() == 0:
            return tokens, {key: value + zero for key, value in losses.items()}
        if future.ndim != 1 or bool(((future <= observed_index) | (future >= robot.shape[1])).any()):
            raise ValueError("Auxiliary future indices must strictly follow the observed frame")
        offsets = future - observed_index - 1
        if int(offsets.max()) >= self.max_future_steps:
            raise ValueError("Object future horizon exceeds configured maximum")
        xy = self._square_grid(robot.shape[-2], robot.device, robot.dtype)
        object_features, geometry = self._pool(robot, masks, xy)
        target_delta = (object_features[:, future] - object_features[:, observed_index:observed_index + 1]).detach().float()
        object_valid = valid[:, future] & valid[:, observed_index:observed_index + 1]
        identities = targets["robot_entity_ids"]
        current_identities = identities[:, observed_index:observed_index + 1]
        object_valid = object_valid & (identities[:, future] == current_identities) & (current_identities >= 0)
        relation = geometry[..., 0, :2] - geometry[..., 1, :2]
        target_relation = (relation[:, future] - relation[:, observed_index:observed_index + 1]).detach().float()
        relation_valid = object_valid.all(-1)

        def effect_losses(effect_memory):
            pred, rel = self._decode_effect(grounded, effect_memory, offsets)
            object_error = F.smooth_l1_loss(pred.float(), target_delta, reduction="none").mean(-1)
            relation_error = F.smooth_l1_loss(rel.float(), target_relation, reduction="none").mean(-1)
            return object_error, relation_error

        obj, rel = effect_losses(memory)
        losses["object_future"] = self._masked_mean(obj, object_valid)
        losses["relation_future"] = self._masked_mean(rel, relation_valid)
        if aux_context["donor_memory"] is not None:
            donor_valid = aux_context["donor_valid"]
            obj, rel = effect_losses(aux_context["donor_memory"])
            losses["donor_effect"] = self._masked_mean(obj, object_valid & donor_valid[:, None, None])
            losses["donor_effect"] = losses["donor_effect"] + self._masked_mean(rel, relation_valid & donor_valid[:, None])
        return tokens, {key: value + zero for key, value in losses.items()}


__all__ = ["RoleEffectConditioner", "ROLE_LOSS_KEYS"]
