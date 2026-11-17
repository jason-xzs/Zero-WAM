"""Role supervision changes learning, never the information available to act."""
from copy import deepcopy

import pytest
import torch

from wan_va.modules.rgb_transition import RGBTransitionConditioner, build_rgb_transition_conditioner
from wan_va.modules.role_effect import ROLE_LOSS_KEYS, RoleEffectConditioner


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(825)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    result = dict(feature_dim=8, hidden_dim=12, num_transition_tokens=2,
                  num_heads=3, encoder_layers=1, decoder_layers=1, max_clips=3,
                  max_future_steps=5, max_patches=16, readout_mode="state",
                  readout_num_tokens=4, role_num_robot_views=1,
                  feature_normalization="none", role_effect_enabled=True)
    result.update(overrides)
    return result


def data():
    human, robot = torch.randn(2, 2, 3, 4, 8), torch.randn(2, 4, 4, 8)
    targets = {"human_masks": torch.rand(2, 2, 3, 2, 4),
               "human_valid": torch.ones(2, 2, 3, 2, dtype=torch.bool),
               "robot_masks": torch.rand(2, 4, 2, 4),
               "robot_valid": torch.ones(2, 4, 2, dtype=torch.bool)}
    state = torch.randn(2, 4, 18)
    positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    return human, robot, targets, state, positions


def test_factory_is_opt_in_and_zero_gate_preserves_hidden():
    assert type(build_rgb_transition_conditioner(config(role_effect_enabled=False), 18)) is RGBTransitionConditioner
    module = build_rgb_transition_conditioner(config(), 18)
    assert isinstance(module, RoleEffectConditioner)
    human, _, _, state, positions = data()
    memory, _ = module(human)
    tokens = module.read_tokens(memory, state, positions)
    assert memory.shape == (2, 12, 12) and tokens.shape == (2, 4, 12)
    hidden = torch.randn(2, 7, 18)
    torch.testing.assert_close(module.fuse(hidden, tokens), hidden, rtol=0, atol=0)


def test_supervision_future_and_donor_cannot_change_policy_conditioning():
    module = RoleEffectConditioner(config(), 18)
    human, robot, targets, state, positions = data()
    memory, losses, ctx = module(human, robot, True, role_targets=targets,
        donor_human_features=human * 2, donor_valid=torch.ones(2, dtype=torch.bool), return_aux_context=True)
    changed = deepcopy(targets)
    changed["human_masks"] = 1 - targets["human_masks"]
    changed["robot_masks"] = 1 - targets["robot_masks"]
    other_robot = robot.clone()
    other_robot[:, 1:] += torch.randn_like(other_robot[:, 1:]) * 4
    other, other_losses, other_ctx = module(human, other_robot, True, role_targets=changed,
        donor_human_features=-human, donor_valid=torch.ones(2, dtype=torch.bool), return_aux_context=True)
    inference_memory, empty = module(human)
    assert empty == {}
    for actual in (other, inference_memory):
        torch.testing.assert_close(memory, actual, rtol=0, atol=0)
    first, first_losses = module.training_read_tokens(memory, state, positions,
        aux_context=ctx, observed_index=0, future_indices=[1, 2])
    second, second_losses = module.training_read_tokens(other, state, positions,
        aux_context=other_ctx, observed_index=0, future_indices=[1, 2])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, module.read_tokens(inference_memory, state, positions), rtol=0, atol=0)
    assert not torch.equal(losses["role_human"], other_losses["role_human"])
    assert not torch.equal(first_losses["object_future"], second_losses["object_future"])


def test_deployed_binder_uses_current_observation_and_has_gradients():
    module = RoleEffectConditioner(config(), 18)
    human, _, _, state, positions = data()
    memory, _ = module(human)
    first = module.read_tokens(memory, state, positions)
    second = module.read_tokens(memory, -state, positions)
    assert not torch.equal(first, second)
    (first * torch.randn_like(first)).sum().backward()
    for parameter in (module.robot_query_projection.weight, module.robot_binding_projection[0].weight,
                      module.effect_projection.weight, module.binding_projection[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_donor_loss_uses_only_donor_effect_slots_and_trains_their_encoder():
    module = RoleEffectConditioner(config(), 18)
    human, robot, targets, state, positions = data()
    memory, _, ctx = module(human, robot, True, role_targets=targets,
        donor_human_features=human * 2, donor_valid=torch.ones(2, dtype=torch.bool), return_aux_context=True)
    ctx["donor_memory"].retain_grad()
    _, losses = module.training_read_tokens(memory, state, positions,
        aux_context=ctx, observed_index=0, future_indices=[1, 2])
    losses["donor_effect"].backward()
    gradient = ctx["donor_memory"].grad.reshape(2, 2, 6, 12)
    assert torch.count_nonzero(gradient[:, :, :-2]) == 0
    assert torch.count_nonzero(gradient[:, :, -2:]) > 0
    assert module.effect_projection.weight.grad.abs().sum() > 0
    # Exchanging donor identity/base slots cannot alter the effect decoder.
    memory_copy = ctx["donor_memory"].detach().clone().reshape(2, 2, 6, 12)
    memory_copy[:, :, :-2] = torch.randn_like(memory_copy[:, :, :-2]) * 100
    grounded = module._bind(memory.detach(), state, positions)[1]
    offsets = torch.tensor([0, 1])
    first = module._decode_effect(grounded, ctx["donor_memory"].detach(), offsets)
    second = module._decode_effect(grounded, memory_copy.flatten(1, 2), offsets)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("future_indices", [[], [1]])
def test_missing_and_invalid_roles_have_zero_auxiliary_losses_connected_to_all_parameters(future_indices):
    module = RoleEffectConditioner(config(), 18)
    human, robot, targets, state, positions = data()
    targets["human_valid"].zero_(); targets["robot_valid"].zero_()
    memory, losses, ctx = module(human, robot, True, role_targets=targets,
                                 donor_human_features=human, donor_valid=torch.zeros(2, dtype=torch.bool),
                                 return_aux_context=True)
    _, chunk = module.training_read_tokens(memory, state, positions, aux_context=ctx,
                                           observed_index=0, future_indices=future_indices)
    assert losses["role_human"].item() == 0
    assert all(value.item() == 0 for value in chunk.values())
    (losses["role_human"] + sum(chunk.values())).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    memory, _, ctx = module(human, robot, True, return_aux_context=True)
    _, chunk = module.training_read_tokens(memory, state, positions, aux_context=ctx,
                                           observed_index=0, future_indices=future_indices)
    assert all(value.item() == 0 for value in chunk.values())


def test_camera_supervision_selects_only_configured_view_in_raster_order():
    module = RoleEffectConditioner(config(role_num_robot_views=3), 18)
    positions = torch.tensor([[1, 5], [0, 1], [0, 0], [1, 0], [1, 1], [0, 2],
                              [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4]])
    indices, shape = module._teacher_view(positions)
    assert shape == (2, 2)
    assert positions[indices].tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_role_binder_ignores_unsupervised_views_but_full_policy_still_reads_them():
    module = RoleEffectConditioner(config(role_num_robot_views=3), 18)
    human, _, _, _, _ = data()
    memory, _ = module(human)
    positions = torch.tensor([[r, c] for r in range(2) for c in range(6)])
    state = torch.randn(2, 12, 18)
    changed = state.clone()
    changed[:, positions[:, 1] >= 2] += torch.randn_like(changed[:, positions[:, 1] >= 2]) * 20
    policy, grounded, logits = module._bind(memory, state, positions)
    other_policy, other_grounded, other_logits = module._bind(memory, changed, positions)
    torch.testing.assert_close(grounded, other_grounded, rtol=0, atol=0)
    torch.testing.assert_close(logits, other_logits, rtol=0, atol=0)
    assert not torch.equal(policy, other_policy)


def test_entity_changes_mask_false_cross_object_deltas_without_changing_policy():
    module = RoleEffectConditioner(config(), 18)
    human, robot, targets, state, positions = data()
    own, _, own_context = module(human, robot, True, role_targets=targets, return_aux_context=True)
    targets["robot_entity_ids"] = torch.tensor([0, 1]).expand(2, 4, 2).clone()
    targets["robot_entity_ids"][:, 1] = torch.tensor([1, 0])
    targets["robot_entity_ids"][:, 2] = -1
    switched, _, switched_context = module(human, robot, True, role_targets=targets, return_aux_context=True)
    torch.testing.assert_close(own, switched, rtol=0, atol=0)
    first, first_losses = module.training_read_tokens(own, state, positions,
        aux_context=own_context, observed_index=0, future_indices=[1, 2])
    second, second_losses = module.training_read_tokens(switched, state, positions,
        aux_context=switched_context, observed_index=0, future_indices=[1, 2])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first_losses["object_future"] > 0
    assert second_losses["object_future"].item() == 0
    assert second_losses["relation_future"].item() == 0
    assert second_losses["role_robot"] > 0


def test_invalid_role_and_donor_contracts_fail_explicitly():
    module = RoleEffectConditioner(config(), 18)
    human, robot, targets, state, positions = data()
    with pytest.raises(ValueError, match="explicit donor_valid"):
        module(human, robot, True, donor_human_features=human)
    with pytest.raises(ValueError, match="boolean"):
        module(human, robot, True, donor_valid=torch.ones(2))
    with pytest.raises(ValueError, match="requires donor_human_features"):
        module(human, robot, True, donor_valid=torch.ones(2, dtype=torch.bool))
    targets["human_masks"].zero_()
    with pytest.raises(ValueError, match="nonempty"):
        module(human, robot, True, role_targets=targets)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_and_training_are_finite(dtype):
    module = RoleEffectConditioner(config(), 18).to(dtype=dtype)
    human, robot, targets, state, positions = data()
    human, robot, state = human.to(dtype), robot.to(dtype), state.to(dtype)
    memory, losses, ctx = module(human, robot, True, role_targets=targets, return_aux_context=True)
    tokens, chunk = module.training_read_tokens(memory, state, positions, aux_context=ctx,
                                               observed_index=0, future_indices=[1, 2])
    assert torch.isfinite(tokens).all() and tokens.dtype == dtype
    (losses["total"] + sum(chunk.values()) + tokens.float().square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    restored = RoleEffectConditioner(config(), 18).to(dtype=dtype)
    restored.load_state_dict(module.state_dict(), strict=True)
    restored_memory, _ = restored(human)
    torch.testing.assert_close(restored.read_tokens(restored_memory, state, positions), tokens, rtol=0, atol=0)
