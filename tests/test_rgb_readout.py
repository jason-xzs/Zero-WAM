"""Behavioral contracts for matched demonstration-readout experiments."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from wan_va.modules.rgb_readout import DemoReadout
from wan_va.modules.rgb_transition import RGBTransitionConditioner
from wan_va.modules.third_party.open_flamingo import PerceiverAttention


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(71)
    yield
    torch.set_num_threads(old)


def readout(mode="static", **kwargs):
    arguments = dict(hidden_dim=12, state_dim=18, num_heads=3,
                     num_tokens=6, global_tokens=2, mode=mode)
    arguments.update(kwargs)
    return DemoReadout(**arguments)


def inputs():
    return (torch.randn(2, 8, 12), torch.randn(2, 4, 18),
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))


def conditioner_config(**overrides):
    config = dict(feature_dim=8, hidden_dim=12, num_heads=3,
                  num_transition_tokens=2, encoder_layers=1, decoder_layers=1,
                  max_clips=3, max_future_steps=4, max_patches=8,
                  readout_num_tokens=6, readout_global_tokens=2)
    config.update(overrides)
    return config


def test_attention_port_matches_independent_sdpa_forward_and_gradients():
    """The upstream einsum attention and torch SDPA implement the same operation."""
    module = PerceiverAttention(dim=12, dim_head=4, heads=3)
    reference = deepcopy(module)
    media = torch.randn(2, 1, 7, 12, requires_grad=True)
    queries = torch.randn(2, 1, 6, 12, requires_grad=True)
    media_ref = media.detach().clone().requires_grad_(True)
    queries_ref = queries.detach().clone().requires_grad_(True)
    actual = module(media, queries)

    x, q = reference.norm_media(media_ref), reference.norm_latents(queries_ref)
    query = reference.to_q(q)
    key, value = reference.to_kv(torch.cat([x, q], dim=-2)).chunk(2, dim=-1)
    def heads(tensor):
        return tensor.reshape(2, 1, -1, 3, 4).permute(0, 3, 1, 2, 4)
    result = F.scaled_dot_product_attention(heads(query), heads(key), heads(value))
    result = result.permute(0, 2, 3, 1, 4).reshape(2, 1, 6, 12)
    expected = reference.to_out(result)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(media.grad, media_ref.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(queries.grad, queries_ref.grad, rtol=2e-5, atol=2e-6)
    for (name, value), (ref_name, ref_value) in zip(
            module.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        torch.testing.assert_close(value.grad, ref_value.grad, rtol=2e-5, atol=2e-6)


def test_modes_with_equal_token_budget_and_seed_have_identical_initialization():
    modules = []
    for mode in ("static", "state", "global_local"):
        torch.manual_seed(404)
        modules.append(readout(mode))
    states = [module.state_dict() for module in modules]
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for name in states[0]:
        for state in states[1:]:
            torch.testing.assert_close(states[0][name], state[name], atol=0, rtol=0)
    assert len({sum(p.numel() for p in m.parameters()) for m in modules}) == 1


def test_same_parameter_schema_does_not_imply_same_initialization_across_seeds():
    torch.manual_seed(404)
    first = readout("static")
    torch.manual_seed(405)
    second = readout("state")
    left, right = first.state_dict(), second.state_dict()
    assert {name: value.shape for name, value in left.items()} == {
        name: value.shape for name, value in right.items()}
    assert not torch.equal(left["queries"], right["queries"])
    assert not torch.equal(left["state_projection.weight"], right["state_projection.weight"])


def test_compressed_16_token_budget_has_different_query_capacity_from_32():
    torch.manual_seed(404)
    reference = readout("state", num_tokens=32, global_tokens=8)
    torch.manual_seed(404)
    compressed = readout("global_local", num_tokens=16, global_tokens=4)
    original, smaller = reference.state_dict(), compressed.state_dict()
    assert original.keys() == smaller.keys()
    assert [name for name in original if original[name].shape != smaller[name].shape] == ["queries"]
    assert sum(p.numel() for p in reference.parameters()) - sum(
        p.numel() for p in compressed.parameters()) == 16 * reference.hidden_dim
    # Equal seeds alone do not make differently shaped checkpoints equal, and
    # strict loading must reject silently reusing the wrong token budget.
    with pytest.raises(RuntimeError, match="size mismatch for queries"):
        compressed.load_state_dict(original, strict=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_human_32_tokens_compress_to_4_global_and_12_local_with_state_isolation(dtype):
    module = RGBTransitionConditioner(
        conditioner_config(readout_mode="global_local", max_clips=8,
                           num_transition_tokens=4,
                           readout_num_tokens=16, readout_global_tokens=4),
        video_dim=18,
    ).to(dtype=dtype)
    # Exercise the actual eight-clip human encoder, not an already fabricated
    # sixteen-token input: the new readout must halve the 8 x 4 memory length.
    human_features = torch.randn(1, 8, 3, 4, 8, dtype=dtype)
    memory, _ = module(human_features)
    assert memory.shape == (1, 32, 12)
    _, state, positions = inputs()
    state = state[:1].to(dtype=dtype)
    result = module.read_tokens(memory, state, positions)
    changed = module.read_tokens(memory, -state * 3, positions + 8)
    assert result.shape == (1, 16, 12) and result.dtype == dtype
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[:, :4], changed[:, :4], atol=0, rtol=0)
    assert not torch.equal(result[:, 4:], changed[:, 4:])
    (result.float() * torch.randn_like(result.float())).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.readout.parameters())


def test_static_is_exactly_independent_of_observation_and_positions():
    module = readout("static")
    human, state, positions = inputs()
    first = module(human, state, positions)
    second = module(human, -state * 3, positions + 8)
    assert first.shape == (2, 6, 12)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, module(human + torch.randn_like(human), state, positions))


def test_state_readout_changes_with_observation_and_stays_differentiable():
    module = readout("state")
    human, state, positions = inputs()
    human.requires_grad_(True)
    state.requires_grad_(True)
    actual = module(human, state, positions)
    assert not torch.equal(actual, module(human, -state, positions))
    (actual * torch.randn_like(actual)).sum().backward()
    for value in (human, state):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_global_queries_do_not_receive_state_through_local_latent_self_kv():
    module = readout("global_local", num_layers=2)
    human, state, positions = inputs()
    first = module(human, state, positions)
    second = module(human, -state * 3, positions + 8)
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=0, rtol=0)
    assert not torch.equal(first[:, 2:], second[:, 2:])
    changed_human = module(human + torch.randn_like(human), state, positions)
    assert not torch.equal(first[:, :2], changed_human[:, :2])


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_all_parameter_gradients_are_present_in_every_mode(mode):
    module = readout(mode)
    actual = module(*inputs())
    (actual * torch.randn_like(actual)).sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if mode == "static" and name.startswith("state_"):
            assert torch.count_nonzero(parameter.grad) == 0, name


def test_state_spatial_positions_distinguish_relocated_observation_content():
    module = readout("state")
    human, state, positions = inputs()
    baseline = module(human, state, positions)
    # Moving both content and its coordinate is a harmless storage permutation.
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = module(human, state[:, permutation], positions[permutation])
    torch.testing.assert_close(baseline, permuted, atol=1e-6, rtol=1e-6)
    # Moving just the visual content changes spatial layout and should matter.
    relocated = module(human, state[:, permutation], positions)
    assert not torch.allclose(baseline, relocated, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_bfloat16_readout_and_gradients_are_finite(mode):
    module = readout(mode).to(dtype=torch.bfloat16)
    human, state, positions = inputs()
    result = module(human.bfloat16(), state.bfloat16(), positions)
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    (result.float() * torch.randn_like(result.float())).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())


def test_legacy_conditioner_parameters_are_unchanged_by_optional_readout():
    torch.manual_seed(501)
    legacy = RGBTransitionConditioner(conditioner_config(), video_dim=18)
    torch.manual_seed(501)
    current = RGBTransitionConditioner(
        conditioner_config(readout_mode="state"), video_dim=18
    )
    assert legacy.readout_mode == "human" and legacy.readout is None
    for name, tensor in legacy.state_dict().items():
        torch.testing.assert_close(tensor, current.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="requires a robot-side"):
        legacy.read_tokens(*inputs()[:2])


def test_conditioner_keeps_human_order_and_gate_zero_preserves_robot_hidden():
    module = RGBTransitionConditioner(
        conditioner_config(readout_mode="state"), video_dim=18
    )
    human_features = torch.randn(2, 2, 3, 4, 8)
    human, _ = module(human_features)
    reordered_human, _ = module(human_features.flip(1))
    _, state, positions = inputs()
    tokens = module.read_tokens(human, state, positions)
    reordered_tokens = module.read_tokens(reordered_human, state, positions)
    assert not torch.equal(tokens, reordered_tokens)
    robot_hidden = torch.randn(2, 9, 18)
    actual = module.fuse(robot_hidden, tokens)
    torch.testing.assert_close(actual, robot_hidden, atol=0, rtol=0)
    actual.square().mean().backward()
    assert module.gate.grad is not None and module.gate.grad.abs().item() > 0
    assert all(p.grad is not None for p in module.readout.parameters())


@pytest.mark.parametrize("kwargs,match", [
    ({"mode": "invalid"}, "Unsupported readout"),
    ({"num_tokens": 0}, "positive integer"),
    ({"num_tokens": True}, "positive integer"),
    ({"global_tokens": -1}, "0..num_tokens"),
    ({"global_tokens": 7}, "0..num_tokens"),
    ({"global_tokens": True}, "integer"),
    ({"mode": "global_local", "global_tokens": 0}, "both global and local"),
    ({"mode": "global_local", "global_tokens": 6}, "both global and local"),
    ({"hidden_dim": 13}, "divisible"),
    ({"num_layers": 0}, "positive integer"),
    ({"norm_eps": float("nan")}, "finite and positive"),
])
def test_invalid_readout_configs_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        readout(**kwargs)


def test_invalid_feature_or_position_inputs_fail():
    module = readout()
    human, state, positions = inputs()
    with pytest.raises(ValueError, match="batch mismatch"):
        module(human[:1], state, positions)
    with pytest.raises(ValueError, match="invalid shape"):
        module(human, state[:, :, :-1], positions)
    with pytest.raises(ValueError, match="non-finite"):
        module(human * float("nan"), state, positions)
    with pytest.raises(TypeError, match="floating-point"):
        module(human.long(), state, positions)
    with pytest.raises(ValueError, match="state_positions must have shape"):
        module(human, state, positions[:-1])
    with pytest.raises(ValueError, match="non-finite coordinates"):
        module(human, state, positions.float() * float("nan"))
    with pytest.raises(TypeError, match="real coordinates"):
        module(human, state, positions.bool())
    # Both permitted position shapes have the same result.
    torch.testing.assert_close(
        module(human, state, positions),
        module(human, state, positions[None].expand(2, -1, -1)), atol=0, rtol=0,
    )


def test_unknown_readout_mode_is_rejected_by_conditioner():
    with pytest.raises(ValueError, match="Unsupported readout_mode"):
        RGBTransitionConditioner(conditioner_config(readout_mode="invalid"), video_dim=18)
