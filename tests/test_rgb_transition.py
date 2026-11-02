import io

import pytest
import torch
from torch.nn import functional as F

from wan_va.modules.rgb_transition import RGBTransitionConditioner
from wan_va.modules.third_party.wala import ContinuousTransitionBottleneckTokenizer


@pytest.fixture(autouse=True)
def _small_cpu_workload():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(407)
    yield
    torch.set_num_threads(original)


def _config(**overrides):
    config = dict(
        feature_dim=8,
        hidden_dim=12,
        num_transition_tokens=2,
        num_heads=3,
        encoder_layers=1,
        decoder_layers=1,
        max_clips=3,
        max_future_steps=4,
        max_patches=16,
        feature_normalization="none",
    )
    config.update(overrides)
    return config


def _inputs(requires_grad=False):
    human = torch.randn(2, 2, 3, 4, 8, requires_grad=requires_grad)
    robot = torch.randn(2, 4, 5, 8, requires_grad=requires_grad)
    return human, robot


def _assert_nonzero_finite_grad(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


def test_rgb_self_objective_matches_unmodified_wala_forward():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    reference = ContinuousTransitionBottleneckTokenizer(
        dino_dim=8,
        hidden_dim=12,
        num_transition_tokens=2,
        num_heads=3,
        encoder_layers=1,
        decoder_layers=1,
        max_future_steps=4,
        max_patches=16,
        use_depth=False,
    )
    reference.load_state_dict(model.tokenizer.state_dict(), strict=True)
    human, robot = _inputs()
    tokens, losses = model(human, robot, compute_aux=True)
    flat = human.flatten(0, 1)
    expected_human = reference(flat[:, 0], flat[:, 1:])
    expected_robot = reference(robot[:, 0], robot[:, 1:])
    expected_tokens = expected_human["transition_tokens"].reshape(2, 2, 2, 12)
    expected_tokens = expected_tokens + model.clip_order_embedding[None, :2]
    torch.testing.assert_close(tokens, expected_tokens.flatten(1, 2), rtol=0, atol=0)
    for name, expected in (
        ("human_self", expected_human),
        ("robot_self", expected_robot),
    ):
        for suffix, key in (
            ("", "loss"),
            ("_l1", "recon_loss"),
            ("_cosine", "cosine_loss"),
        ):
            torch.testing.assert_close(
                losses[name + suffix], expected[key], rtol=0, atol=0
            )
    # Verify the wrapper preserves the encoder gradient of the original loss.
    losses["human_self"].backward()
    expected_human["loss"].backward()
    torch.testing.assert_close(
        model.tokenizer.delta_input_proj.weight.grad,
        reference.delta_input_proj.weight.grad,
        rtol=0,
        atol=0,
    )


def test_depth_is_never_instantiated_and_unknown_depth_input_fails():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    assert model.tokenizer.use_depth is False
    assert not any("depth" in name for name, _ in model.named_parameters())
    human, robot = _inputs()
    _, losses = model(human, robot, compute_aux=True)
    losses["total"].backward()
    _assert_nonzero_finite_grad(model.tokenizer.delta_head[-1].weight)
    with pytest.raises(ValueError, match="use_depth=False"):
        RGBTransitionConditioner(_config(use_depth=True))
    with pytest.raises(TypeError, match="current_depth"):
        model(human, current_depth=torch.ones(2, 8, 8))


def test_robot_future_only_changes_auxiliary_targets_not_human_tokens():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    human, robot = _inputs()
    changed = robot.clone()
    changed[:, 1:] += 6.0
    tokens, losses = model(human, robot, compute_aux=True)
    changed_tokens, changed_losses = model(human, changed, compute_aux=True)
    inference_tokens, no_losses = model(human)
    current_only_tokens, _ = model(human, robot[:, :1])
    for actual in (changed_tokens, inference_tokens, current_only_tokens):
        torch.testing.assert_close(tokens, actual, rtol=0, atol=0)
    assert no_losses == {}
    assert not torch.isclose(losses["transfer"], changed_losses["transfer"])
    assert not torch.isclose(losses["robot_self"], changed_losses["robot_self"])


def test_transfer_prediction_ignores_robot_future_and_detaches_targets():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    human, robot = _inputs(requires_grad=True)
    recorded = []
    original = model.tokenizer.decode_delta

    def record_prediction(current, tokens, *, num_future_steps):
        prediction = original(current, tokens, num_future_steps=num_future_steps)
        recorded.append(prediction.detach().clone())
        return prediction

    model.tokenizer.decode_delta = record_prediction
    _, losses = model(human, robot, compute_aux=True)
    first_transfer = recorded[-1]
    altered_robot = robot.detach().clone()
    altered_robot[:, 1:] *= -3
    model(human, altered_robot, compute_aux=True)
    torch.testing.assert_close(first_transfer, recorded[-1], rtol=0, atol=0)
    losses["transfer"].backward()
    assert human.grad is None
    assert robot.grad is None
    _assert_nonzero_finite_grad(model.tokenizer.delta_input_proj.weight)
    _assert_nonzero_finite_grad(model.tokenizer.delta_head[-1].weight)
    _assert_nonzero_finite_grad(model.clip_order_embedding)


def test_zero_gate_preserves_hidden_exactly_but_can_start_learning():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    assert all(parameter.ndim > 0 for parameter in model.parameters())
    assert model.gate.numel() == 1
    human, _ = _inputs()
    tokens, _ = model(human)
    hidden = torch.randn(2, 7, 18, requires_grad=True)
    fused = model.fuse(hidden, tokens)
    assert torch.equal(fused, hidden)
    fused.square().mean().backward()
    _assert_nonzero_finite_grad(model.gate)
    assert model.output_proj.weight.grad is not None
    assert torch.count_nonzero(model.output_proj.weight.grad) == 0
    torch.testing.assert_close(hidden.grad, 2 * hidden.detach() / hidden.numel())


def test_nonzero_gate_propagates_policy_gradient_to_transition_encoder():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    with torch.no_grad():
        model.gate.fill_(0.2)
    human, _ = _inputs()
    tokens, _ = model(human)
    hidden = torch.randn(2, 7, 18)
    fused = model.fuse(hidden, tokens)
    assert not torch.equal(fused, hidden)
    fused.square().mean().backward()
    for parameter in (
        model.query_proj.weight,
        model.key_proj.weight,
        model.value_proj.weight,
        model.output_proj.weight,
        model.tokenizer.delta_input_proj.weight,
        model.clip_order_embedding,
    ):
        _assert_nonzero_finite_grad(parameter)


def test_feature_normalization_is_per_patch_and_frozen():
    model = RGBTransitionConditioner(
        _config(feature_normalization="layer_norm"), video_dim=18
    )
    human, _ = _inputs(requires_grad=True)
    actual = model._prepare_features(human)
    expected = F.layer_norm(human.detach(), (8,), eps=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    changed = human.detach().clone()
    changed[:, :, 1:] *= 5
    torch.testing.assert_close(
        actual[:, :, 0], model._prepare_features(changed)[:, :, 0], rtol=0, atol=0
    )


def test_clip_position_disambiguates_order_for_transfer_attention():
    model = RGBTransitionConditioner(_config(), video_dim=18)
    human, robot = _inputs()
    _, original = model(human, robot, compute_aux=True)
    _, reordered = model(human.flip(1), robot, compute_aux=True)
    assert abs((original["transfer"] - reordered["transfer"]).item()) > 1e-7
    with torch.no_grad():
        model.clip_order_embedding.zero_()
    _, original = model(human, robot, compute_aux=True)
    _, reordered = model(human.flip(1), robot, compute_aux=True)
    torch.testing.assert_close(original["transfer"], reordered["transfer"])


def test_save_load_preserves_tokens_losses_and_fusion():
    model = RGBTransitionConditioner(_config(), video_dim=18).eval()
    with torch.no_grad():
        model.gate.fill_(0.3)
    human, robot = _inputs()
    hidden = torch.randn(2, 7, 18)
    tokens, losses = model(human, robot, compute_aux=True)
    fused = model.fuse(hidden, tokens)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = RGBTransitionConditioner(_config(), video_dim=18).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    restored_tokens, restored_losses = restored(human, robot, compute_aux=True)
    torch.testing.assert_close(tokens, restored_tokens, rtol=0, atol=0)
    for key in losses:
        torch.testing.assert_close(losses[key], restored_losses[key], rtol=0, atol=0)
    torch.testing.assert_close(
        fused, restored.fuse(hidden, restored_tokens), rtol=0, atol=0
    )


def test_gradient_checkpointing_preserves_auxiliary_loss_and_encoder_gradient():
    plain = RGBTransitionConditioner(_config(), video_dim=18).train()
    checkpointed = RGBTransitionConditioner(
        _config(gradient_checkpointing=True), video_dim=18
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    human, robot = _inputs()
    plain_tokens, plain_losses = plain(human, robot, compute_aux=True)
    ckpt_tokens, ckpt_losses = checkpointed(human, robot, compute_aux=True)
    torch.testing.assert_close(plain_tokens, ckpt_tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        plain_losses["total"], ckpt_losses["total"], rtol=0, atol=0
    )
    plain_losses["total"].backward()
    ckpt_losses["total"].backward()
    torch.testing.assert_close(
        plain.tokenizer.delta_input_proj.weight.grad,
        checkpointed.tokenizer.delta_input_proj.weight.grad,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("autocast", [False, True])
def test_bfloat16_forward_backward_has_fp32_finite_aux_losses(autocast):
    model = RGBTransitionConditioner(
        _config(feature_normalization="layer_norm"), video_dim=18
    )
    if not autocast:
        model = model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.gate.fill_(0.2)
    human, robot = _inputs()
    hidden = torch.randn(2, 7, 18, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        tokens, losses = model(human, robot, compute_aux=True)
        fused = model.fuse(hidden, tokens)
        objective = losses["total"] + fused.float().square().mean()
    assert fused.dtype == hidden.dtype
    assert all(value.dtype == torch.float32 for value in losses.values())
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert torch.isfinite(fused).all()
    objective.backward()
    for parameter in (
        model.gate,
        model.query_proj.weight,
        model.tokenizer.delta_input_proj.weight,
    ):
        _assert_nonzero_finite_grad(parameter)


@pytest.mark.parametrize(
    "bad_human,error",
    [
        (torch.randn(2, 3, 4, 8), "5 dimensions"),
        (torch.randn(2, 2, 1, 4, 8), "2..5 frames"),
        (torch.randn(2, 2, 6, 4, 8), "2..5 frames"),
        (torch.randn(2, 2, 3, 4, 7), "feature dim"),
        (torch.randn(2, 4, 3, 4, 8), "clip count"),
        (torch.randn(2, 2, 3, 17, 8), "patch count"),
        (torch.full((2, 2, 3, 4, 8), float("nan")), "non-finite"),
    ],
)
def test_invalid_cache_inputs_fail_early(bad_human, error):
    model = RGBTransitionConditioner(_config(), video_dim=18)
    with pytest.raises(ValueError, match=error):
        model(bad_human)
