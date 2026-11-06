import json

import pytest
import torch

from tools.train_rgb_future_teacher import arguments, select_training_chunk
from wan_va.modules.future_teacher import (
    ActionFutureTeacher, load_future_teacher, save_future_teacher,
)


@pytest.fixture
def teacher():
    torch.manual_seed(19)
    return ActionFutureTeacher(feature_dim=8, hidden_dim=12, num_tokens=4,
                               num_heads=3, encoder_layers=1, action_layers=1,
                               action_dim=6, max_future_steps=4, max_patches=4,
                               max_action_steps=8)


@pytest.fixture
def inputs():
    torch.manual_seed(20)
    current = torch.randn(2, 4, 8, requires_grad=True)
    future = torch.randn(2, 3, 4, 8, requires_grad=True)
    actions = torch.randn(2, 6, 6, requires_grad=True)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, 2:4] = False
    return current, future, actions, mask


def _spec():
    return {"rgb_encoder_id": "test/frozen-rgb", "rgb_encoder_revision": "abc123",
            "rgb_preprocessing": {"rgb": True, "output": "patches"}}


def test_action_loss_trains_future_queries_without_backprop_into_cached_inputs(teacher, inputs):
    loss = teacher.action_loss(*inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert teacher.query_tokens.grad.norm() > 0
    assert teacher.visual_projection.weight.grad.norm() > 0
    assert teacher.query_encoder.encoder.layer[0].crossattention.self.key.weight.grad.norm() > 0
    assert teacher.action_projection.weight.grad.norm() > 0
    assert all(t.grad is None for t in inputs[:3])


def test_future_target_cannot_read_current_state_or_action_noise(teacher, inputs):
    current, future, actions, mask = inputs
    before = teacher.encode_future(future).detach().clone()
    teacher.action_loss(current, future, actions, mask, noise=torch.zeros_like(actions), flow_time=torch.zeros(2))
    teacher.action_loss(current * 2, future, -actions, mask, noise=torch.ones_like(actions), flow_time=torch.ones(2))
    torch.testing.assert_close(teacher.encode_future(future), before, rtol=0, atol=0)
    # The target API intentionally has no action/current-state argument.
    with pytest.raises(TypeError):
        teacher.encode_future(future, actions=actions)


def test_inactive_actions_cannot_change_input_or_supervised_loss(teacher, inputs):
    current, future, actions, mask = inputs
    noise, times = torch.randn_like(actions), torch.tensor([.2, .8])
    first = teacher.action_loss(current, future, actions, mask, noise=noise, flow_time=times)
    changed_actions, changed_noise = actions.detach().clone(), noise.clone()
    changed_actions[~mask] = 1e20
    changed_noise[~mask] = -1e20
    second = teacher.action_loss(current, future, changed_actions, mask, noise=changed_noise, flow_time=times)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no valid action"):
        teacher.action_loss(current, future, actions, torch.zeros_like(mask))


def test_future_positions_preserve_temporal_and_spatial_order(teacher, inputs):
    future = inputs[1]
    original = teacher.encode_future(future)
    temporal = teacher.encode_future(future.flip(1))
    spatial = teacher.encode_future(future.flip(2))
    assert (original - temporal).abs().max() > 1e-7
    assert (original - spatial).abs().max() > 1e-7


def test_action_chunk_uses_preceding_observation_and_exact_anchor_blocks():
    rgb = torch.arange(5.).view(1, 5, 1, 1).expand(1, 5, 4, 8).clone()
    # Encode anchor, within-anchor action offset, and channel into values.
    actions = torch.zeros(1, 6, 5, 2, 1)
    for channel in range(6):
        for frame in range(5):
            for offset in range(2):
                actions[0, channel, frame, offset, 0] = 100 * frame + 10 * offset + channel
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, 3] = False
    selected, interval = select_training_chunk(
        dict(rgb_robot_features=rgb, actions=actions, actions_mask=mask), start=2, end=4)
    current, future, selected_actions, selected_mask = selected
    assert interval == (2, 4)
    assert torch.equal(current, rgb[:, 1])
    assert torch.equal(future, rgb[:, 2:4])
    assert selected_actions.shape == (1, 4, 6)
    assert selected_actions[0, :, 1].tolist() == [201, 211, 301, 311]
    assert not selected_mask[:, :, 3].any()
    assert selected_mask[:, :, 0].all()
    with pytest.raises(ValueError, match="Invalid future interval"):
        select_training_chunk(dict(rgb_robot_features=rgb, actions=actions, actions_mask=mask), start=0, end=1)


def test_saved_teacher_roundtrip_is_frozen_and_has_verified_provenance(teacher, inputs, tmp_path):
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    teacher.action_loss(*inputs).backward()
    optimizer.step()
    target = teacher.eval().encode_future(inputs[1]).detach()
    path = tmp_path / "teacher"
    metadata = save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=1,
                                   smoke_test=False, training={"heldout_tasks_used": False})
    restored, loaded_metadata = load_future_teacher(path, expected_feature_spec=_spec())
    assert loaded_metadata == metadata
    assert not restored.training
    assert not any(p.requires_grad for p in restored.parameters())
    torch.testing.assert_close(restored.encode_future(inputs[1]), target, rtol=0, atol=0)
    with pytest.raises(ValueError, match="provenance differs"):
        load_future_teacher(path, expected_feature_spec={**_spec(), "rgb_encoder_revision": "wrong"})
    with pytest.raises(FileExistsError, match="overwrite"):
        save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=1, smoke_test=False)


def test_teacher_chunk_sampler_excludes_future_anchors_with_no_valid_actions():
    batch = dict(rgb_robot_features=torch.randn(1, 5, 4, 8), actions=torch.randn(1, 6, 5, 2, 1),
                 actions_mask=torch.zeros(1, 6, 5, 2, 1, dtype=torch.bool))
    batch["actions_mask"][:, :2, 2] = True
    for _ in range(8):
        selected, (start, _) = select_training_chunk(batch)
        assert start == 2 and selected[-1].any()
    batch["actions_mask"].zero_()
    with pytest.raises(ValueError, match="no valid future action targets"):
        select_training_chunk(batch)


def test_smoke_teacher_cannot_silently_initialize_formal_policy(teacher, tmp_path):
    path = tmp_path / "teacher"
    save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=2, smoke_test=True)
    with pytest.raises(ValueError, match="Smoke-only"):
        load_future_teacher(path)
    restored, metadata = load_future_teacher(path, allow_smoke=True)
    assert metadata["smoke_test"] and not restored.training


def test_teacher_requires_successful_training_and_completed_hashes(teacher, tmp_path):
    with pytest.raises(ValueError, match="successful optimizer step"):
        save_future_teacher(teacher, tmp_path / "zero", feature_spec=_spec(), successful_steps=0, smoke_test=False)
    with pytest.raises(ValueError, match="incomplete"):
        save_future_teacher(teacher, tmp_path / "no_spec", feature_spec={}, successful_steps=1, smoke_test=False)
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        load_future_teacher(tmp_path / "missing")
    path = tmp_path / "teacher"
    save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=1, smoke_test=False)
    payload = json.loads((path / "metadata.json").read_text())
    payload["successful_steps"] = 2
    (path / "metadata.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_future_teacher(path)


@pytest.mark.parametrize("extra", [
    ["--smoke-test"],
    ["--smoke-test", "--num-steps", "21", "--sample-ids", "pair"],
    ["--sample-ids", "pair"],
    ["--num-steps", "0"],
    ["--gradient-accumulation-steps", "0"],
    ["--learning-rate", "nan"],
    ["--num-tokens", "0"],
    ["--num-tokens", "-16"],
])
def test_cli_cannot_accidentally_start_long_or_implicit_subset_smoke(extra):
    with pytest.raises(SystemExit):
        arguments(["--output", "/workspace/unused-teacher-test", *extra])


def test_cli_explicit_bounded_smoke():
    args = arguments(["--output", "/workspace/unused-teacher-test", "--smoke-test", "--num-steps", "2",
                      "--sample-ids", "task:explicit-pair"])
    assert args.num_steps == 2 and args.gradient_accumulation_steps == 8
    assert args.num_tokens == 32


def test_teacher_query_budget_is_explicitly_configurable_for_matched_student():
    args = arguments(["--output", "/workspace/unused-teacher-test", "--num-tokens", "16"])
    assert args.num_tokens == 16
