"""Role/effect training, causal policy conditioning and checkpoint contracts."""
from copy import deepcopy

import pytest
import torch

from test_role_effect import small_cpu_workload
from test_rgb_transition_integration import (
    cpu_attention, tiny_model, training_input, inference_input, save_checkpoint,
)
from wan_va.modules.icl_model import WanICLTransformer3DModel, ICL_CACHE_TYPE, PREDICTION_CACHE_TYPE
from wan_va.modules.role_effect import ROLE_LOSS_KEYS, RoleEffectConditioner


def role_model():
    cfg = dict(tiny_model().config)
    cfg["rgb_transition_config"] = dict(cfg["rgb_transition_config"],
        role_effect_enabled=True, readout_mode="state", readout_num_tokens=4,
        role_num_robot_views=1)
    return WanICLTransformer3DModel.from_config(cfg).eval()


def role_inputs():
    inputs = training_input()
    inputs["readout_observation_latents"] = inputs["latent_dict"]["latent"].clone()
    rgb = inputs["rgb_transition"]
    rgb["robot_features"] = rgb["robot_features"][:, :2]
    rgb["role_targets"] = {
        "human_masks": torch.rand(1, 2, 3, 2, 4),
        "human_valid": torch.ones(1, 2, 3, 2, dtype=torch.bool),
        "robot_masks": torch.rand(1, 2, 2, 4),
        "robot_valid": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    rgb["donor_human_features"] = torch.randn_like(rgb["human_features"])
    rgb["donor_valid"] = torch.ones(1, dtype=torch.bool)
    return inputs


def test_complete_role_training_uses_all_losses_without_exposing_auxiliary_context():
    model = role_model()
    base = tiny_model(False)
    base.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("rgb_transition.")}, strict=True)
    inputs = role_inputs()
    output = model(inputs, train_mode=True)
    expected = base(inputs, train_mode=True)
    for a, b in zip(output["base_pred"], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(key in output["rgb_losses"] for key in ROLE_LOSS_KEYS)
    assert all(torch.is_tensor(value) for value in output["rgb_losses"].values())
    assert all(output["rgb_losses"][key] > 0 for key in ROLE_LOSS_KEYS)
    loss = output["rgb_losses"]["total"] + sum(x.square().mean() for x in output["base_pred"])
    loss.backward()
    for name, parameter in model.rgb_transition.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.rgb_transition.robot_query_projection.weight.grad.abs().sum() > 0
    assert model.rgb_transition.gate.grad.abs().sum() > 0


def test_teacher_and_donor_changes_leave_actual_nonzero_gate_policy_unchanged():
    model = role_model()
    model.rgb_transition.gate.data.fill_(1)
    inputs = role_inputs()
    first = model(inputs, train_mode=True)
    changed = deepcopy(inputs)
    rgb = changed["rgb_transition"]
    rgb["robot_features"][:, 1:] += 5
    rgb["donor_human_features"] *= -3
    for name in ("human_masks", "robot_masks"):
        rgb["role_targets"][name] = 1 - rgb["role_targets"][name]
    second = model(changed, train_mode=True)
    for a, b in zip(first["base_pred"], second["base_pred"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(first["rgb_losses"]["total"], second["rgb_losses"]["total"])


def test_deployment_needs_only_human_features_and_current_wan_observation():
    model = role_model()
    model.rgb_transition.gate.data.fill_(1)
    prompt = inference_input(ICL_CACHE_TYPE)
    prompt["rgb_transition"] = {"human_features": torch.randn(1, 2, 3, 4, 8)}
    target = inference_input(PREDICTION_CACHE_TYPE, frame_id=2)
    target["readout_state_latent"] = torch.randn(1, 4, 1, 1, 2)
    with torch.no_grad():
        model(prompt, update_cache=1)
        original_memory = model.rgb_readout_memory.clone()
        first = model(target)
        target["readout_state_latent"] *= -2
        second = model(target)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.equal(first, second)
    torch.testing.assert_close(model.rgb_readout_memory, original_memory, rtol=0, atol=0)
    model.clear_cache()
    assert model.rgb_readout_memory is None


def test_role_model_roundtrip_and_original_checkpoint_initialization(tmp_path):
    original = tiny_model(False)
    save_checkpoint(original, tmp_path / "base")
    config = dict(role_model().config["rgb_transition_config"])
    adapted = WanICLTransformer3DModel.from_pretrained(tmp_path / "base",
        enable_rgb_transition=True, rgb_transition_config=config, low_cpu_mem_usage=False)
    assert isinstance(adapted.rgb_transition, RoleEffectConditioner)
    assert all(torch.isfinite(parameter).all() for parameter in adapted.rgb_transition.parameters())
    save_checkpoint(adapted, tmp_path / "role")
    restored = WanICLTransformer3DModel.from_pretrained(tmp_path / "role", low_cpu_mem_usage=False)
    assert isinstance(restored.rgb_transition, RoleEffectConditioner)
    for key, value in adapted.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_wrong_robot_timeline_is_rejected_before_role_readout():
    model = role_model()
    inputs = role_inputs()
    inputs["rgb_transition"]["robot_features"] = torch.randn(1, 3, 4, 8)
    inputs["rgb_transition"]["role_targets"].pop("robot_masks")
    inputs["rgb_transition"]["role_targets"].pop("robot_valid")
    with pytest.raises(ValueError, match="complete Wan frame timeline"):
        model(inputs, train_mode=True)
