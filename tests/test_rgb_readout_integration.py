"""Causal observation selection and deployment contracts for readout policies."""
from copy import deepcopy

import pytest
import torch

from test_rgb_transition_integration import (
    cpu_attention, tiny_model, training_input, inference_input, save_checkpoint,
)
from wan_va.modules.icl_model import (
    WanICLTransformer3DModel, ICL_CACHE_TYPE, OBSERVATION_CACHE_TYPE,
    PREDICTION_CACHE_TYPE,
)
from wan_va.utils import get_mesh_id


def model_for(mode, mcp=False):
    cfg = dict(tiny_model(mcp=mcp).config)
    cfg["rgb_transition_config"] = dict(cfg["rgb_transition_config"],
        readout_mode=mode, readout_num_tokens=4, readout_global_tokens=1)
    return WanICLTransformer3DModel.from_config(cfg).eval()


def readout_training_input(with_icl=True):
    inputs = training_input(with_icl=with_icl)
    if with_icl:
        inputs["readout_observation_latents"] = inputs["latent_dict"]["latent"].clone()
    return inputs


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_zero_gate_preserves_policy_and_all_new_gradients(mode):
    model = model_for(mode, mcp=True)
    base = tiny_model(False, mcp=True)
    base.load_state_dict({k: v for k, v in model.state_dict().items()
                         if not k.startswith("rgb_transition.")}, strict=True)
    inputs = readout_training_input()
    inputs["mcp_latent_dicts"] = [deepcopy(inputs["latent_dict"])]
    output = model(inputs, train_mode=True)
    expected = base(inputs, train_mode=True)
    for before, after in zip(expected[:2], output["base_pred"][:2]):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.testing.assert_close(expected[2][0], output["base_pred"][2][0], atol=0, rtol=0)
    loss = sum(x.square().mean() for x in output["base_pred"][:2])
    (loss + output["rgb_losses"]["total"]).backward()
    for name, parameter in model.rgb_transition.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.rgb_transition.gate.grad.abs().item() > 0


def test_chunk_reads_preceding_observation_not_current_or_future():
    model = model_for("state")
    inputs = readout_training_input()
    data = torch.randn(1, 4, 6, 1, 2)
    inputs["latent_dict"].update(latent=data, grid_id=get_mesh_id(6, 1, 2, 0)[None])
    inputs["readout_observation_latents"] = data.clone()
    inputs["chunk_size"] = 2
    clean = torch.randn(1, 12, 36)
    noisy = torch.randn_like(clean)
    memory = torch.randn(1, 4, 12)
    observed_features, _, _ = model._training_embed(
        data, torch.zeros(1, 6), "video")
    states, outputs = [], []
    hook = model.rgb_transition.readout.register_forward_hook(
        lambda _m, args, out: (states.append(args[1].clone()), outputs.append(out.clone())) and None)
    model._training_rgb_readout(noisy, clean, memory, inputs)
    for actual, frame in zip(states, [0, 1, 3]):
        torch.testing.assert_close(actual, observed_features[:, 2 * frame:2 * frame + 2])
    original_outputs = [x.clone() for x in outputs]
    states.clear(); outputs.clear()
    changed = clean.clone()
    changed[:, 8:] += 10  # frames 4 and 5 are unavailable for every readout here
    inputs["readout_observation_latents"][:, :, 4:] += 10
    model._training_rgb_readout(noisy, changed, memory, inputs)
    for before, after in zip(original_outputs, outputs):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    hook.remove()


def test_readout_uses_clean_observation_despite_conditioned_history_noise():
    model = model_for("state")
    inputs = readout_training_input()
    tokens = []
    hook = model.rgb_transition.readout.register_forward_hook(
        lambda _m, _args, out: tokens.append(out.detach().clone()))
    model(inputs, train_mode=True)
    expected = list(tokens)
    tokens.clear()
    inputs["latent_dict"]["latent"] = torch.randn_like(inputs["latent_dict"]["latent"]) * 4
    inputs["latent_dict"]["cond_timesteps"].fill_(900)
    model(inputs, train_mode=True)
    assert len(tokens) == len(expected) == 2
    for before, after in zip(expected, tokens):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    hook.remove()


def test_training_requires_explicit_original_clean_observation():
    model = model_for("state")
    inputs = training_input()
    with pytest.raises(ValueError, match="original clean readout_observation_latents"):
        model(inputs, train_mode=True)
    inputs["readout_observation_latents"] = inputs["latent_dict"]["latent"][:, :, :1]
    with pytest.raises(ValueError, match="match the robot latent layout"):
        model(inputs, train_mode=True)


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_future_teacher_changes_only_alignment_not_policy(mode):
    model = model_for(mode)
    model.rgb_transition.gate.data.fill_(0.5)
    inputs = readout_training_input()
    inputs["readout_teacher_valid"] = torch.tensor([False, True])
    targets = torch.randn(2, 4, 12, requires_grad=True)
    inputs["readout_teacher_targets"] = targets
    a = model(inputs, train_mode=True)
    inputs["readout_teacher_targets"] = torch.randn_like(targets)
    b = model(inputs, train_mode=True)
    for before, after in zip(a["base_pred"], b["base_pred"]):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert a["rgb_losses"]["teacher_alignment"] != b["rgb_losses"]["teacher_alignment"]
    a["rgb_losses"]["teacher_alignment"].backward()
    assert targets.grad is None
    assert model.rgb_transition.readout.queries.grad.abs().sum() > 0


def cached_inputs():
    human = inference_input(ICL_CACHE_TYPE)
    human["rgb_transition"] = {"human_features": torch.randn(1, 2, 3, 4, 8)}
    target = inference_input(PREDICTION_CACHE_TYPE, frame_id=2)
    target["readout_state_latent"] = torch.randn(1, 4, 1, 1, 2)
    return human, target


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_readout_cfg_reset_and_actual_observation_effect(mode):
    model = model_for(mode)
    model.rgb_transition.gate.data.fill_(0.5)
    human, target = cached_inputs()
    negative = deepcopy(target)
    negative["current_seq_ids"].fill_(1)
    negative["encoder_seq_ids"].fill_(1)
    with torch.no_grad():
        model(human, update_cache=1)
        assert model.rgb_readout_memory is not None
        before = model(target)
        negative_before = model(negative)
        changed = deepcopy(target)
        changed["readout_state_latent"] = torch.randn_like(changed["readout_state_latent"])
        after = model(changed)
        if mode == "static":
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        else:
            assert not torch.equal(before, after)
        model.clear_prediction_cache()
        assert model.rgb_readout_memory is not None
        model.clear_cache()
        assert model.rgb_readout_memory is None
        human["rgb_transition"]["human_features"] *= -3
        model(human, update_cache=1)
        negative_after = model(negative)
        torch.testing.assert_close(negative_before, negative_after, atol=0, rtol=0)


def test_packed_cfg_negative_is_untouched_by_readout_memory():
    model = model_for("state")
    model.rgb_transition.gate.data.fill_(1)
    human, packed = cached_inputs()
    stream = packed["latent_res_lst"]
    stream["noisy_latents"] = stream["noisy_latents"].repeat(1, 1, 2, 1, 1)
    stream["timesteps"] = stream["timesteps"].repeat(2)
    stream["cache_type_ids"] = stream["cache_type_ids"].repeat(2)
    packed["latent_grid_id"] = packed["latent_grid_id"].repeat(1, 2)
    packed["current_seq_ids"] = torch.tensor([0, 0, 1, 1])
    packed["current_frame_ids"] = torch.full((4,), 2)
    packed["text_emb"] = packed["text_emb"].repeat(1, 2, 1)
    packed["encoder_seq_ids"] = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        model(human, update_cache=1)
        a = model(packed)
        model.rgb_readout_memory = torch.randn_like(model.rgb_readout_memory)
        b = model(packed)
    assert not torch.equal(a[:, :2], b[:, :2])
    torch.testing.assert_close(a[:, 2:], b[:, 2:], atol=0, rtol=0)


@pytest.mark.parametrize("packed_cfg", [False, True])
def test_server_updates_readout_only_after_caching_real_observation(packed_cfg):
    from test_icl_server import _mock_server

    server = _mock_server()
    server.transformer.uses_rgb_readout = True
    server.transformer.clear_prediction_cache = lambda *_args: None
    server.use_icl_cfg = packed_cfg
    server.icl_guidance_scale = 2.0
    server._encode_initial_obs = lambda _obs: torch.full((1, 48, 1, 1, 1), 2.0)
    server._encode_obs = lambda _obs: torch.full((1, 48, 1, 1, 1), 7.0)
    server._infer_icl({"obs": []})
    first_calls = [item for mode, _, item in server.transformer.calls
                   if mode == "forward_latent_only"]
    assert first_calls
    for item in first_calls:
        assert torch.all(item["readout_state_latent"] == 2)
        assert item["readout_state_latent"].shape[2] == 1
    # Neither generated future frames nor this chunk's newly observed outcome
    # may rewrite the conditioning used when its history is cached.
    server.last_predicted_latents.fill_(999)
    server.transformer.calls.clear()
    server._compute_icl_kv_cache({"obs": []})
    cached_video = server.transformer.calls[0][2]
    assert torch.all(cached_video["readout_state_latent"] == 2)
    assert torch.all(server.readout_observation_latent == 7)
    server.transformer.calls.clear()
    server._infer_icl({"obs": []})
    second_calls = [item for mode, _, item in server.transformer.calls
                    if mode == "forward_latent_only"]
    for item in second_calls:
        assert torch.all(item["readout_state_latent"] == 7)
        assert item["readout_state_latent"].shape[2] == 1
        assert item["latent_res_lst"]["noisy_latents"].shape[2] == (4 if packed_cfg else 2)
        assert set(item["current_seq_ids"].tolist()) == ({0, 1} if packed_cfg else {0})
    # The earlier call's captured state owns its data across later updates.
    assert torch.all(first_calls[0]["readout_state_latent"] == 2)


def test_missing_observation_rejected_and_dropicl_needs_none():
    model = model_for("state")
    human, target = cached_inputs()
    del target["readout_state_latent"]
    with torch.no_grad():
        model(human, update_cache=1)
        with pytest.raises(ValueError, match="actual observed"):
            model(target)
        model.clear_cache()
        assert torch.isfinite(model(target)).all()
    dropped = model(training_input(with_icl=False), train_mode=True)
    dropped["rgb_losses"]["total"].backward()
    assert all(p.grad is not None for p in model.rgb_transition.parameters())


@pytest.mark.parametrize("mode", ["static", "state", "global_local"])
def test_checkpoint_roundtrip_readout_and_original_initialization(tmp_path, mode):
    model = model_for(mode)
    save_checkpoint(model, tmp_path / "readout")
    restored, info = WanICLTransformer3DModel.from_pretrained(
        tmp_path / "readout", output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert restored.rgb_transition.readout_mode == mode
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    save_checkpoint(tiny_model(False), tmp_path / "base")
    initialized = WanICLTransformer3DModel.from_pretrained(
        tmp_path / "base", enable_rgb_transition=True,
        rgb_transition_config=dict(model.config.rgb_transition_config))
    assert initialized.rgb_transition.gate.item() == 0
    assert all(torch.isfinite(p).all() for p in initialized.rgb_transition.parameters())
