"""CPU contracts for the new condition, with real attention masks and KV caches.

Only the CUDA FlexAttention kernel is replaced by dense SDPA. Model projection,
mask predicates, timestep modulation, RGB branch, and cache code are exercised.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

import wan_va.modules.icl_model as icl_module
from wan_va.modules.icl_model import (
    ICLAttentionBackend,
    ICL_CACHE_TYPE,
    PREDICTION_CACHE_TYPE,
    WanICLTransformer3DModel,
)
from wan_va.utils import get_mesh_id


RGB_CONFIG = {
    "feature_dim": 8, "hidden_dim": 12, "num_heads": 3,
    "num_transition_tokens": 2, "encoder_layers": 1, "decoder_layers": 1,
    "max_clips": 4, "max_future_steps": 8, "max_patches": 8,
}


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    def make_mask(mask_mod, *_args, **_kwargs):
        return SimpleNamespace(mask_mod=mask_mod)

    def apply(cls, query, key, value, *, cross_attention, block_mask=None):
        mask = block_mask or (cls.cross_mask if cross_attention else cls.self_mask)
        q = torch.arange(query.shape[1], device=query.device)[:, None]
        k = torch.arange(key.shape[1], device=key.device)[None, :]
        index = q.new_zeros(())
        allowed = mask.mask_mod(index, index, q, k)
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=allowed,
        ).transpose(1, 2)

    monkeypatch.setattr(icl_module, "_compiled_create_block_mask", make_mask)
    monkeypatch.setattr(icl_module, "create_block_mask", make_mask)
    monkeypatch.setattr(ICLAttentionBackend, "apply", classmethod(apply))


def tiny_model(rgb=True, mcp=False):
    return WanICLTransformer3DModel(
        patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
        in_channels=4, out_channels=4, action_dim=3, text_dim=8, freq_dim=4,
        ffn_dim=16, num_layers=1, rope_max_seq_len=32,
        action_inner_dim=36, action_ffn_dim=16, attn_window=4,
        enable_mcp=mcp, num_mcp_modules=1, mcp_blocks_per_group=1,
        mcp_hidden_collect_layers=(0,), enable_rgb_transition=rgb,
        rgb_transition_config=RGB_CONFIG if rgb else None,
    ).eval()


def save_checkpoint(model, path):
    # Like the trainer, break the existing backbone's shared K/V storage before
    # safetensors serialization. This is unrelated to the new RGB branch.
    from safetensors.torch import save_file

    path.mkdir(exist_ok=True)
    model.save_config(path)
    save_file({name: value.clone().contiguous() for name, value in model.state_dict().items()},
              path / "diffusion_pytorch_model.safetensors")


def training_input(with_icl=True):
    def stream(data, grid):
        return {
            "noisy_latents": data, "latent": data.clone(),
            "timesteps": torch.zeros(1, 2), "cond_timesteps": torch.zeros(1, 2),
            "grid_id": grid[None],
        }

    result = {
        "latent_dict": stream(torch.randn(1, 4, 2, 1, 2), get_mesh_id(2, 1, 2, 0)),
        "action_dict": stream(torch.randn(1, 3, 2, 2, 1), get_mesh_id(2, 2, 1, 1)),
        "icl_latent_dict": None,
        "text_emb": torch.randn(1, 2, 8), "encoder_seq_ids": torch.zeros(2, dtype=torch.int),
        "chunk_size": 1, "max_frame_chunk_size": 4, "window_size": 4,
    }
    if with_icl:
        result.update(
            icl_latent_dict={
                "latent": torch.randn(1, 4, 2, 1, 2),
                "timesteps": torch.zeros(1, 2),
                "grid_id": get_mesh_id(2, 1, 2, 0, h_shift=4)[None],
            },
            text_emb=torch.randn(1, 4, 8), encoder_seq_ids=torch.tensor([0, 0, 1, 1]),
            rgb_transition={
                "human_features": torch.randn(1, 2, 3, 4, 8),
                "robot_features": torch.randn(1, 3, 4, 8),
            },
        )
    return result


def inference_input(cache_type, seq_id=0, frame_id=0):
    grid = get_mesh_id(1, 1, 2, 0)
    return {
        "latent_res_lst": {
            "noisy_latents": torch.randn(1, 4, 1, 1, 2),
            "timesteps": torch.zeros(1),
            "cache_type_ids": torch.full((2,), cache_type, dtype=torch.int),
        },
        "latent_grid_id": grid,
        "current_seq_ids": torch.full((2,), seq_id, dtype=torch.int),
        "current_frame_ids": torch.full((2,), frame_id, dtype=torch.int),
        "encoder_seq_ids": torch.full((2,), seq_id, dtype=torch.int),
        "text_emb": torch.randn(1, 2, 8),
    }


def test_zero_gate_preserves_base_predictions_and_auxiliary_training():
    base = tiny_model(False)
    model = tiny_model(True)
    model.load_state_dict(base.state_dict(), strict=False)
    inputs = training_input()
    expected = base(inputs, train_mode=True)
    actual = model(inputs, train_mode=True)
    assert set(actual) == {"base_pred", "rgb_losses"}
    for wanted, received in zip(expected, actual["base_pred"]):
        torch.testing.assert_close(received, wanted, atol=0, rtol=0)
    loss = sum(value.square().mean() for value in actual["base_pred"])
    loss = loss + actual["rgb_losses"]["total"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.rgb_transition.parameters())
    assert model.rgb_transition.gate.grad.abs() > 0


def test_robot_future_is_only_an_auxiliary_target():
    model = tiny_model()
    model.rgb_transition.gate.data.fill_(1)
    inputs = training_input()
    first = model(inputs, train_mode=True)
    inputs["rgb_transition"]["robot_features"] = torch.randn(1, 3, 4, 8) * 3
    second = model(inputs, train_mode=True)
    for before, after in zip(first["base_pred"], second["base_pred"]):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(first["rgb_losses"]["transfer"], second["rgb_losses"]["transfer"])


def test_drop_icl_has_zero_loss_and_gradients_for_every_new_parameter():
    model = tiny_model()
    result = model(training_input(with_icl=False), train_mode=True)
    assert result["rgb_losses"]["total"].item() == 0
    result["rgb_losses"]["total"].backward()
    for name, parameter in model.rgb_transition.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name


def test_drop_icl_bfloat16_zero_losses_match_other_ranks_fp32_loss_dtype():
    model = tiny_model().to(dtype=torch.bfloat16)
    inputs = training_input(with_icl=False)
    for name in ("latent_dict", "action_dict"):
        for key in ("noisy_latents", "latent"):
            inputs[name][key] = inputs[name][key].bfloat16()
    inputs["text_emb"] = inputs["text_emb"].bfloat16()
    result = model(inputs, train_mode=True)
    assert all(value.dtype == torch.float32 for value in result["rgb_losses"].values())
    result["rgb_losses"]["total"].backward()
    assert all(parameter.grad is not None for parameter in model.rgb_transition.parameters())


def test_required_feature_inputs_fail_before_attention():
    model = tiny_model()
    inputs = training_input()
    del inputs["rgb_transition"]
    with pytest.raises(ValueError, match="human_features are missing"):
        model(inputs, train_mode=True)
    with pytest.raises(ValueError, match="human_features are missing"):
        model(inference_input(ICL_CACHE_TYPE), update_cache=1)


def test_rgb_cache_does_not_leak_to_cfg_negative_or_survive_reset():
    model = tiny_model()
    model.rgb_transition.gate.data.fill_(1)
    human = inference_input(ICL_CACHE_TYPE)
    target = inference_input(PREDICTION_CACHE_TYPE, frame_id=2)
    negative = deepcopy(target)
    negative["current_seq_ids"].fill_(1)
    negative["encoder_seq_ids"].fill_(1)
    human["rgb_transition"] = {"human_features": torch.randn(1, 2, 3, 4, 8)}
    with torch.no_grad():
        model(human, update_cache=1)
        first_cond = model(target)
        first_uncond = model(negative)
        cached = model.blocks[0].attn1.attn_caches["pos"]["k"].clone()
        model(target, update_cache=1)
        model.clear_prediction_cache()
        torch.testing.assert_close(model.blocks[0].attn1.attn_caches["pos"]["k"], cached)
        assert model.cache_counts()[ICL_CACHE_TYPE] == 2
        model.clear_cache()
        assert model.cache_counts() == {}
        assert "pos" not in model.blocks[0].attn1.attn_caches
        human["rgb_transition"]["human_features"] = torch.randn(1, 2, 3, 4, 8)
        model(human, update_cache=1)
        second_cond = model(target)
        second_uncond = model(negative)
    assert not torch.equal(first_cond, second_cond)
    torch.testing.assert_close(first_uncond, second_uncond, atol=0, rtol=0)


def test_mcp_output_contract_is_preserved():
    model = tiny_model(mcp=True)
    inputs = training_input()
    inputs["mcp_latent_dicts"] = [deepcopy(inputs["latent_dict"])]
    output = model(inputs, train_mode=True)
    assert len(output["base_pred"]) == 3
    assert len(output["base_pred"][2]) == 1
    assert output["base_pred"][2][0].shape == output["base_pred"][0].shape


def test_old_checkpoint_initializes_branch_and_new_checkpoint_roundtrips(tmp_path):
    base = tiny_model(False)
    save_checkpoint(base, tmp_path / "base")
    model, info = WanICLTransformer3DModel.from_pretrained(
        tmp_path / "base", enable_rgb_transition=True,
        rgb_transition_config=RGB_CONFIG, output_loading_info=True,
    )
    assert info["missing_keys"]
    assert all(key.startswith("rgb_transition.") for key in info["missing_keys"])
    assert model.rgb_transition.gate.item() == 0
    assert all(not p.is_meta and torch.isfinite(p).all() for p in model.parameters())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    model.rgb_transition.gate.data.fill_(0.37)
    save_checkpoint(model, tmp_path / "rgb")
    restored, info = WanICLTransformer3DModel.from_pretrained(
        tmp_path / "rgb", output_loading_info=True,
    )
    assert not info["missing_keys"]
    assert restored.enable_rgb_transition
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_partial_rgb_checkpoint_is_rejected(tmp_path):
    from safetensors.torch import load_file, save_file

    save_checkpoint(tiny_model(), tmp_path)
    filename = tmp_path / "diffusion_pytorch_model.safetensors"
    state = load_file(filename)
    del state["rgb_transition.gate"]
    save_file(state, filename)
    with pytest.raises(RuntimeError, match="partially missing"):
        WanICLTransformer3DModel.from_pretrained(tmp_path)


def make_server(tmp_path):
    from wan_va.wan_va_server import VA_Server

    class Transformer:
        patch_size = (1, 1, 1)
        enable_rgb_transition = True
        config = SimpleNamespace(rgb_transition_config={
            "rgb_encoder_id": "tiny/teacher", "rgb_encoder_revision": "v1",
            "rgb_preprocessing": {"resize": 28},
        })

        def __init__(self):
            self.calls = []
            self.resets = 0

        def __call__(self, inputs, **_kwargs):
            self.calls.append(inputs)

        def clear_cache(self, _name):
            self.calls.clear()
            self.resets += 1

        def cache_counts(self):
            return {}

    server = VA_Server.__new__(VA_Server)
    server.device, server.dtype = torch.device("cpu"), torch.float32
    server.cache_name, server.env_type = "pos", "franka"
    server.transformer = Transformer()
    server.streaming_vae = SimpleNamespace(clear_cache=lambda: None)
    server.streaming_vae_half = None
    server.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[0] * 4))
    server.empty_text_emb = None
    server.encode_prompt = lambda **_: (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8))
    server._load_or_encode_icl = lambda *_: (torch.zeros(1, 4, 1, 1, 2), None)
    server.job_config = SimpleNamespace(
        rgb_feature_root=tmp_path, rgb_encoder_id="runtime/cannot-override-checkpoint",
        rgb_encoder_revision="wrong-runtime-revision", icl_rope_h=4,
        frame_chunk_size=2, action_per_frame=2, height=16, width=16,
        obs_cam_keys=["cam"], action_dim=3, used_action_channel_ids=[0, 1, 2],
        norm_stat={"q01": [-1] * 3, "q99": [1] * 3}, action_norm_method="quantiles",
    )
    return server


def write_human_features(root, name="demo", **metadata_overrides):
    from wan_va.dataset.rgb_feature_cache import human_feature_path

    latent_path = f"/latents/run_test/{name}.pth"
    filename = human_feature_path(root, latent_path)
    filename.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": torch.randn(2, 3, 4, 8),
        "frame_indices": torch.arange(6).reshape(2, 3),
        "metadata": {
            "schema_version": 1, "kind": "human", "encoder_id": "tiny/teacher",
            "encoder_revision": "v1", "preprocessing": {"resize": 28},
            "sample_id": name, "source_video_path": f"run_test/{name}.mp4",
            **metadata_overrides,
        },
    }
    torch.save(payload, filename)
    return latent_path, payload["features"]


def test_server_feature_contract_is_pinned_to_checkpoint(tmp_path):
    server = make_server(tmp_path)
    latent_path, features = write_human_features(tmp_path)
    torch.testing.assert_close(server._load_rgb_icl_features(latent_path), features[None])
    with pytest.raises(FileNotFoundError, match="Missing frozen RGB human"):
        server._load_rgb_icl_features("/latents/run_test/missing.pth")
    write_human_features(tmp_path, encoder_revision="v2")
    with pytest.raises(ValueError, match="encoder_revision mismatch"):
        server._load_rgb_icl_features(latent_path)
    write_human_features(tmp_path, preprocessing={"resize": 56})
    with pytest.raises(ValueError, match="preprocessing does not match"):
        server._load_rgb_icl_features(latent_path)


def test_server_reset_replaces_human_condition_and_cfg_has_no_new_features(tmp_path):
    server = make_server(tmp_path)
    first_path, first = write_human_features(tmp_path, "first")
    second_path, second = write_human_features(tmp_path, "second")
    server._reset_icl("task", True, None, first_path, 1, 2)
    assert len(server.transformer.calls) == 1
    cached = server.transformer.calls[0]
    torch.testing.assert_close(cached["rgb_transition"]["human_features"], first[None])
    assert torch.all(cached["current_seq_ids"] == 0)
    assert torch.all(cached["latent_res_lst"]["cache_type_ids"] == ICL_CACHE_TYPE)
    video = server._prepare_icl_stream(torch.zeros(1, 4, 1, 1, 2), 0, "video", PREDICTION_CACHE_TYPE)
    action = server._prepare_icl_stream(torch.zeros(1, 3, 1, 2, 1), 0, "action", PREDICTION_CACHE_TYPE)
    assert "rgb_transition" not in video and "rgb_transition" not in action
    assert set(video["current_seq_ids"].tolist()) == {0, 1}
    assert set(action["current_seq_ids"].tolist()) == {0}
    server._reset_icl("task", True, None, second_path, 1, 2)
    assert server.transformer.resets == 2 and len(server.transformer.calls) == 1
    torch.testing.assert_close(server.transformer.calls[0]["rgb_transition"]["human_features"], second[None])
    server._reset_icl("task", False, None, None, 1, 1)
    assert server.transformer.resets == 3 and not server.transformer.calls


def test_rgb_inference_rejects_human_tokens_in_negative_branch():
    model = tiny_model()
    inputs = inference_input(ICL_CACHE_TYPE, seq_id=1)
    inputs["rgb_transition"] = {"human_features": torch.randn(1, 2, 3, 4, 8)}
    with pytest.raises(ValueError, match="human-only, seq_id=0"):
        model(inputs, update_cache=1)


@pytest.mark.parametrize("rgb_enabled", [False, True])
def test_fp32_cfg_negative_matches_same_packing_and_single_stream_without_cache(rgb_enabled):
    model = tiny_model(rgb_enabled)
    human = inference_input(ICL_CACHE_TYPE)
    if rgb_enabled:
        model.rgb_transition.gate.data.fill_(0.2)
        human["rgb_transition"] = {"human_features": torch.randn(1, 2, 3, 4, 8)}
    single = inference_input(PREDICTION_CACHE_TYPE, frame_id=2)
    packed = deepcopy(single)
    packed["latent_res_lst"]["noisy_latents"] = single["latent_res_lst"]["noisy_latents"].repeat(1, 1, 2, 1, 1)
    packed["latent_res_lst"]["timesteps"] = single["latent_res_lst"]["timesteps"].repeat(2)
    packed["latent_res_lst"]["cache_type_ids"] = single["latent_res_lst"]["cache_type_ids"].repeat(2)
    packed["latent_grid_id"] = single["latent_grid_id"].repeat(1, 2)
    packed["current_frame_ids"] = single["current_frame_ids"].repeat(2)
    packed["current_seq_ids"] = torch.tensor([0, 0, 1, 1])
    packed["encoder_seq_ids"] = torch.tensor([0, 0, 1, 1])
    packed["text_emb"] = single["text_emb"].repeat(1, 2, 1)
    with torch.no_grad():
        model(human, update_cache=1)
        negative = model(packed).chunk(2, dim=1)[1]
        model.clear_cache()
        packed_no_cache = model(packed).chunk(2, dim=1)[1]
        single_no_cache = model(single)
    torch.testing.assert_close(negative, packed_no_cache, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(negative, single_no_cache, atol=1e-5, rtol=1e-5)
