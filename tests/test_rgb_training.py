"""CPU tests of the real Trainer's accumulation, logging and launch guards."""

import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import wan_va.train as train_module
from wan_va.configs import VA_CONFIGS
from wan_va.train import Trainer, build_training_config


class _Scheduler:
    num_train_timesteps = 1000
    timesteps = torch.arange(1000, dtype=torch.float32)

    def training_weight(self, timesteps):
        return torch.ones_like(timesteps)

    def add_noise(self, latent, noise, timesteps, t_dim):
        return latent

    def training_target(self, latent, noise, timesteps):
        return torch.zeros_like(latent)


class _TinyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.ones(1))
        self.rgb_transition = nn.Module()
        self.rgb_transition.encoder_weight = nn.Parameter(torch.ones(1))
        self.rgb_transition.gate = nn.Parameter(torch.zeros(1))
        self.sync_calls = []

    def set_requires_gradient_sync(self, enabled):
        self.sync_calls.append(enabled)

    def forward(self, inputs, train_mode):
        assert train_mode
        value = inputs["scale"]
        prediction = (self.backbone * value).expand(1, 2, 1)
        part = (self.rgb_transition.encoder_weight * value).square().mean()
        losses = {}
        for name, factor in (("human_self", 1), ("robot_self", 2), ("transfer", 3)):
            losses[name] = part * factor
            losses[name + "_l1"] = part * factor * 0.9
            losses[name + "_cosine"] = part * factor
        losses["total"] = part * 6 + self.rgb_transition.gate.sum() * 0
        return {"base_pred": (prediction, prediction), "rgb_losses": losses}


def _loss_input(value):
    return {
        "scale": torch.tensor(float(value)),
        "latent_dict": {
            "targets": torch.zeros(1, 1, 2, 1, 1),
            "timesteps": torch.zeros(1, 2),
        },
        "action_dict": {
            "targets": torch.zeros(1, 1, 2, 1, 1),
            "timesteps": torch.zeros(1, 2),
            "actions_mask": torch.ones(1, 1, 2, 1, 1),
        },
    }


def _trainer(sync_each_microbatch=False):
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.step = 0
    trainer.enable_mcp = False
    trainer.enable_rgb_transition = True
    trainer.patch_size = (1, 1, 1)
    trainer.gradient_accumulation_steps = 2
    trainer.config = SimpleNamespace(
        rank=1,
        world_size=4,
        batch_size=1,
        num_steps=1,
        enable_wandb=False,
        video_loss_reweight=False,
        action_loss_reweight=False,
        sync_each_microbatch=sync_each_microbatch,
        rgb_loss_weight=0.1,
        max_norm=1e6,
        skip_step_grad_norm_multiplier=0,
        save_interval=5000,
        gc_interval=5000,
    )
    trainer.transformer = _TinyTrainingModel()
    trainer.optimizer = torch.optim.SGD(trainer.transformer.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda=lambda _: 1
    )
    trainer.train_scheduler_latent = _Scheduler()
    trainer.train_scheduler_action = _Scheduler()
    trainer._prepare_input_dict = lambda batch: batch
    return trainer


@pytest.mark.parametrize("sync_each_microbatch", [False, True])
def test_auxiliary_weight_and_accumulation_are_applied_once(sync_each_microbatch):
    trainer = _trainer(sync_each_microbatch)
    first = trainer._train_step(_loss_input(1), 0)
    assert not first["should_log"]
    assert trainer.lr_scheduler.last_epoch == 0
    torch.testing.assert_close(trainer.transformer.backbone, torch.ones(1))
    torch.testing.assert_close(
        trainer.transformer.rgb_transition.encoder_weight.grad, torch.tensor([0.6])
    )
    second = trainer._train_step(_loss_input(3), 1)
    assert second["should_log"]
    assert not second["optimizer_step_skipped"]
    assert trainer.lr_scheduler.last_epoch == 1
    # mean(x**2)=5; d(base video+action)/dp=4*5=20;
    # d(0.1*(human+robot+transfer))/dq=0.1*12*5=6.
    torch.testing.assert_close(trainer.transformer.backbone, torch.tensor([0.8]))
    torch.testing.assert_close(
        trainer.transformer.rgb_transition.encoder_weight, torch.tensor([0.94])
    )
    assert trainer.transformer.sync_calls == [sync_each_microbatch, True]
    torch.testing.assert_close(
        first["rgb_loss"] + second["rgb_loss"], torch.tensor(3.0)
    )
    for key, expected in (("human_self", 5), ("robot_self", 10), ("transfer", 15)):
        actual = first["rgb_components"][key] + second["rgb_components"][key]
        torch.testing.assert_close(actual, torch.tensor(float(expected)))


def test_wandb_reports_optimizer_window_and_all_rgb_loss_components(monkeypatch):
    trainer = _trainer()
    trainer.config.rank = 0
    trainer.config.enable_wandb = True
    batches = iter([_loss_input(1), _loss_input(3)])
    trainer._get_next_batch = lambda: next(batches)
    events = []
    trainer.wandb = SimpleNamespace(
        log=lambda values, step: events.append((dict(values), step))
    )
    monkeypatch.delenv("ZERO_WAM_METRICS_PATH", raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(train_module, "dist_mean", lambda value: value)
    monkeypatch.setattr(train_module, "dist_max", lambda value: value)
    trainer.train()
    assert trainer.step == 1
    assert len(events) == 1
    metrics, step = events[0]
    assert step == 1
    assert metrics["train/effective_batch_size"] == 8
    assert metrics["rgb/weighted_total"] == pytest.approx(3)
    assert metrics["rgb/human_self"] == pytest.approx(5)
    assert metrics["rgb/robot_self"] == pytest.approx(10)
    assert metrics["rgb/transfer"] == pytest.approx(15)
    assert metrics["rgb/gate"] == 0
    for key in train_module.RGB_LOSS_COMPONENTS:
        assert "rgb/" + key in metrics
    assert metrics["rgb/human_self_l1"] == pytest.approx(4.5)
    assert metrics["loss_metrics/global_avg_video_loss"] == pytest.approx(5)


@pytest.mark.parametrize("drop_icl", [0.0, 1.0])
def test_prepare_input_drops_rgb_and_human_latents_together(drop_icl):
    trainer = _trainer()
    del trainer._prepare_input_dict
    trainer.config.frame_chunk_size = 1
    trainer.config.max_frame_chunk_size = 4
    trainer.config.noisy_img_prob = 0.0
    trainer.config.noisy_cond_min_timestep_bd = 0.0
    trainer.config.noisy_cond_max_timestep_bd = 1.0
    trainer.config.drop_icl = drop_icl
    trainer.config.icl_rope_h = 4
    trainer.config.attn_window = 4
    batch = {
        "latents": torch.ones(1, 1, 2, 1, 1),
        "actions": torch.ones(1, 1, 2, 1, 1),
        "actions_mask": torch.ones(1, 1, 2, 1, 1),
        "text_emb": torch.ones(1, 2, 4),
    }
    if drop_icl == 0:
        batch.update(
            icl_latents=torch.ones(1, 1, 2, 1, 1),
            icl_text_emb=torch.ones(1, 2, 4),
            rgb_human_features=torch.randn(1, 2, 3, 4, 8),
            rgb_robot_features=torch.randn(1, 3, 4, 8),
        )
    # For dropped ICL, no human/RGB key exists. Accessing any would fail here.
    result = trainer._prepare_input_dict(batch)
    if drop_icl == 1:
        assert result["icl_latent_dict"] is None
        assert "rgb_transition" not in result
        assert result["text_emb"].shape[1] == 2
    else:
        assert result["icl_latent_dict"] is not None
        assert result["rgb_transition"]["human_features"] is batch["rgb_human_features"]
        assert result["rgb_transition"]["robot_features"] is batch["rgb_robot_features"]


def _args(**overrides):
    values = dict(
        config_name="robotwin_subset12_rgb_train",
        datasets="robotwin:1.0",
        check_config=True,
        enable_wandb=False,
        disable_wandb=False,
        smoke_test=True,
        rgb_smoke_sample_ids="run_1/12,run_2/12",
        rgb_feature_root=None,
    )
    for key in (
        "model_path", "empty_emb_path", "learning_rate", "droptext_target",
        "drop_icl", "init_worker", "load_worker", "batch_size",
        "gradient_accumulation_steps", "num_steps", "save_interval", "save_root",
        "dataset_path", "icl_manifest_path", "human_latent_path", "robot_latent_path",
    ):
        values[key] = None
    values["num_steps"] = 2
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def config_data_root(tmp_path, monkeypatch):
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "ZERO_WAM_SYNC_EACH_MICROBATCH"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "meta").mkdir()
    shutil.copyfile(
        Path(__file__).parents[1] / "wan_va/assets/norm_stats/robotwin_icl.json",
        tmp_path / "meta/action_stats.json",
    )
    return str(tmp_path)


@pytest.mark.parametrize("steps", [None, 0, 21, 15000])
def test_smoke_guard_requires_explicit_bounded_steps(steps):
    with pytest.raises(ValueError, match="num-steps between 1 and 20"):
        build_training_config(_args(num_steps=steps))


@pytest.mark.parametrize("ids", [None, "", " , ", "run_1/12,run_1/12"])
def test_smoke_guard_requires_nonempty_unique_explicit_whitelist(ids):
    with pytest.raises(ValueError, match="whitelist"):
        build_training_config(_args(rgb_smoke_sample_ids=ids))


def test_sample_whitelist_cannot_silently_shrink_an_unbounded_run():
    with pytest.raises(ValueError, match="requires --smoke-test"):
        build_training_config(_args(smoke_test=False))


def test_smoke_selection_is_propagated_without_changing_registry(config_data_root):
    original = deepcopy(VA_CONFIGS["robotwin_subset12_rgb_train"])
    config = build_training_config(_args(dataset_path=config_data_root))
    source = config.dataset_sources[0]["config"]
    assert config.num_steps == 2
    effective_batch = (
        config.world_size * config.batch_size * config.gradient_accumulation_steps
    )
    assert effective_batch == 8
    assert config.smoke_test
    assert source.rgb_smoke_sample_ids == ["run_1/12", "run_2/12"]
    assert source.rgb_smoke_sample_ids is not config.rgb_smoke_sample_ids
    assert source.rgb_transition_config == config.rgb_transition_config
    assert source.enable_rgb_transition
    assert VA_CONFIGS["robotwin_subset12_rgb_train"] == original


def test_rgb_checkpoint_conflict_fails_before_model_allocation(tmp_path, monkeypatch):
    saved_rgb = {"feature_dim": 8, "rgb_encoder_revision": "previous-revision"}
    (tmp_path / "config.json").write_text(json.dumps({
        "enable_rgb_transition": True, "rgb_transition_config": saved_rgb,
    }))
    config = SimpleNamespace(
        enable_wandb=False, rank=0, local_rank=0, param_dtype=torch.float32,
        patch_size=(1, 1, 1), enable_mcp=False, enable_rgb_transition=True,
        batch_size=1, model_path=str(tmp_path),
        rgb_transition_config={**saved_rgb, "rgb_encoder_revision": "new-revision"},
    )
    monkeypatch.setattr(
        Trainer, "_resolve_transformer_path", staticmethod(lambda path: str(tmp_path))
    )
    monkeypatch.setattr(
        train_module, "load_transformer",
        lambda *_a, **_kw: pytest.fail("Allocated model before validating provenance"),
    )
    with pytest.raises(ValueError, match="architecture/feature provenance differs"):
        Trainer(config)
