"""CPU contracts for the real Trainer's frozen-teacher integration."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import wan_va.train as train_module
from wan_va.checkpoint_io import save_checkpoint_verified
from wan_va.modules.future_teacher import ActionFutureTeacher, load_future_teacher, save_future_teacher
from wan_va.train import Trainer, future_teacher_chunk_targets, validated_future_teacher


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class _RecordingTeacher(nn.Module):
    num_tokens, hidden_dim = 2, 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []

    def encode_future(self, features):
        self.calls.append(features.detach().clone())
        return (features.mean() * self.weight).expand(1, self.num_tokens, self.hidden_dim)


@pytest.mark.parametrize("chunk,expected,valid", [
    (1, [[1], [2], [3], [4], [5], [6]], [False, True, True, True, True, True, True]),
    (2, [[1], [2, 3], [4, 5], [6]], [True, True, True, True]),
    (4, [[1, 2, 3], [4, 5, 6]], [True, True]),
])
def test_future_targets_use_exact_chunks_exclude_initial_observation_and_are_detached(chunk, expected, valid):
    teacher = _RecordingTeacher()
    features = torch.arange(7.).view(1, 7, 1, 1).expand(1, 7, 4, 8).clone().requires_grad_(True)
    actions_mask = torch.ones(1, 30, 7, 16, 1, dtype=torch.bool)
    targets, present = future_teacher_chunk_targets(teacher, features, chunk, actions_mask)
    assert [call[0, :, 0, 0].tolist() for call in teacher.calls] == expected
    assert present.tolist() == valid
    assert targets.shape == (len(valid), 2, 3)
    assert not targets.requires_grad and targets.grad_fn is None
    assert teacher.weight.grad is None and features.grad is None
    if chunk == 1:
        assert targets[0].count_nonzero() == 0


def test_entirely_masked_chunks_do_not_run_teacher_or_contribute_targets():
    teacher = _RecordingTeacher()
    features = torch.arange(7.).view(1, 7, 1, 1).expand(1, 7, 4, 8)
    mask = torch.zeros(1, 30, 7, 16, 1, dtype=torch.bool)
    mask[:, 0, 0] = True  # Initial observation alone cannot validate first chunk.
    mask[:, 29, 4, 15, 0] = True  # One active action suffices for its future chunk.
    targets, valid = future_teacher_chunk_targets(teacher, features, 2, mask)
    assert valid.tolist() == [False, False, True, False]
    assert [call[0, :, 0, 0].tolist() for call in teacher.calls] == [[4, 5]]
    assert targets[~valid].count_nonzero() == 0
    with pytest.raises(ValueError, match="same latent anchors"):
        future_teacher_chunk_targets(teacher, features, 2, mask[:, :, :-1])


def _spec():
    return dict(rgb_encoder_id="test/rgb", rgb_encoder_revision="immutable", rgb_preprocessing={"rgb": True})


def _teacher_config(**overrides):
    cfg = dict(future_teacher_loss_weight=.1, future_teacher_checkpoint="unused", enable_rgb_transition=True,
               rgb_transition_config=dict(readout_mode="state", readout_num_tokens=4, hidden_dim=12, feature_dim=8, max_patches=4),
               max_frame_chunk_size=4, action_dim=6, action_per_frame=2, smoke_test=False)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


@pytest.fixture
def artifact(tmp_path):
    teacher = ActionFutureTeacher(feature_dim=8, hidden_dim=12, num_tokens=4, num_heads=3,
                                  encoder_layers=1, action_layers=1, action_dim=6,
                                  max_future_steps=4, max_patches=4, max_action_steps=8)
    path = tmp_path / "teacher"
    save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=1, smoke_test=True)
    return path


@pytest.mark.parametrize("overrides,error", [
    ({"future_teacher_loss_weight": float("nan")}, "finite"),
    ({"future_teacher_loss_weight": -1}, "nonnegative"),
    ({"future_teacher_loss_weight": 0}, "positive loss weight"),
    ({"future_teacher_checkpoint": None}, "ZERO_WAM_FUTURE_TEACHER"),
    ({"enable_rgb_transition": False}, "state-conditioned"),
    ({"rgb_transition_config": {"readout_mode": "static"}}, "state-conditioned"),
])
def test_invalid_teacher_experiment_rejected_before_loading_any_model(overrides, error, monkeypatch):
    def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid experiment should fail before artifact/model loading")
    monkeypatch.setattr("wan_va.modules.future_teacher.load_future_teacher", forbidden_load)
    with pytest.raises(ValueError, match=error):
        validated_future_teacher(_teacher_config(**overrides), _spec())


def test_baseline_needs_no_teacher_and_formal_policy_rejects_smoke_artifact(artifact):
    assert validated_future_teacher(_teacher_config(future_teacher_loss_weight=0, future_teacher_checkpoint=None), {}) == (None, None)
    cfg = _teacher_config(future_teacher_checkpoint=str(artifact))
    with pytest.raises(ValueError, match="Smoke-only"):
        validated_future_teacher(cfg, _spec())
    cfg.smoke_test = True
    teacher, metadata = validated_future_teacher(cfg, _spec())
    assert metadata["smoke_test"] and not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    cfg.max_frame_chunk_size = 5
    with pytest.raises(ValueError, match="dimensions/horizon"):
        validated_future_teacher(cfg, _spec())
    cfg.max_frame_chunk_size = 4
    cfg.rgb_transition_config["readout_num_tokens"] = 5
    with pytest.raises(ValueError, match="dimensions/horizon"):
        validated_future_teacher(cfg, _spec())


def test_run_preflight_rejects_teacher_before_allocating_large_policy(artifact, monkeypatch):
    cfg = _teacher_config(future_teacher_checkpoint=str(artifact), rank=0, local_rank=0, world_size=1,
                          dataset_sources=[{"config": SimpleNamespace()}])
    monkeypatch.setattr(train_module, "build_training_config", lambda args: cfg)
    monkeypatch.setattr(train_module, "init_distributed", lambda *args: None)
    monkeypatch.setattr("wan_va.dataset.rgb_feature_cache.validate_rgb_feature_cache",
                        lambda config: {"feature_spec": _spec()})
    def forbidden_trainer(config):
        pytest.fail("Large policy allocation happened before teacher preflight")
    monkeypatch.setattr(train_module, "Trainer", forbidden_trainer)
    with pytest.raises(RuntimeError, match="preflight failed.*Smoke-only"):
        train_module.run(SimpleNamespace(check_config=False, config_name="teacher_test", datasets="robotwin:1"))


@pytest.mark.parametrize("field,value", [
    ("action_dim", 7),
    ("action_per_frame", 3),
    ("max_patches", 5),
    ("feature_dim", 9),
    ("hidden_dim", 16),
    ("readout_num_tokens", 16),
])
def test_mismatched_teacher_capacity_or_action_definition_rejected_before_policy(artifact, field, value):
    cfg = _teacher_config(future_teacher_checkpoint=str(artifact), smoke_test=True)
    if field in {"action_dim", "action_per_frame"}:
        setattr(cfg, field, value)
    else:
        cfg.rgb_transition_config[field] = value
    with pytest.raises(ValueError, match="dimensions/horizon"):
        validated_future_teacher(cfg, _spec())


@pytest.fixture
def provenance_artifact(artifact, tmp_path):
    teacher, _ = load_future_teacher(artifact, allow_smoke=True)

    def create(training):
        path = tmp_path / "teacher_with_training_provenance"
        # Write a valid checksummed artifact so errors exercise provenance
        # validation rather than accidental metadata-checksum corruption.
        save_future_teacher(teacher, path, feature_spec=_spec(), successful_steps=1,
                            smoke_test=True, training=training)
        return path

    return create


@pytest.mark.parametrize("overrides,missing,error", [
    ({}, "included_tasks", "task split"),
    ({}, "heldout_tasks_used", "task split"),
    ({"included_tasks": ["train_a", "heldout_task"]}, None, "task split"),
    ({"included_tasks": ["train_a"]}, None, "task split"),
    ({"included_tasks": ["train_a", "train_b", "train_a"]}, None, "task split"),
    ({"heldout_tasks_used": True}, None, "task split"),
    ({"heldout_tasks_used": 0}, None, "task split"),
    ({"num_steps": 2}, None, "step provenance"),
    ({"num_steps": True}, None, "step provenance"),
    ({}, "num_steps", "step provenance"),
])
def test_teacher_training_provenance_rejects_missing_wrong_or_inconsistent_claims(
        provenance_artifact, overrides, missing, error):
    training = dict(included_tasks=["train_a", "train_b"], heldout_tasks_used=False, num_steps=1)
    training.update(overrides)
    if missing:
        training.pop(missing)
    path = provenance_artifact(training)
    cfg = _teacher_config(future_teacher_checkpoint=str(path), smoke_test=True,
                          included_task_names=["train_a", "train_b"])
    with pytest.raises(ValueError, match=error):
        validated_future_teacher(cfg, _spec())


@pytest.mark.parametrize("tasks", [["train_a", "train_b"], ["train_b", "train_a"]])
def test_teacher_training_provenance_accepts_matching_task_set_regardless_of_order(provenance_artifact, tasks):
    path = provenance_artifact(dict(included_tasks=tasks, heldout_tasks_used=False, num_steps=1))
    cfg = _teacher_config(future_teacher_checkpoint=str(path), smoke_test=True,
                          included_task_names=["train_a", "train_b"])
    teacher, metadata = validated_future_teacher(cfg, _spec())
    assert not teacher.training
    assert metadata["training"]["included_tasks"] == tasks
    assert metadata["training"]["num_steps"] == metadata["successful_steps"] == 1


def test_teacher_training_provenance_error_stops_run_before_policy_allocation(provenance_artifact, monkeypatch):
    path = provenance_artifact(dict(included_tasks=["train_a", "heldout_task"],
                                    heldout_tasks_used=False, num_steps=1))
    cfg = _teacher_config(future_teacher_checkpoint=str(path), smoke_test=True,
                          included_task_names=["train_a", "train_b"], rank=0, local_rank=0, world_size=1,
                          dataset_sources=[{"config": SimpleNamespace()}])
    monkeypatch.setattr(train_module, "build_training_config", lambda args: cfg)
    monkeypatch.setattr(train_module, "init_distributed", lambda *args: None)
    monkeypatch.setattr("wan_va.dataset.rgb_feature_cache.validate_rgb_feature_cache",
                        lambda config: {"feature_spec": _spec()})
    def forbidden_trainer(config):
        pytest.fail("Policy allocation happened before teacher task provenance was checked")
    monkeypatch.setattr(train_module, "Trainer", forbidden_trainer)
    with pytest.raises(RuntimeError, match="preflight failed.*task split"):
        train_module.run(SimpleNamespace(check_config=False, config_name="teacher_test", datasets="robotwin:1"))


class _Scheduler:
    num_train_timesteps = 1000
    timesteps = torch.arange(1000, dtype=torch.float32)

    def training_weight(self, timesteps):
        return torch.ones_like(timesteps)

    def add_noise(self, latent, noise, timesteps, t_dim):
        return latent

    def training_target(self, latent, noise, timesteps):
        return torch.zeros_like(latent)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.ones(()))
        self.rgb_transition = nn.Module()
        self.rgb_transition.encoder = nn.Parameter(torch.ones(()))
        self.rgb_transition.student = nn.Parameter(torch.ones(()))
        self.rgb_transition.gate = nn.Parameter(torch.zeros(()))
        self.config = {"rgb_transition_config": {"readout_mode": "state"}}

    def set_requires_gradient_sync(self, enabled):
        pass

    def forward(self, inputs, train_mode):
        scale = inputs["scale"]
        prediction = (self.backbone * scale).expand(1, 2, 1)
        losses = dict(total=(self.rgb_transition.encoder * scale).square(),
                      teacher_alignment=(self.rgb_transition.student * scale).square())
        return {"base_pred": (prediction, prediction), "rgb_losses": losses}


def _trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.step, trainer.enable_mcp, trainer.enable_rgb_transition = 0, False, True
    trainer.patch_size = (1, 1, 1)
    trainer.gradient_accumulation_steps = 2
    trainer.config = SimpleNamespace(rank=1, enable_wandb=False, video_loss_reweight=False,
        action_loss_reweight=False, rgb_loss_weight=.2, future_teacher_loss_weight=.3,
        max_norm=1e6, skip_step_grad_norm_multiplier=0, sync_each_microbatch=True,
        frame_chunk_size=2, max_frame_chunk_size=4, noisy_img_prob=0., noisy_cond_min_timestep_bd=0.,
        noisy_cond_max_timestep_bd=1., drop_icl=0., icl_rope_h=4, attn_window=4,
        rgb_transition_config={"readout_mode": "state"})
    trainer.transformer = _Policy()
    trainer.optimizer = torch.optim.SGD(trainer.transformer.parameters(), lr=.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=lambda _: 1)
    trainer.train_scheduler_latent = _Scheduler()
    trainer.train_scheduler_action = _Scheduler()
    return trainer


def _loss_input(scale):
    return dict(scale=torch.tensor(float(scale)),
                latent_dict=dict(targets=torch.zeros(1, 1, 2, 1, 1), timesteps=torch.zeros(1, 2)),
                action_dict=dict(targets=torch.zeros(1, 1, 2, 1, 1), timesteps=torch.zeros(1, 2),
                                 actions_mask=torch.ones(1, 1, 2, 1, 1)))


@pytest.mark.parametrize("rgb_weight", [0., .2])
def test_teacher_loss_has_independent_weight_and_accumulation_exactly_once(rgb_weight):
    trainer = _trainer()
    trainer.config.rgb_loss_weight = rgb_weight
    trainer._prepare_input_dict = lambda batch: batch
    first = trainer._train_step(_loss_input(1), 0)
    torch.testing.assert_close(trainer.transformer.rgb_transition.student.grad, torch.tensor(.3))
    second = trainer._train_step(_loss_input(3), 1)
    assert not first["should_log"] and second["should_log"]
    assert not second["optimizer_step_skipped"]
    # Mean(scale**2)=5, so student gradient=.3*2*5=3; no RGB weight coupling.
    torch.testing.assert_close(trainer.transformer.rgb_transition.student, torch.tensor(.97))
    torch.testing.assert_close(trainer.transformer.rgb_transition.encoder, torch.tensor(1 - .01 * rgb_weight * 10))
    torch.testing.assert_close(first["future_teacher_loss"] + second["future_teacher_loss"], torch.tensor(1.5))
    torch.testing.assert_close(first["rgb_components"]["teacher_alignment"] + second["rgb_components"]["teacher_alignment"], torch.tensor(5.))


@pytest.mark.parametrize("drop_icl", [0., 1.])
def test_prepare_input_detaches_teacher_and_dropped_icl_never_reads_teacher_or_rgb(drop_icl):
    trainer = _trainer()
    trainer.config.drop_icl = drop_icl
    trainer.future_teacher = _RecordingTeacher()
    batch = dict(latents=torch.randn(1, 1, 4, 1, 1, requires_grad=True), actions=torch.ones(1, 1, 4, 1, 1),
                 actions_mask=torch.ones(1, 1, 4, 1, 1), text_emb=torch.ones(1, 2, 4))
    if not drop_icl:
        batch.update(icl_latents=torch.ones(1, 1, 2, 1, 1), icl_text_emb=torch.ones(1, 2, 4),
                     rgb_human_features=torch.randn(1, 2, 3, 4, 8),
                     rgb_robot_features=torch.randn(1, 4, 4, 8, requires_grad=True))
    prepared = trainer._prepare_input_dict(batch)
    if drop_icl:
        assert trainer.future_teacher.calls == []
        assert prepared["icl_latent_dict"] is None
        assert "readout_teacher_targets" not in prepared and "rgb_transition" not in prepared
    else:
        assert prepared["readout_teacher_targets"].shape == (2, 2, 3)
        assert prepared["readout_teacher_valid"].tolist() == [True, True]
        assert not prepared["readout_teacher_targets"].requires_grad
        assert not prepared["readout_observation_latents"].requires_grad
        torch.testing.assert_close(prepared["readout_observation_latents"], batch["latents"])


def test_verified_checkpoint_records_teacher_provenance_without_embedding_teacher(tmp_path):
    trainer = _trainer()
    trainer.config.rank, trainer.step, trainer.save_dir = 0, 2, tmp_path
    trainer.future_teacher = _RecordingTeacher()
    trainer.training_provenance = dict(smoke_test=True, future_teacher_loss_weight=.3,
        future_teacher={"weights_sha256": "teacher-sha", "successful_steps": 2, "smoke_test": True,
                        "feature_spec": _spec()})
    save_checkpoint_verified(trainer)
    manifest = json.loads((tmp_path / "checkpoint_step_2/COMPLETE.json").read_text())
    assert manifest["weights_only"] and manifest["step"] == 2
    assert manifest["training_provenance"] == trainer.training_provenance
    assert manifest["tensor_count"] == len(trainer.transformer.state_dict())
    for record in manifest["files"]:
        path = tmp_path / "checkpoint_step_2/transformer" / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    from safetensors.torch import load_file
    weights = load_file(str(tmp_path / "checkpoint_step_2/transformer/diffusion_pytorch_model.safetensors"))
    assert not any("future_teacher" in key for key in weights)
