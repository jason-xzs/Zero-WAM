"""Verify controlled recipes and the bounded, portable launch interfaces."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from wan_va.configs import VA_CONFIGS
from wan_va.configs.va_robotwin_subset12_rgb_readout_cfg import (
    READOUT_CONFIGS, READOUT_VARIANTS, make_rgb_readout_config,
)
from wan_va.train import build_training_config
from test_rgb_training import _args, config_data_root


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", READOUT_VARIANTS)
@pytest.mark.parametrize("platform,world,accum", [("b300", 1, 8), ("train", 4, 2)])
def test_all_recipes_keep_data_training_and_explicit_readout_budgets(variant, platform, world, accum):
    key = f"robotwin_subset12_rgb_readout_{variant}_{platform}"
    cfg = VA_CONFIGS[key]
    baseline = VA_CONFIGS[f"robotwin_subset12_rgb_{platform}"]
    assert key in READOUT_CONFIGS
    assert (cfg.expected_world_size, cfg.batch_size, cfg.gradient_accumulation_steps) == (world, 1, accum)
    assert cfg.expected_effective_batch_size == 8
    assert (cfg.num_steps, cfg.save_interval) == (15000, 5000)
    assert (cfg.learning_rate, cfg.warmup_steps) == (1e-4, 200)
    assert cfg.param_dtype == torch.bfloat16 and cfg.rgb_init_seed == 42
    assert cfg.enable_mcp and cfg.num_mcp_modules == 4
    assert cfg.expected_num_train_tasks == 12
    assert cfg.included_task_names == baseline.included_task_names
    assert cfg.excluded_task_names == baseline.excluded_task_names
    assert cfg.enable_wandb == (platform == "train")
    rgb = cfg.rgb_transition_config
    assert rgb["readout_mode"] == ("state" if variant == "teacher" else variant)
    assert (rgb["readout_num_tokens"], rgb["readout_global_tokens"]) == ((16, 4) if variant == "global_local" else (32, 8))
    assert cfg.rgb_loss_weight == baseline.rgb_loss_weight == 0.1
    assert {k: rgb[k] for k in baseline.rgb_transition_config} == baseline.rgb_transition_config
    assert cfg.future_teacher_loss_weight == (0.1 if variant == "teacher" else 0.0)
    if variant != "teacher":
        assert cfg.future_teacher_checkpoint is None


def test_factory_isolated_from_baseline_registry_and_other_variants(monkeypatch):
    originals = {key: deepcopy(value) for key, value in VA_CONFIGS.items()}
    monkeypatch.setenv("ZERO_WAM_FUTURE_TEACHER", "/workspace/trained teacher")
    teacher = make_rgb_readout_config("teacher", "b300")
    assert teacher.future_teacher_checkpoint == "/workspace/trained teacher"
    teacher.rgb_transition_config["readout_num_tokens"] = 3
    teacher.included_task_names.clear()
    assert VA_CONFIGS == originals
    assert make_rgb_readout_config("state", "b300").future_teacher_checkpoint is None


@pytest.mark.parametrize("variant,platform", [("typo", "b300"), ("state", "typo")])
def test_factory_rejects_unknown_choices(variant, platform):
    with pytest.raises(ValueError, match="Unknown RGB readout"):
        make_rgb_readout_config(variant, platform)


@pytest.mark.parametrize("variant", READOUT_VARIANTS)
@pytest.mark.parametrize("tokens,global_tokens", [(16, 4), (32, 8)])
def test_factory_supports_explicit_matched_budget_controls(monkeypatch, variant, tokens, global_tokens):
    monkeypatch.setenv("ZERO_WAM_READOUT_TOKENS", str(tokens))
    monkeypatch.setenv("ZERO_WAM_READOUT_GLOBAL_TOKENS", str(global_tokens))
    for platform in ("b300", "train"):
        rgb = make_rgb_readout_config(variant, platform).rgb_transition_config
        assert (rgb["readout_num_tokens"], rgb["readout_global_tokens"]) == (tokens, global_tokens)


@pytest.mark.parametrize("tokens,global_tokens", [("0", "4"), ("16", "0"), ("16", "16"),
                                                 ("16", "17"), ("bad", "4"), ("16", "-1"),
                                                 ("1.5", "1"), ("", "4")])
def test_factory_rejects_invalid_budget_before_model_construction(monkeypatch, tokens, global_tokens):
    monkeypatch.setenv("ZERO_WAM_READOUT_TOKENS", tokens)
    monkeypatch.setenv("ZERO_WAM_READOUT_GLOBAL_TOKENS", global_tokens)
    with pytest.raises(ValueError, match="ZERO_WAM_READOUT"):
        make_rgb_readout_config("global_local", "b300")


@pytest.mark.parametrize("variant", ["static", "state", "global_local"])
def test_b300_readout_builds_and_enforces_world_size_and_global_batch(config_data_root, monkeypatch, variant):
    key = f"robotwin_subset12_rgb_readout_{variant}_b300"
    cfg = build_training_config(_args(config_name=key, dataset_path=config_data_root))
    assert cfg.smoke_test and cfg.num_steps == 2
    assert cfg.world_size * cfg.batch_size * cfg.gradient_accumulation_steps == 8
    assert cfg.dataset_sources[0]["config"].rgb_transition_config == cfg.rgb_transition_config
    with pytest.raises(ValueError, match="requires global batch 8"):
        build_training_config(_args(config_name=key, dataset_path=config_data_root,
                                    gradient_accumulation_steps=2))
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="requires 1 GPUs"):
        build_training_config(_args(config_name=key, dataset_path=config_data_root))


@pytest.fixture
def readout_launch_env(tmp_path):
    capture = tmp_path / "capture.json"
    executable = tmp_path / "python with spaces"
    executable.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'if sys.argv[1] == "-c":\n'
        '    r = json.loads(Path(sys.argv[3]).read_text())\n'
        '    assert r["smoke_subset"] and r["pairs"] == r["tasks"] == 12\n'
        '    print(",".join(r["sample_ids"]))\n'
        'else:\n'
        '    keys = ("NGPU", "CONFIG_NAME", "DATASETS", "CUDA_VISIBLE_DEVICES", "LOG_RANK", '
        '"ZERO_WAM_CPU_OFFLOAD", "ZERO_WAM_ACTIVATION_OFFLOAD", '
        '"ZERO_WAM_SYNC_EACH_MICROBATCH", "ZERO_WAM_GPU_MEMORY_FRACTION", '
        '"ZERO_WAM_FUTURE_TEACHER", "RGB_READOUT_VARIANT", '
        '"ZERO_WAM_READOUT_TOKENS", "ZERO_WAM_READOUT_GLOBAL_TOKENS")\n'
        '    Path(os.environ["LAUNCH_CAPTURE"]).write_text(json.dumps('
        '{"args":sys.argv[1:], "env":{k:os.environ.get(k) for k in keys}}))\n'
    )
    executable.chmod(0o755)
    report = tmp_path / "smoke report.json"
    report.write_text(json.dumps({"smoke_subset": True, "pairs": 12, "tasks": 12,
                                  "sample_ids": [f"pair_{i}" for i in range(12)]}))
    env = {**os.environ, "PYTHON_BIN": str(executable), "LAUNCH_CAPTURE": str(capture),
           "MODEL_PATH": "/workspace/model with spaces", "HUMAN_GEN_ROOT": "/workspace/data",
           "ZERO_WAM_RGB_FEATURE_ROOT": "/workspace/rgb", "ZERO_WAM_SAVE_ROOT": "/workspace/new run",
           "CUDA_VISIBLE_DEVICES": "4", "NGPU": "4", "CONFIG_NAME": "robotwin_train",
           "DATASETS": "agibot:1.0", "LOG_RANK": "3", "ZERO_WAM_CPU_OFFLOAD": "1",
           "ZERO_WAM_ACTIVATION_OFFLOAD": "1", "ZERO_WAM_SYNC_EACH_MICROBATCH": "0",
           "ZERO_WAM_GPU_MEMORY_FRACTION": "0.1", "RGB_SMOKE_REPORT": str(report)}
    # Tests choose their own baseline; a branch may intentionally default to
    # teacher and require an artifact before reaching the behavior under test.
    env["RGB_READOUT_VARIANT"] = "static"
    env.pop("ZERO_WAM_FUTURE_TEACHER", None)
    env.pop("RGB_SMOKE_STEPS", None)
    env.pop("ZERO_WAM_READOUT_TOKENS", None)
    env.pop("ZERO_WAM_READOUT_GLOBAL_TOKENS", None)
    return env, capture


@pytest.mark.parametrize("variant", READOUT_VARIANTS)
@pytest.mark.parametrize("check_only", [False, True])
def test_b300_launcher_routes_variant_and_preserves_single_gpu_recipe(readout_launch_env, variant, check_only):
    env, capture = readout_launch_env
    env["RGB_READOUT_VARIANT"] = variant
    if variant == "teacher":
        env["ZERO_WAM_FUTURE_TEACHER"] = "/workspace/teacher artifact"
    extra = ["--save-root", "/workspace/output with spaces"]
    if check_only:
        extra += ["--check-config"]
    subprocess.run(["bash", str(ROOT / "script/train_rgb_readout_b300.sh"), *extra],
                   env=env, check=True, capture_output=True, text=True)
    record = json.loads(capture.read_text())
    args, child = record["args"], record["env"]
    assert args[:2] == ["-m", "wan_va.train" if check_only else "torch.distributed.run"]
    if not check_only:
        assert "--nproc_per_node=1" in args
    assert args[args.index("--config-name") + 1] == f"robotwin_subset12_rgb_readout_{variant}_b300"
    assert args[args.index("--datasets") + 1] == "robotwin:1.0"
    assert args[-len(extra):] == extra
    assert child["NGPU"] == "1" and child["CUDA_VISIBLE_DEVICES"] == "4"
    assert child["LOG_RANK"] == "0"
    assert child["ZERO_WAM_CPU_OFFLOAD"] == child["ZERO_WAM_ACTIVATION_OFFLOAD"] == "0"
    assert child["ZERO_WAM_SYNC_EACH_MICROBATCH"] is None
    assert child["ZERO_WAM_GPU_MEMORY_FRACTION"] is None
    assert (child["ZERO_WAM_READOUT_TOKENS"], child["ZERO_WAM_READOUT_GLOBAL_TOKENS"]) == (
        ("16", "4") if variant == "global_local" else ("32", "8"))
    if variant == "teacher":
        assert child["ZERO_WAM_FUTURE_TEACHER"] == "/workspace/teacher artifact"


@pytest.mark.parametrize("missing", ["MODEL_PATH", "HUMAN_GEN_ROOT", "ZERO_WAM_RGB_FEATURE_ROOT", "ZERO_WAM_SAVE_ROOT"])
def test_b300_launcher_requires_all_target_roots(readout_launch_env, missing):
    env, capture = readout_launch_env
    env.pop(missing)
    result = subprocess.run(["bash", str(ROOT / "script/train_rgb_readout_b300.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and missing in result.stderr
    assert not capture.exists()


def test_cpu_config_route_preserves_explicitly_hidden_gpus(readout_launch_env):
    env, capture = readout_launch_env
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(["bash", str(ROOT / "script/train_rgb_readout_b300.sh"), "--check-config"],
                   env=env, check=True, capture_output=True, text=True)
    record = json.loads(capture.read_text())
    assert record["args"][:2] == ["-m", "wan_va.train"]
    assert record["env"]["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("script", ["train_rgb_readout_b300.sh", "smoke_rgb_readout.sh"])
@pytest.mark.parametrize("variant,tokens,global_tokens", [("global_local", "32", "8"),
                                                         ("static", "16", "4"), ("state", "16", "4")])
def test_launchers_preserve_matched_budget_overrides(readout_launch_env, script, variant, tokens, global_tokens):
    env, capture = readout_launch_env
    env.update(RGB_READOUT_VARIANT=variant, ZERO_WAM_READOUT_TOKENS=tokens,
               ZERO_WAM_READOUT_GLOBAL_TOKENS=global_tokens)
    subprocess.run(["bash", str(ROOT / "script" / script)], env=env, check=True,
                   capture_output=True, text=True)
    child = json.loads(capture.read_text())["env"]
    assert (child["ZERO_WAM_READOUT_TOKENS"], child["ZERO_WAM_READOUT_GLOBAL_TOKENS"]) == (tokens, global_tokens)


@pytest.mark.parametrize("script", ["train_rgb_readout_b300.sh", "smoke_rgb_readout.sh"])
@pytest.mark.parametrize("tokens,global_tokens", [("0", "4"), ("16", "16"), ("16", "-1"), ("bad", "4"), ("", "4")])
def test_launchers_reject_invalid_budgets_before_spawning(readout_launch_env, script, tokens, global_tokens):
    env, capture = readout_launch_env
    env.update(ZERO_WAM_READOUT_TOKENS=tokens, ZERO_WAM_READOUT_GLOBAL_TOKENS=global_tokens)
    result = subprocess.run(["bash", str(ROOT / "script" / script)], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "ZERO_WAM_READOUT" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("script", ["train_rgb_readout_b300.sh", "smoke_rgb_readout.sh"])
@pytest.mark.parametrize("variant", ["teacher", "bad_variant"])
def test_launchers_reject_missing_teacher_and_unknown_variant(readout_launch_env, script, variant):
    env, capture = readout_launch_env
    env["RGB_READOUT_VARIANT"] = variant
    result = subprocess.run(["bash", str(ROOT / "script" / script)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert ("ZERO_WAM_FUTURE_TEACHER" if variant == "teacher" else "Invalid RGB_READOUT_VARIANT") in result.stderr
    assert not capture.exists()


def test_smoke_launcher_defaults_to_six_steps_four_gpus_and_explicit_samples(readout_launch_env):
    env, capture = readout_launch_env
    env["RGB_READOUT_VARIANT"] = "global_local"
    subprocess.run(["bash", str(ROOT / "script/smoke_rgb_readout.sh")],
                   env=env, check=True, capture_output=True, text=True)
    record = json.loads(capture.read_text())
    args = record["args"]
    assert "--nproc_per_node=4" in args
    assert args[args.index("--config-name") + 1] == "robotwin_subset12_rgb_readout_global_local_train"
    assert "--smoke-test" in args and "--disable-wandb" in args
    assert args[args.index("--num-steps") + 1] == args[args.index("--save-interval") + 1] == "6"
    assert args[args.index("--rgb-smoke-sample-ids") + 1].split(",") == [f"pair_{i}" for i in range(12)]


@pytest.mark.parametrize("steps", ["0", "21", "15000", "-1", "1.5", "not_a_number"])
def test_smoke_launcher_refuses_unbounded_steps_before_loading_data(readout_launch_env, steps):
    env, capture = readout_launch_env
    env["RGB_SMOKE_STEPS"] = steps
    result = subprocess.run(["bash", str(ROOT / "script/smoke_rgb_readout.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "from 1 to 20" in result.stderr
    assert not capture.exists()
